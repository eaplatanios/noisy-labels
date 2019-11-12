// Copyright 2019, Emmanouil Antonios Platanios. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

import Foundation
import TensorFlow

public struct Model<Predictor: Graphs.Predictor, Optimizer: TensorFlow.Optimizer>
where Optimizer.Model == Predictor {
  public let useSoftPredictions: Bool
  public let entropyWeight: Float

  public let randomSeed: Int64
  public let batchSize: Int
  public let useWarmStarting: Bool
  public let mStepCount: Int
  public let emStepCount: Int
  public let mStepLogCount: Int?
  public let emStepCallback: (Model) -> Void
  public let verbose: Bool

  public var predictor: Predictor
  public var optimizer: Optimizer
  public var eStepAccumulator: Tensor<Float> // [NodeCount, ClassCount]
  public var expectedLabels: Tensor<Float>   // [NodeCount, ClassCount]

  public init(
    predictor: Predictor,
    optimizer: Optimizer,
    useSoftPredictions: Bool,
    entropyWeight: Float,
    randomSeed: Int64,
    batchSize: Int = 128,
    useWarmStarting: Bool = true,
    mStepCount: Int = 1000,
    emStepCount: Int = 100,
    mStepLogCount: Int? = 100,
    emStepCallback: @escaping (Model) -> Void = { _ in },
    verbose: Bool = false
  ) {
    self.predictor = predictor
    self.optimizer = optimizer
    self.useSoftPredictions = useSoftPredictions
    self.entropyWeight = entropyWeight
    self.randomSeed = randomSeed
    self.batchSize = batchSize
    self.useWarmStarting = useWarmStarting
    self.mStepCount = mStepCount
    self.emStepCount = emStepCount
    self.mStepLogCount = mStepLogCount
    self.emStepCallback = emStepCallback
    self.verbose = verbose
    self.eStepAccumulator = Tensor<Float>(zeros: [predictor.nodeCount, predictor.classCount])
    self.expectedLabels = Tensor<Float>(zeros: [predictor.nodeCount, predictor.classCount])
  }

  public mutating func train(using data: Data) {
    let labeledData = Dataset(elements: data.labeledData)
    let unlabeledData = Dataset(elements: data.unlabeledData)
    let allUnlabeledData = Dataset(elements: data.allUnlabeledData)

    for emStep in 0..<emStepCount {
      // E-Step
      if verbose { logger.info("Iteration \(emStep) - Running E-Step") }
      performEStep(labeledData: labeledData, unlabeledData: unlabeledData)

      // M-Step
      if verbose { logger.info("Iteration \(emStep) - Running M-Step") }
      performMStep(data: allUnlabeledData, emStep: emStep)

      // TODO: Add support for marginal likelihood fine-tuning.
      // How do we handle labeled data in that case?

      emStepCallback(self)
    }
  }

  private mutating func performEStep(
    labeledData: Dataset<LabeledData>,
    unlabeledData: Dataset<UnlabeledData>
  ) {
    eStepAccumulator = Tensor<Float>(zeros: [predictor.nodeCount, predictor.classCount])

    // Set the labeled node labels to their true labels.
    for batch in labeledData.batched(batchSize) {
      eStepAccumulator = _Raw.tensorScatterAdd(
        eStepAccumulator,
        indices: batch.nodeIndices.expandingShape(at: -1),
        updates: Tensor<Float>(oneHotAtIndices: batch.nodeLabels, depth: predictor.classCount))
    }

    // Compute expectations for the labels of the unlabeled nodes.
    for batch in unlabeledData.batched(batchSize) {
      let qualities = predictor.qualities(batch.nodeFeatures, batch.neighborFeatures)
      let qLog = logSoftmax(qualities, alongAxis: 2)                                                // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
      let neighborValues = expectedLabels.gathering(atIndices: batch.neighborIndices)               // [BatchSize, MaxNeighborCount, ClassCount]
      let yHat = useSoftPredictions ?
        neighborValues :
        Tensor<Float>(
          oneHotAtIndices: neighborValues.argmax(squeezingAxis: -1),
          depth: predictor.classCount)                                                              // [BatchSize, MaxNeighborCount, ClassCount]
      let qLogYHat = (qLog * yHat.expandingShape(at: 2)).sum(squeezingAxes: 2)                      // [BatchSize, MaxNeighborCount, ClassCount]
      eStepAccumulator = _Raw.tensorScatterAdd(
        eStepAccumulator,
        indices: batch.nodeIndices.expandingShape(at: -1),
        updates: (qLogYHat * batch.neighborMask.expandingShape(at: -1)).sum(squeezingAxes: 1))      // [BatchSize, ClassCount]
    }
    expectedLabels = exp(eStepAccumulator - eStepAccumulator.logSumExp(alongAxes: -1))
  }

  private mutating func performMStep(data: Dataset<UnlabeledData>, emStep: Int) {
    if !useWarmStarting { predictor.reset() }
    let classCount = predictor.classCount
    var accumulatedNLL = Float(0.0)
    var accumulatedSteps = 0
    var dataIterator = data.repeated()
      .shuffled(sampleCount: 10000, randomSeed: randomSeed &+ Int64(emStep))
      .batched(batchSize)
      .prefetched(count: 10)
      .makeIterator()
    for mStep in 0..<mStepCount {
      let batch = dataIterator.next()!
      let (negativeLogLikelihood, gradient) = predictor.valueWithGradient {
        [expectedLabels, useSoftPredictions, entropyWeight]
        predictor -> Tensor<Float> in
        let predictions = predictor(batch.nodeFeatures, batch.neighborFeatures)
        let hLog = predictions.labelProbabilities
        let qLog = predictions.qualities
        let neighborValues = expectedLabels.gathering(atIndices: batch.neighborIndices)             // [BatchSize, MaxNeighborCount, ClassCount]
        let yHat = useSoftPredictions ?
          neighborValues :
          Tensor<Float>(
            oneHotAtIndices: neighborValues.argmax(squeezingAxis: -1),
            depth: classCount)                                                                      // [BatchSize, MaxNeighborCount, ClassCount]
        var qLogYHat = (qLog * yHat.expandingShape(at: 2)).sum(squeezingAxes: 2)                    // [BatchSize, MaxNeighborCount, ClassCount]
        qLogYHat = (qLogYHat * batch.neighborMask.expandingShape(at: -1)).sum(squeezingAxes: 1)     // [BatchSize, ClassCount]
        let yExpected = expectedLabels.gathering(atIndices: batch.nodeIndices)
        return entropyWeight * (exp(hLog) * hLog).sum() -
          (yExpected * hLog).sum() -
          (yExpected * qLogYHat).sum() + predictions.regularizationTerm
      }
      optimizer.update(&predictor, along: gradient)
      accumulatedNLL += negativeLogLikelihood.scalarized()
      accumulatedSteps += 1
      if verbose {
        if let logSteps = mStepLogCount, mStep % logSteps == 0 || mStep == mStepCount - 1 {
          let nll = accumulatedNLL / Float(accumulatedSteps)
          let message = "M-Step \(String(format: "%5d", mStep)) | " +
            "Negative Log-Likelihood: \(String(format: "%.8f", nll))"
          logger.info("\(message)")
          accumulatedNLL = 0.0
          accumulatedSteps = 0
        }
      }
    }
  }
}
