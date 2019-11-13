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
  public let eStepCount: Int
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
    eStepCount: Int = 10,
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
    self.eStepCount = eStepCount
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
    let unlabeledNodeIndices = Dataset(elements: data.unlabeledNodeIndices)

    if verbose { logger.info("Initialization") }
    initialize(using: data)
    // performMStep(data: labeledData, emStep: 0)
    // for _ in 0..<eStepCount {
    //   performEStep(
    //     labeledData: labeledData,
    //     unlabeledData: unlabeledData,
    //     unlabeledNodeIndices: unlabeledNodeIndices)
    //   emStepCallback(self)
    // }

    for emStep in 0..<emStepCount {
      // M-Step
      if verbose { logger.info("Iteration \(emStep) - Running M-Step") }
      performMStep(data: allUnlabeledData, emStep: emStep)

      // E-Step
      if verbose { logger.info("Iteration \(emStep) - Running E-Step") }
      for _ in 0..<eStepCount {
        performEStep(
          labeledData: labeledData,
          unlabeledData: unlabeledData,
          unlabeledNodeIndices: unlabeledNodeIndices)
        emStepCallback(self)
      }

      // TODO: Add support for marginal likelihood fine-tuning.
      // How do we handle labeled data in that case?
    }
  }

  public func labelProbabilities(for nodeIndices: [Int]) -> Tensor<Float> {
    var batches = [Tensor<Float>]()
    for batch in Dataset(elements: Tensor<Int32>(nodeIndices.map(Int32.init))).batched(batchSize) {
      batches.append(expectedLabels.gathering(atIndices: batch))
//      batches.append(predictor.labelProbabilities(batch))
    }
    if batches.count == 1 { return batches[0] }
    return Tensor<Float>(concatenating: batches, alongAxis: 0)
  }

  private mutating func initialize(using data: Data) {
    eStepAccumulator = Tensor<Float>(zeros: [predictor.nodeCount, predictor.classCount])
    var labeledNodes = Set<Int>()
    var nodesToLabel = Set<Int>()

    // Start with the labeled nodes.
    for node in data.trainNodes {
      let label = data.nodeLabels[node]!
      eStepAccumulator = _Raw.tensorScatterAdd(
        eStepAccumulator,
        indices: Tensor<Int32>([Int32(node)]).expandingShape(at: -1),
        updates: Tensor<Float>(
          oneHotAtIndices: Tensor<Int32>(Int32(label)),
          depth: predictor.classCount).expandingShape(at: 0))
      labeledNodes.update(with: node)
      nodesToLabel.remove(node)
      data.nodeNeighbors[node].forEach {
        if !labeledNodes.contains($0) {
          nodesToLabel.update(with: $0)
        }
      }
    }

    // Proceed with label propagation for the unlabeled nodes.
    while !nodesToLabel.isEmpty {
      for node in nodesToLabel {
        let labeledNeighbors = data.nodeNeighbors[node].filter(labeledNodes.contains)
        var probabilities = Tensor<Float>(zeros: [data.classCount])
        for neighbor in labeledNeighbors {
          probabilities += eStepAccumulator[neighbor]
        }
        probabilities /= probabilities.sum()
        eStepAccumulator = _Raw.tensorScatterAdd(
          eStepAccumulator,
          indices: Tensor<Int32>([Int32(node)]).expandingShape(at: -1),
          updates: probabilities.expandingShape(at: 0))
      }
      for node in nodesToLabel {
        labeledNodes.update(with: node)
        nodesToLabel.remove(node)
        data.nodeNeighbors[node].forEach {
          if !labeledNodes.contains($0) {
            nodesToLabel.update(with: $0)
          }
        }
      }
    }

    for node in (0..<data.nodeCount).filter({ !labeledNodes.contains($0) }) {
      eStepAccumulator = _Raw.tensorScatterAdd(
        eStepAccumulator,
        indices: Tensor<Int32>([Int32(node)]).expandingShape(at: -1),
        updates: Tensor<Float>(repeating: 1 / Float(data.classCount), shape: [1, data.classCount]))
    }

    expectedLabels = eStepAccumulator
  }

  private mutating func performEStep(
    labeledData: Dataset<LabeledData>,
    unlabeledData: Dataset<UnlabeledData>,
    unlabeledNodeIndices: Dataset<Tensor<Int32>>
  ) {
    eStepAccumulator = Tensor<Float>(zeros: [predictor.nodeCount, predictor.classCount])

    // Set the labeled node labels to their true labels.
    for batch in labeledData.batched(batchSize) {
      eStepAccumulator = _Raw.tensorScatterAdd(
        eStepAccumulator,
        indices: batch.nodeIndices.expandingShape(at: -1),
        updates: Tensor<Float>(
          oneHotAtIndices: batch.nodeLabels,
          depth: predictor.classCount,
          onValue: 0,
          offValue: -999999))
    }

    // Compute expectations for the labels of the unlabeled nodes.
    for batch in unlabeledData.batched(batchSize) {
      let qualities = predictor.qualities(batch.nodeIndices, batch.neighborIndices)
      let qLog = logSoftmax(qualities, alongAxis: 2)                                                // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
      let neighborValues = expectedLabels.gathering(atIndices: batch.neighborIndices)               // [BatchSize, MaxNeighborCount, ClassCount]
      let yHat = useSoftPredictions ?
        neighborValues :
        Tensor<Float>(
          oneHotAtIndices: neighborValues.argmax(squeezingAxis: -1),
          depth: predictor.classCount)                                                              // [BatchSize, MaxNeighborCount, ClassCount]
      let maskOffset = (1 - batch.neighborMask.expandingShape(at: -1)) * -999999
      var qLogYHat = (qLog * yHat.expandingShape(at: 2)).sum(squeezingAxes: -1)                     // [BatchSize, MaxNeighborCount, ClassCount]
      qLogYHat = (qLogYHat + maskOffset).logSumExp(squeezingAxes: 1)                                // [BatchSize, ClassCount]
      eStepAccumulator = _Raw.tensorScatterAdd(
        eStepAccumulator,
        indices: batch.nodeIndices.expandingShape(at: -1),
        updates: qLogYHat)
    }
    for batch in unlabeledNodeIndices.batched(batchSize) {
      eStepAccumulator = _Raw.tensorScatterAdd(
        eStepAccumulator,
        indices: batch.expandingShape(at: -1),
        updates: predictor.labelProbabilities(batch))
    }
    expectedLabels = exp(logSoftmax(eStepAccumulator, alongAxis: -1))
  }

  // private mutating func performMStep(data: Dataset<LabeledData>, emStep: Int) {
  //   if !useWarmStarting { predictor.reset() }
  //   var accumulatedLoss = Float(0.0)
  //   var accumulatedSteps = 0
  //   var dataIterator = data.repeated()
  //     .shuffled(sampleCount: 10000, randomSeed: randomSeed &+ Int64(emStep))
  //     .batched(batchSize)
  //     .prefetched(count: 10)
  //     .makeIterator()
  //   for mStep in 0..<mStepCount {
  //     let batch = dataIterator.next()!
  //     let neighborIndices = Tensor<Int32>(
  //       repeating: 0,
  //       shape: [Int(batch.nodeIndices.shape[0]), predictor.maxBatchNeighborCount])
  //     let (loss, gradient) = predictor.valueWithGradient { predictor -> Tensor<Float> in
  //       let predictions = predictor(batch.nodeIndices, neighborIndices)
  //       let crossEntropy = softmaxCrossEntropy(
  //         logits: predictions.labelProbabilities,
  //         labels: batch.nodeLabels)
  //       let zero = predictions.qualities.sum() * 0.0
  //       return crossEntropy + zero
  //       // softmaxCrossEntropy(
  //       //   logits: predictor.labelProbabilities(batch.nodeIndices),
  //       //   labels: batch.nodeLabels)
  //     }
  //     optimizer.update(&predictor, along: gradient)
  //     accumulatedLoss += loss.scalarized()
  //     accumulatedSteps += 1
  //     if verbose {
  //       if let logSteps = mStepLogCount, mStep % logSteps == 0 || mStep == mStepCount - 1 {
  //         let nll = accumulatedLoss / Float(accumulatedSteps)
  //         let message = "Supervised M-Step \(String(format: "%5d", mStep)) | " +
  //           "Loss: \(String(format: "%.8f", nll))"
  //         logger.info("\(message)")
  //         accumulatedLoss = 0.0
  //         accumulatedSteps = 0
  //       }
  //     }
  //   }
  // }

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
        let predictions = predictor(batch.nodeIndices, batch.neighborIndices)
        let hLog = predictions.labelProbabilities
        let qLog = logSoftmax(predictions.qualities, alongAxis: 2)                                  // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
        let neighborValues = expectedLabels.gathering(atIndices: batch.neighborIndices)             // [BatchSize, MaxNeighborCount, ClassCount]
        let yHat = useSoftPredictions ?
          neighborValues :
          Tensor<Float>(
            oneHotAtIndices: neighborValues.argmax(squeezingAxis: -1),
            depth: classCount)                                                                      // [BatchSize, MaxNeighborCount, ClassCount]
        let maskOffset = (1 - batch.neighborMask.expandingShape(at: -1)) * -999999
        var qLogYHat = (qLog * yHat.expandingShape(at: 2)).sum(squeezingAxes: -1)                     // [BatchSize, MaxNeighborCount, ClassCount]
        qLogYHat = (qLogYHat + maskOffset).logSumExp(squeezingAxes: 1)                                // [BatchSize, ClassCount]
        let yExpected = expectedLabels.gathering(atIndices: batch.nodeIndices)
        return entropyWeight * (exp(hLog) * hLog).sum() -
          (yExpected * hLog).sum() -
          (yExpected * qLogYHat).sum()
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
