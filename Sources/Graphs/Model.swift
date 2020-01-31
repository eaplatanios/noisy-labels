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

public struct Model<Predictor: GraphPredictor, Optimizer: TensorFlow.Optimizer>
where Optimizer.Model == Predictor {
  public let entropyWeight: Float
  public let qualitiesRegularizationWeight: Float

  public let optimizerFn: () -> Optimizer
  public let randomSeed: Int64
  public let batchSize: Int
  public let useWarmStarting: Bool
  public let useThresholdedExpectations: Bool
  public let labelSmoothing: Float
  public let mStepCount: Int
  public let emStepCount: Int
  public let marginalStepCount: Int?
  public let evaluationStepCount: Int?
  public let mStepLogCount: Int?
  public let mConvergenceEvaluationCount: Int?
  public let emStepCallback: (Model) -> Bool
  public let verbose: Bool

  public var resultAccumulator: Accumulator

  public var predictor: Predictor
  public var optimizer: Optimizer
  public var eStepAccumulator: Tensor<Float> // [NodeCount, ClassCount]
  public var expectedLabels: Tensor<Float>   // [NodeCount, ClassCount]

  public var bestPredictor: Predictor
  public var bestResult: Result?

  public init(
    predictor: Predictor,
    optimizerFn: @escaping () -> Optimizer,
    entropyWeight: Float,
    qualitiesRegularizationWeight: Float,
    randomSeed: Int64,
    batchSize: Int = 128,
    useWarmStarting: Bool = false,
    useThresholdedExpectations: Bool = false,
    labelSmoothing: Float = 0.0,
    resultAccumulator: Accumulator = ExactAccumulator(),
    mStepCount: Int = 1000,
    emStepCount: Int = 100,
    marginalStepCount: Int? = nil,
    evaluationStepCount: Int? = 1,
    mStepLogCount: Int? = 100,
    mConvergenceEvaluationCount: Int? = 10,
    emStepCallback: @escaping (Model) -> Bool = { _ in false },
    verbose: Bool = false
  ) {
    self.resultAccumulator = resultAccumulator
    self.predictor = predictor
    self.optimizer = optimizerFn()
    self.optimizerFn = optimizerFn
    self.entropyWeight = entropyWeight
    self.qualitiesRegularizationWeight = qualitiesRegularizationWeight
    self.randomSeed = randomSeed
    self.batchSize = batchSize
    self.useWarmStarting = useWarmStarting
    self.useThresholdedExpectations = useThresholdedExpectations
    self.labelSmoothing = labelSmoothing
    self.mStepCount = mStepCount
    self.emStepCount = emStepCount
    self.marginalStepCount = marginalStepCount
    self.evaluationStepCount = evaluationStepCount
    self.mStepLogCount = mStepLogCount
    self.mConvergenceEvaluationCount = mConvergenceEvaluationCount
    self.emStepCallback = emStepCallback
    self.verbose = verbose
    self.eStepAccumulator = Tensor<Float>(zeros: [predictor.nodeCount, predictor.classCount])
    self.expectedLabels = Tensor<Float>(zeros: [predictor.nodeCount, predictor.classCount])
    self.bestPredictor = predictor
    self.bestResult = nil
  }

  public mutating func train(using graph: Graph) {
    let labeledData = Dataset(elements: graph.labeledData)
    let unlabeledData = Dataset(elements: graph.unlabeledData)
    let allUnlabeledData = Dataset(elements: graph.allUnlabeledData)
    let unlabeledNodeIndices = Dataset(elements: graph.unlabeledNodeIndices)

    if verbose { logger.info("Initialization") }
    // initialize(using: graph)
    performMStep(data: labeledData, emStep: 0)
    performEStep(
      labeledData: labeledData,
      unlabeledData: nil,
      unlabeledNodeIndices: unlabeledNodeIndices)
    emStepCallback(self)

    for emStep in 0..<emStepCount {
      // M-Step
      if verbose { logger.info("Iteration \(emStep) - Running M-Step") }
      performMStep(
        data: allUnlabeledData,
        emStep: emStep,
        eStepLabeledData: labeledData,
        eStepUnlabeledData: unlabeledData,
        eStepUnlabeledNodeIndices: unlabeledNodeIndices)

      // E-Step
      if verbose { logger.info("Iteration \(emStep) - Running E-Step") }
      performEStep(
        labeledData: labeledData,
        unlabeledData: unlabeledData,
        unlabeledNodeIndices: unlabeledNodeIndices)
      
      if emStepCallback(self) { break }
    }

    performMarginalStep(
      labeledData: labeledData,
      unlabeledData: unlabeledData,
      eStepLabeledData: labeledData,
      eStepUnlabeledData: unlabeledData,
      eStepUnlabeledNodeIndices: unlabeledNodeIndices)
    performEStep(
      labeledData: labeledData,
      unlabeledData: unlabeledData,
      unlabeledNodeIndices: unlabeledNodeIndices)
    emStepCallback(self)
  }

  public func labelProbabilities(
    for nodeIndices: [Int32],
    usePrior: Bool = false
  ) -> Tensor<Float> {
    var batches = [Tensor<Float>]()
    for batch in Dataset(elements: Tensor<Int32>(nodeIndices)).batched(batchSize) {
      batches.append(usePrior ? 
        predictor.labelProbabilities(batch.scalars) :
        expectedLabels.gathering(atIndices: batch))
    }
    if batches.count == 1 { return batches[0] }
    return Tensor<Float>(concatenating: batches, alongAxis: 0)
  }

  private mutating func initialize(using graph: Graph) {
    eStepAccumulator = Tensor<Float>(zeros: [predictor.nodeCount, predictor.classCount])
    var labeledNodes = Set<Int32>()
    var nodesToLabel = Set<Int32>()

    // Start with the labeled nodes.
    for node in graph.trainNodes {
      let label = graph.labels[node]!
      eStepAccumulator = _Raw.tensorScatterAdd(
        eStepAccumulator,
        indices: Tensor<Int32>([node]).expandingShape(at: -1),
        updates: Tensor<Float>(
          oneHotAtIndices: Tensor<Int32>(Int32(label)),
          depth: predictor.classCount).expandingShape(at: 0))
      labeledNodes.update(with: node)
      nodesToLabel.remove(node)
      graph.neighbors[Int(node)].forEach {
        if !labeledNodes.contains($0) {
          nodesToLabel.update(with: $0)
        }
      }
    }

    // Proceed with label propagation for the unlabeled nodes.
    while !nodesToLabel.isEmpty {
      for node in nodesToLabel {
        let labeledNeighbors = graph.neighbors[Int(node)].filter(labeledNodes.contains)
        var probabilities = Tensor<Float>(zeros: [graph.classCount])
        for neighbor in labeledNeighbors {
          probabilities += eStepAccumulator[Int(neighbor)]
        }
        probabilities /= probabilities.sum()
        eStepAccumulator = _Raw.tensorScatterAdd(
          eStepAccumulator,
          indices: Tensor<Int32>([node]).expandingShape(at: -1),
          updates: probabilities.expandingShape(at: 0))
      }
      for node in nodesToLabel {
        labeledNodes.update(with: node)
        nodesToLabel.remove(node)
        graph.neighbors[Int(node)].forEach {
          if !labeledNodes.contains($0) {
            nodesToLabel.update(with: $0)
          }
        }
      }
    }

    for node in (0..<Int32(graph.nodeCount)).filter({ !labeledNodes.contains($0) }) {
      eStepAccumulator = _Raw.tensorScatterAdd(
        eStepAccumulator,
        indices: Tensor<Int32>([node]).expandingShape(at: -1),
        updates: Tensor<Float>(
          repeating: 1 / Float(graph.classCount),
          shape: [1, graph.classCount]))
    }

    expectedLabels = eStepAccumulator
    expectedLabels = useThresholdedExpectations ?
      Tensor<Float>(expectedLabels .== expectedLabels.max(alongAxes: -1)) :
      expectedLabels
  }

  private mutating func performEStep(
    labeledData: Dataset<LabeledData>,
    unlabeledData: Dataset<Tensor<Int32>>?,
    unlabeledNodeIndices: Dataset<Tensor<Int32>>
  ) {
    eStepAccumulator = Tensor<Float>(zeros: [predictor.nodeCount, predictor.classCount])

    // Set the labeled node labels to their true labels.
    var labeledAccumulator = eStepAccumulator
    for batch in labeledData.batched(batchSize) {
      labeledAccumulator = _Raw.tensorScatterAdd(
        labeledAccumulator,
        indices: batch.nodeIndices.expandingShape(at: -1),
        updates: Tensor<Float>(
          oneHotAtIndices: batch.nodeLabels,
          depth: predictor.classCount,
          onValue: 0,
          offValue: -100000000))
    }

    // Compute expectations for the labels of the unlabeled nodes.
    var unlabeledAccumulator = eStepAccumulator
    if let unlabeledData = unlabeledData {
      for batch in unlabeledData.batched(batchSize) {
        let predictions = predictor(batch.scalars)
        let qLog = predictions.qualities                                                            // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
        let qLogMask = predictions.qualitiesMask                                                    // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
        let neighborValues = expectedLabels.gathering(atIndices: predictions.neighborIndices)       // [BatchSize, MaxNeighborCount, ClassCount]
        let yHat = neighborValues.expandingShape(at: 2)                                             // [BatchSize, MaxNeighborCount, 1, ClassCount]
        let mask = qLogMask.expandingShape(at: -1)                                                  // [BatchSize, MaxNeighborCount, 1]
        let qLogYHat = ((qLog * yHat).sum(squeezingAxes: -1) * mask).sum(squeezingAxes: 1)          // [BatchSize, ClassCount]
        unlabeledAccumulator = _Raw.tensorScatterAdd(
          unlabeledAccumulator,
          indices: batch.expandingShape(at: -1),
          updates: qLogYHat)
      }
    }
    for batch in unlabeledNodeIndices.batched(batchSize) {
      unlabeledAccumulator = _Raw.tensorScatterAdd(
        unlabeledAccumulator,
        indices: batch.expandingShape(at: -1),
        updates: predictor.labelProbabilities(batch.scalars))
    }
    eStepAccumulator = labeledAccumulator + unlabeledAccumulator
    labeledAccumulator = exp(logSoftmax(labeledAccumulator, alongAxis: -1))
    unlabeledAccumulator = exp(logSoftmax(unlabeledAccumulator, alongAxis: -1))
    unlabeledAccumulator = (1 - labelSmoothing) * unlabeledAccumulator +
      labelSmoothing / Float(unlabeledAccumulator.shape[unlabeledAccumulator.rank - 1])
    expectedLabels = labeledAccumulator + unlabeledAccumulator
    expectedLabels = useThresholdedExpectations ?
      Tensor<Float>(expectedLabels .== expectedLabels.max(alongAxes: -1)) :
      expectedLabels
  }

  private mutating func performMStep(data: Dataset<LabeledData>, emStep: Int) {
    bestResult = nil
    resultAccumulator.reset()
    if !useWarmStarting {
      predictor.reset()
      optimizer = optimizerFn()
    }
    var accumulatedLoss = Float(0.0)
    var accumulatedSteps = 0
    var dataIterator = data.repeated()
      .shuffled(sampleCount: 10000, randomSeed: randomSeed &+ Int64(emStep))
      .batched(batchSize)
      .prefetched(count: 10)
      .makeIterator()
    for mStep in 0..<mStepCount {
      let batch = dataIterator.next()!
      let labels = (1 - labelSmoothing) * Tensor<Float>(
        oneHotAtIndices: batch.nodeLabels,
        depth: predictor.classCount
      ) + labelSmoothing / Float(predictor.classCount)
      withLearningPhase(.training) {
        let (loss, gradient) = valueWithGradient(at: predictor) { predictor -> Tensor<Float> in
          let predictions = predictor(batch.nodeIndices.scalars)
          let crossEntropy = softmaxCrossEntropy(
            logits: predictions.labelProbabilities,
            probabilities: labels)
          let loss = crossEntropy + predictions.qualities.sum() * 0.0
          return loss / Float(predictions.labelProbabilities.shape[0])
        }
        optimizer.update(&predictor, along: gradient)
        accumulatedLoss += loss.scalarized()
        accumulatedSteps += 1
        if verbose {
          if let logSteps = mStepLogCount, mStep % logSteps == 0 || mStep == mStepCount - 1 {
            let nll = accumulatedLoss / Float(accumulatedSteps)
            let message = "Supervised M-Step \(String(format: "%5d", mStep)) | " +
              "Loss: \(String(format: "%.8f", nll))"
            logger.info("\(message)")
            accumulatedLoss = 0.0
            accumulatedSteps = 0
          }
        }
      }
      if let c = evaluationStepCount, mStep % c == 0 {
        let result = resultAccumulator.update(
          with: evaluate(model: self, using: predictor.graph, usePrior: true))
        if let bestResult = self.bestResult {
          if result.validationAccuracy > bestResult.validationAccuracy {
            self.bestPredictor = predictor
            self.bestResult = result
          }
        } else {
          self.bestPredictor = predictor
          self.bestResult = result
        }
      }
    }
    if evaluationStepCount != nil {
      predictor = bestPredictor
    }
  }

  private mutating func performMStep(
    data: Dataset<Tensor<Int32>>,
    emStep: Int,
    eStepLabeledData: Dataset<LabeledData>,
    eStepUnlabeledData: Dataset<Tensor<Int32>>?,
    eStepUnlabeledNodeIndices: Dataset<Tensor<Int32>>
  ) {
    var convergenceStepCount = 0
    bestResult = nil
    resultAccumulator.reset()
    if !useWarmStarting {
      predictor.reset()
      optimizer = optimizerFn()
    }
    var accumulatedNLL = Float(0.0)
    var accumulatedSteps = 0
    var dataIterator = data.repeated()
      .shuffled(sampleCount: 10000, randomSeed: randomSeed &+ Int64(emStep))
      .batched(batchSize)
      .prefetched(count: 10)
      .makeIterator()
    for mStep in 0..<mStepCount {
      let batch = dataIterator.next()!
      withLearningPhase(.training) {
        let (negativeLogLikelihood, gradient) = valueWithGradient(at: predictor) {
          [expectedLabels, entropyWeight, qualitiesRegularizationWeight]
          predictor -> Tensor<Float> in
          let predictions = predictor(batch.scalars)
          let hLog = predictions.labelProbabilities
          let qLog = predictions.qualities                                                          // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
          let qLogMask = predictions.qualitiesMask                                                  // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
          let neighborValues = expectedLabels.gathering(atIndices: predictions.neighborIndices)     // [BatchSize, MaxNeighborCount, ClassCount]
          let yHat = neighborValues.expandingShape(at: 2)                                           // [BatchSize, MaxNeighborCount, 1, ClassCount]
          let mask = qLogMask.expandingShape(at: -1)                                                // [BatchSize, MaxNeighborCount, 1]
          let qLogYHat = ((qLog * yHat).sum(squeezingAxes: -1) * mask).sum(squeezingAxes: 1)        // [BatchSize, ClassCount]
          let yExpected = expectedLabels.gathering(atIndices: batch)
          let maskedQEntropy = exp(qLog) * qLog * mask.expandingShape(at: -1)
          let qEntropy = qualitiesRegularizationWeight * maskedQEntropy.sum()
          let hEntropy = entropyWeight * (exp(hLog) * hLog).sum()
          let loss = hEntropy + qEntropy - (yExpected * hLog).sum() - (yExpected * qLogYHat).sum()
          return loss / Float(predictions.labelProbabilities.shape[0])
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
      if let c = evaluationStepCount, mStep % c == 0 {
        // Run the E step.
        var modelCopy = self
        modelCopy.performEStep(
          labeledData: eStepLabeledData,
          unlabeledData: eStepUnlabeledData,
          unlabeledNodeIndices: eStepUnlabeledNodeIndices)

        let evaluationResult = evaluate(model: modelCopy, using: predictor.graph, usePrior: false)
        let result = resultAccumulator.update(with: evaluationResult)
        if let bestResult = self.bestResult {
          if result.validationAccuracy > bestResult.validationAccuracy {
            logger.info("Best predictor accuracy: \(evaluationResult.validationAccuracy) | \(result.validationAccuracy)")
            self.bestPredictor = predictor
            self.bestResult = result
            convergenceStepCount = 0
          } else {
            convergenceStepCount += 1
            if let c = self.mConvergenceEvaluationCount, convergenceStepCount > c {
              predictor = bestPredictor
              return
            }
          }
        } else {
          self.bestPredictor = predictor
          self.bestResult = result
        }
      }
    }
    if evaluationStepCount != nil {
      predictor = bestPredictor
    }
  }

  private mutating func performMarginalStep(
    labeledData: Dataset<LabeledData>,
    unlabeledData: Dataset<Tensor<Int32>>,
    eStepLabeledData: Dataset<LabeledData>,
    eStepUnlabeledData: Dataset<Tensor<Int32>>?,
    eStepUnlabeledNodeIndices: Dataset<Tensor<Int32>>
  ) {
    guard let marginalStepCount = self.marginalStepCount else { return }
    var convergenceStepCount = 0
    bestResult = nil
    resultAccumulator.reset()
    var accumulatedNLL = Float(0.0)
    var accumulatedSteps = 0
    var labeledDataIterator = labeledData.repeated()
      .shuffled(sampleCount: 10000, randomSeed: randomSeed)
      .batched(batchSize)
      .prefetched(count: 10)
      .makeIterator()
    var unlabeledDataIterator = unlabeledData.repeated()
      .shuffled(sampleCount: 10000, randomSeed: randomSeed)
      .batched(batchSize)
      .prefetched(count: 10)
      .makeIterator()
    for mStep in 0..<marginalStepCount {
      let unlabeledBatch = unlabeledDataIterator.next()!
      withLearningPhase(.training) {
        let (negativeLogLikelihood, gradient) = valueWithGradient(at: predictor) {
          [expectedLabels, entropyWeight, qualitiesRegularizationWeight]
          predictor -> Tensor<Float> in
          let predictions = predictor(unlabeledBatch.scalars)
          let hLog = predictions.labelProbabilities
          let qLog = predictions.qualities                                                          // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
          let qLogMask = predictions.qualitiesMask                                                  // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
          let neighborValues = expectedLabels.gathering(atIndices: predictions.neighborIndices)     // [BatchSize, MaxNeighborCount, ClassCount]
          let yHat = neighborValues.expandingShape(at: 2)                                           // [BatchSize, MaxNeighborCount, 1, ClassCount]
          let mask = qLogMask.expandingShape(at: -1)                                                // [BatchSize, MaxNeighborCount, 1]
          let qLogYHat = ((qLog * yHat).sum(squeezingAxes: -1) * mask).sum(squeezingAxes: 1)        // [BatchSize, ClassCount]
          let logLikelihood = (qLogYHat + hLog).logSumExp(squeezingAxes: -1).sum()
          let maskedQEntropy = exp(qLog) * qLog * mask.expandingShape(at: -1)
          let qEntropy = qualitiesRegularizationWeight * maskedQEntropy.sum()
          let hEntropy = entropyWeight * (exp(hLog) * hLog).sum()
          let loss = hEntropy + qEntropy - logLikelihood
          return loss / Float(predictions.labelProbabilities.shape[0])
        }
        optimizer.update(&predictor, along: gradient)
        accumulatedNLL += negativeLogLikelihood.scalarized()
        accumulatedSteps += 1
        if verbose {
          if let logSteps = mStepLogCount, mStep % logSteps == 0 || mStep == mStepCount - 1 {
            let nll = accumulatedNLL / Float(accumulatedSteps)
            let message = "Marginal Step \(String(format: "%5d", mStep)) | " +
              "Negative Log-Likelihood: \(String(format: "%.8f", nll))"
            logger.info("\(message)")
            accumulatedNLL = 0.0
            accumulatedSteps = 0
          }
        }
      }
      let labeledBatch = labeledDataIterator.next()!
      withLearningPhase(.training) {
        let (negativeLogLikelihood, gradient) = valueWithGradient(at: predictor) {
          [expectedLabels, entropyWeight, qualitiesRegularizationWeight]
          predictor -> Tensor<Float> in
          let predictions = predictor(labeledBatch.nodeIndices.scalars)
          let hLog = predictions.labelProbabilities
          let qLog = predictions.qualities                                                          // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
          let qLogMask = predictions.qualitiesMask                                                  // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
          let neighborValues = expectedLabels.gathering(atIndices: predictions.neighborIndices)     // [BatchSize, MaxNeighborCount, ClassCount]
          let yHat = neighborValues.expandingShape(at: 2)                                           // [BatchSize, MaxNeighborCount, 1, ClassCount]
          let mask = qLogMask.expandingShape(at: -1)                                                // [BatchSize, MaxNeighborCount, 1]
          let qLogYHat = ((qLog * yHat).sum(squeezingAxes: -1) * mask).sum(squeezingAxes: 1)        // [BatchSize, ClassCount]
          let yExpected = expectedLabels.gathering(atIndices: labeledBatch.nodeIndices)
          let logLikelihood = (yExpected * (qLogYHat + hLog)).sum()
          let maskedQEntropy = exp(qLog) * qLog * mask.expandingShape(at: -1)
          let qEntropy = qualitiesRegularizationWeight * maskedQEntropy.sum()
          let hEntropy = entropyWeight * (exp(hLog) * hLog).sum()
          let loss = hEntropy + qEntropy - logLikelihood
          return loss / Float(predictions.labelProbabilities.shape[0])
        }
        optimizer.update(&predictor, along: gradient)
        accumulatedNLL += negativeLogLikelihood.scalarized()
        accumulatedSteps += 1
      }

      // Run the E step.
      performEStep(
        labeledData: eStepLabeledData,
        unlabeledData: eStepUnlabeledData,
        unlabeledNodeIndices: eStepUnlabeledNodeIndices)

      if let c = evaluationStepCount, mStep % c == 0 {
        // Perform evaluation.
        let result = resultAccumulator.update(
          with: evaluate(model: self, using: predictor.graph, usePrior: false))
        if let bestResult = self.bestResult {
          if result.validationAccuracy > bestResult.validationAccuracy {
            logger.info("Best predictor accuracy: \(result.validationAccuracy)")
            self.bestPredictor = predictor
            self.bestResult = result
            convergenceStepCount = 0
          } else {
            convergenceStepCount += 1
            if let c = self.mConvergenceEvaluationCount, convergenceStepCount > c {
              predictor = bestPredictor
              return
            }
          }
        } else {
          self.bestPredictor = predictor
          self.bestResult = result
        }
      }
    }
    if evaluationStepCount != nil {
      predictor = bestPredictor
    }
  }
}

public protocol Accumulator {
  mutating func reset()
  mutating func update(with result: Result) -> Result
}

public struct ExactAccumulator: Accumulator {
  public init() {}
  public mutating func reset() {}
  public mutating func update(with result: Result) -> Result { result }
}

public struct MovingAverageAccumulator: Accumulator {
  public let weight: Float
  private var accumulated: Result? = nil

  public init(weight: Float) {
    self.weight = weight
  }

  public mutating func reset() { accumulated = nil }

  public mutating func update(with result: Result) -> Result {
    if let a = accumulated {
      accumulated = a.scaled(by: 1 - weight).adding(result.scaled(by: weight))
    } else {
      accumulated = result
    }
    return accumulated!
  }
}
