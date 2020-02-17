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
import Progress
import TensorFlow

public enum ModelInitializationMethod {
  case groundTruth
  case random
  case labelPropagation

  public func initialExpectedLabels(forGraph graph: Graph) -> Tensor<Float> {
    switch self {
    case .groundTruth:
      return Tensor<Float>(
        oneHotAtIndices: Tensor<Int32>(
          graph.labels.sorted { $0.key < $1.key }.map { Int32($0.value) }),
        depth: graph.classCount)
    case .random:
      var sample = (0..<graph.nodeCount).map { _ in Int32.random(in: 0..<Int32(graph.classCount)) }
      for node in graph.trainNodes { sample[Int(node)] = Int32(graph.labels[node]!) }
      return Tensor<Float>(oneHotAtIndices: Tensor<Int32>(sample), depth: graph.classCount)
    case .labelPropagation:
      var labelScores = Tensor<Float>(zeros: [graph.nodeCount, graph.classCount])
      var labeledNodes = Set<Int32>()
      var nodesToLabel = Set<Int32>()

      // Start with the labeled nodes.
      for node in graph.trainNodes {
        let label = graph.labels[node]!
        labelScores = _Raw.tensorScatterAdd(
          labelScores,
          indices: Tensor<Int32>([node]).expandingShape(at: -1),
          updates: Tensor<Float>(
            oneHotAtIndices: Tensor<Int32>(Int32(label)),
            depth: graph.classCount).expandingShape(at: 0))
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
            probabilities += labelScores[Int(neighbor)]
          }
          probabilities /= probabilities.sum()
          labelScores = _Raw.tensorScatterAdd(
            labelScores,
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
        labelScores = _Raw.tensorScatterAdd(
          labelScores,
          indices: Tensor<Int32>([node]).expandingShape(at: -1),
          updates: Tensor<Float>(
            repeating: 1 / Float(graph.classCount),
            shape: [1, graph.classCount]))
      }

      return labelScores
    }
  }
}

public struct Model<Predictor: GraphPredictor, Optimizer: TensorFlow.Optimizer>
where Optimizer.Model == Predictor {
  private let graph: Graph

  public let optimizerFn: () -> Optimizer
  public let randomSeed: Int64
  public let batchSize: Int
  public let useWarmStarting: Bool
  public let useIncrementalNeighborhoodExpansion: Bool
  public let initializationMethod: ModelInitializationMethod
  public let mStepCount: Int
  public let emStepCount: Int
  public let preTrainingStepCount: Int
  public let evaluationStepCount: Int?
  public let evaluationConvergenceStepCount: Int?
  public let emStepCallback: (Model) -> ()
  public let mStepLogCount: Int?
  public let verbose: Bool

  public var predictor: Predictor
  public var optimizer: Optimizer
  public var evaluationResultsAccumulator: Accumulator

  private var expectedLabels: Tensor<Float>
  private var bestPredictor: Predictor
  private var bestResult: Result?

  public init(
    graph: Graph,
    predictor: Predictor,
    optimizerFn: @escaping () -> Optimizer,
    randomSeed: Int64,
    batchSize: Int = 128,
    useWarmStarting: Bool = false,
    useIncrementalNeighborhoodExpansion: Bool = false,
    initializationMethod: ModelInitializationMethod = .labelPropagation,
    mStepCount: Int = 1000,
    emStepCount: Int = 1000,
    preTrainingStepCount: Int = 1000,
    evaluationStepCount: Int? = 1,
    evaluationConvergenceStepCount: Int? = 10,
    evaluationResultsAccumulator: Accumulator = ExactAccumulator(),
    emStepCallback: @escaping (Model) -> () = { _ in () },
    mStepLogCount: Int? = 100,
    verbose: Bool = false
  ) {
    self.graph = graph
    self.optimizerFn = optimizerFn
    self.randomSeed = randomSeed
    self.batchSize = batchSize
    self.useWarmStarting = useWarmStarting
    self.useIncrementalNeighborhoodExpansion = useIncrementalNeighborhoodExpansion
    self.initializationMethod = initializationMethod
    self.mStepCount = mStepCount
    self.emStepCount = emStepCount
    self.preTrainingStepCount = preTrainingStepCount
    self.evaluationStepCount = evaluationStepCount
    self.evaluationConvergenceStepCount = evaluationConvergenceStepCount
    self.emStepCallback = emStepCallback
    self.mStepLogCount = mStepLogCount
    self.verbose = verbose
    self.predictor = predictor
    self.optimizer = optimizerFn()
    self.evaluationResultsAccumulator = evaluationResultsAccumulator
    self.expectedLabels = initializationMethod.initialExpectedLabels(forGraph: graph)
    self.bestPredictor = predictor
    self.bestResult = nil
  }

  public func labelLogits(forNodes nodes: [Int32], usePrior: Bool = false) -> Tensor<Float> {
    let nodes = Tensor<Int32>(nodes)
    if usePrior { return predictor.labelLogits(forNodes: nodes, using: graph) }
//    let predictions = predictor.predictions(forNodes: Tensor<Int32>(nodes), using: graph)
//    let neighborY = predictor.labelLogits(
//      forNodes: predictions.neighborIndices.flattened(),
//      using: graph
//    ).reshaped(to: [predictions.neighborIndices.shape[0], -1, graph.classCount])
//    let h = predictions.labelLogits                                                                 // [BatchSize, ClassCount]
//    let g = predictions.qualityLogits                                                               // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
//    let gMask = predictions.neighborMask                                                            // [BatchSize, MaxNeighborCount]
//    let gY = (g * neighborY.expandingShape(at: 2)).sum(squeezingAxes: -1)                           // [BatchSize, MaxNeighborCount, ClassCount]
//    let gYMasked = (gY * gMask.expandingShape(at: -1)).sum(squeezingAxes: 1)                        // [BatchSize, ClassCount]
//    return h + gYMasked
    return expectedLabels.gathering(atIndices: Tensor<Int32>(nodes))
  }

  public mutating func train() {
    if preTrainingStepCount > 0 {
      if verbose { logger.info("Starting model pre-training.") }
      preTrainLabelsPredictor()
    }

    emStepCallback(self)
    if verbose { logger.info("Starting model training.") }

    var previousGraphExpansion = 0
    var subGraph = SubGraph(graph: graph, mapFromOriginalIndex: nil)
    for emStep in 0..<emStepCount {
      if useIncrementalNeighborhoodExpansion {
        // TODO: !!! Make this schedule configurable.
        let graphDepth = (emStep / 2) + 1
        if previousGraphExpansion < graphDepth {
          previousGraphExpansion = graphDepth
          subGraph = graph.subGraph(upToDepth: graphDepth)
          logger.info("Training on \(subGraph.nodeCount) / \(graph.nodeCount) nodes.")
        }
      }

      if verbose { logger.info("Iteration \(emStep) - Running M-Step") }
      performMStep(using: subGraph, randomSeed: randomSeed &+ Int64(emStep))

      if verbose { logger.info("Iteration \(emStep) - Running E-Step") }
      performEStep(using: subGraph, randomSeed: randomSeed &+ Int64(emStep))

      emStepCallback(self)
    }
  }

  private mutating func preTrainLabelsPredictor() {
    let graph = self.graph
    var accumulatedLoss = Float(0.0)
    var accumulatedSteps = 0
    var dataIterator = Dataset(elements: graph.labeledData).repeated()
      .shuffled(sampleCount: 10000, randomSeed: randomSeed)
      .batched(batchSize)
      .prefetched(count: 10)
      .makeIterator()
    for step in 0..<preTrainingStepCount {
      let batch = dataIterator.next()!
      let labels = Tensor<Float>(oneHotAtIndices: batch.labels, depth: graph.classCount)
      withLearningPhase(.training) {
        let (loss, gradient) = valueWithGradient(at: predictor) { predictor -> Tensor<Float> in
          // TODO: We cannot use "predictor.labelLogitsHelper" directly due to a compiler AD bug.
          let predictions = predictor.predictionsHelper(forNodes: batch.nodes, using: graph)
          return softmaxCrossEntropy(
            logits: predictions.labelLogits,
            probabilities: labels) +
            predictions.qualityLogits.sum() * 0 +
            predictions.qualityLogitsTranspose.sum() * 0
        }
        optimizer.update(&predictor, along: gradient)
        accumulatedLoss += loss.scalarized()
        accumulatedSteps += 1
        if verbose {
          if let logStepCount = self.mStepLogCount, step % logStepCount == 0 ||
            step == preTrainingStepCount - 1 {
            let nll = accumulatedLoss / Float(accumulatedSteps)
            let message = "Supervised Step \(String(format: "%5d", step)) | " +
              "Loss: \(String(format: "%.8f", nll))"
            logger.info("\(message)")
            accumulatedLoss = 0.0
            accumulatedSteps = 0
          }
        }
      }

      // Keep track of the best predictor so far.
      if let evaluationStepCount = self.evaluationStepCount, step % evaluationStepCount == 0 {
        let result = evaluationResultsAccumulator.update(
          with: evaluate(model: self, using: graph, usePrior: true))
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

//  private mutating func performEStep(using subGraph: SubGraph, randomSeed: Int64) {
//    // Set the labeled node labels to their true labels.
//    for batch in Dataset(elements: subGraph.labeledData).batched(batchSize) {
//      expectedLabels = _Raw.tensorScatterUpdate(
//        expectedLabels,
//        indices: batch.nodes.expandingShape(at: -1),
//        updates: Tensor<Float>(
//          oneHotAtIndices: batch.labels,
//          depth: subGraph.classCount,
//          onValue: 0,
//          offValue: -100000000))
//    }
//
//    // Compute expectations for the labels of the unlabeled nodes.
//    if subGraph.unlabeledNodes.count > 0 {
////      let unlabeledData = Dataset(elements: subGraph.unlabeledNodesTensor)
////        .shuffled(sampleCount: 10000, randomSeed: randomSeed)
////        .batched(batchSize)
//      let unlabeledData = [subGraph.unlabeledNodesTensor]
//      for batch in unlabeledData {
//        let predictions = predictor.predictions(forNodes: batch, using: subGraph.graph)
//        let neighborY = exp(expectedLabels.gathering(atIndices: predictions.neighborIndices))       // [BatchSize, MaxNeighborCount, ClassCount]
//        let h = predictions.labelLogits                                                             // [BatchSize, ClassCount]
//        let g = predictions.qualityLogits                                                           // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
//        let gMask = predictions.neighborMask                                                        // [BatchSize, MaxNeighborCount]
//        let gY = (g * neighborY.expandingShape(at: 2)).sum(squeezingAxes: -1)                       // [BatchSize, MaxNeighborCount, ClassCount]
//        let gYMasked = (gY * gMask.expandingShape(at: -1)).sum(squeezingAxes: 1)                    // [BatchSize, ClassCount]
//        expectedLabels = _Raw.tensorScatterUpdate(
//          expectedLabels,
//          indices: batch.expandingShape(at: -1),
//          updates: logSoftmax(h + gYMasked, alongAxis: -1))
//      }
//    }
//  }

  private mutating func performEStep(using subGraph: SubGraph, randomSeed: Int64) {
    let predictions = InMemoryPredictions(
      fromPredictions: predictor.predictions(
        forNodes: subGraph.nodesTensor,
        using: subGraph.graph),
      using: subGraph.graph)

    // Flat array representation of a tensor with shape [NodeCount, ClassCount].
    var expectedLabels = self.expectedLabels.scalars

    // Set the labeled node labels to their true labels.
    for node in subGraph.trainNodes {
      let label = subGraph.labels[node]!
      for k in 0..<subGraph.classCount {
        expectedLabels[Int(node) * subGraph.classCount + k] = label == k ? 0 : -100000000
      }
    }

    // Compute expectations for the labels of the unlabeled nodes.
    for node in subGraph.unlabeledNodes {
      let nodeOffset = Int(node) * subGraph.classCount
      for k in 0..<subGraph.classCount {
        expectedLabels[nodeOffset + k] =
          predictions.labelLogits.labelLogit(node: Int(node), label: k)
      }
      let neighbors = subGraph.neighbors[Int(node)]
      let neighborCount = Float(neighbors.count)
      for (neighborIndex, neighbor) in neighbors.enumerated() {
        for k in 0..<subGraph.classCount {
          for l in 0..<subGraph.classCount {
            let neighborLabel = exp(expectedLabels[Int(neighbor) * subGraph.classCount + l])
            expectedLabels[nodeOffset + k] +=
              predictions.qualityLogits[Int(node)].qualityLogit(
                forNeighbor: neighborIndex,
                nodeLabel: k,
                neighborLabel: l) * Float(neighborLabel) / neighborCount
            expectedLabels[nodeOffset + k] +=
              predictions.qualityLogitsTranspose[Int(neighbor)].qualityLogit(
                forNeighbor: subGraph.neighbors[Int(neighbor)].firstIndex(of: node)!,
                nodeLabel: l,
                neighborLabel: k) * Float(neighborLabel) / neighborCount
          }
        }
      }

      // Normalize the node label distribution.
      var maxLogit = -Float.infinity
      for k in 0..<subGraph.classCount {
        let logit = expectedLabels[nodeOffset + k]
        if logit > maxLogit {
          maxLogit = logit
        }
      }
      var logitExpSum: Float = 0
      for k in 0..<subGraph.classCount {
        let logit = expectedLabels[nodeOffset + k]
        logitExpSum += exp(logit - maxLogit)
      }
      let logSumExp = log(logitExpSum) + maxLogit
      for k in 0..<subGraph.classCount {
        expectedLabels[nodeOffset + k] -= logSumExp
      }
    }

    self.expectedLabels = Tensor<Float>(
      shape: self.expectedLabels.shape,
      scalars: expectedLabels)
  }

  public mutating func performMStep(using subGraph: SubGraph, randomSeed: Int64) {
    let expectedLabels = self.expectedLabels
    var dataIterator = Dataset(elements: subGraph.nodesTensor).repeated()
      .shuffled(sampleCount: 10000, randomSeed: randomSeed)
      .batched(batchSize)
      .prefetched(count: 10)
      .makeIterator()
    var convergenceStepCount = 0
    bestResult = nil
    evaluationResultsAccumulator.reset()
    if !useWarmStarting {
      predictor.reset()
      optimizer = optimizerFn()
    }
    var accumulatedNLL = Float(0.0)
    var accumulatedSteps = 0
    for mStep in 0..<mStepCount {
      let batch = dataIterator.next()!
      withLearningPhase(.training) {
        let (negativeLogLikelihood, gradient) = valueWithGradient(at: predictor) {
          predictor -> Tensor<Float> in
          let predictions = predictor.predictionsHelper(forNodes: batch, using: subGraph.graph)
          let y = exp(expectedLabels.gathering(atIndices: batch))                                   // [BatchSize, ClassCount]
          let neighborY = exp(expectedLabels.gathering(atIndices: predictions.neighborIndices))     // [BatchSize, MaxNeighborCount, ClassCount]
          let h = predictions.labelLogits                                                           // [BatchSize, ClassCount]
          let g = predictions.qualityLogits                                                         // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
          let gT = predictions.qualityLogitsTranspose                                               // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
          let gMask = predictions.neighborMask                                                      // [BatchSize, MaxNeighborCount]
          let gY = (g * neighborY.expandingShape(at: -2)).sum(squeezingAxes: -1)                    // [BatchSize, MaxNeighborCount, ClassCount]
          let gTY = (gT * neighborY.expandingShape(at: -1)).sum(squeezingAxes: -2)                  // [BatchSize, MaxNeighborCount, ClassCount]
          let gYMasked = (gY * gMask.expandingShape(at: -1)).sum(squeezingAxes: 1)                  // [BatchSize, ClassCount]
          let gTYMasked = (gTY * gMask.expandingShape(at: -1)).sum(squeezingAxes: 1)                // [BatchSize, ClassCount]

//          return softmaxCrossEntropy(
//            logits: h + gYMasked,
//            probabilities: y,
//            reduction: { $0.mean() })

          let neighbors = predictions.neighborIndices
          let neighborLabelSamples = Tensor<Float>(
            oneHotAtIndices: Tensor<Int32>(
              randomCategorialLogits: expectedLabels.gathering(atIndices: neighbors.flattened()),
              sampleCount: 100),
            depth: subGraph.classCount
          ).reshaped(to: TensorShape(
            [neighbors.shape[0], neighbors.shape[1], 100, subGraph.classCount]))                    // [BatchSize, MaxNeighborCount, SampleCount, ClassCount]
          let gYSamples = (g.expandingShape(at: 2) * neighborLabelSamples.expandingShape(at: -2))
            .sum(squeezingAxes: -1)                                                                 // [BatchSize, MaxNeighborCount, SampleCount, ClassCount]
          let gTYSamples = (gT.expandingShape(at: 2) * neighborLabelSamples.expandingShape(at: -1))
            .sum(squeezingAxes: -2)                                                                 // [BatchSize, MaxNeighborCount, SampleCount, ClassCount]
          let gYSamplesMasked = (gYSamples * gMask.expandingShape(at: -1, -2)).sum(squeezingAxes: 1) // [BatchSize, SampleCount, ClassCount]
          let gTYSamplesMasked = (gTYSamples * gMask.expandingShape(at: -1, -2)).sum(squeezingAxes: 1) // [BatchSize, SampleCount, ClassCount]
          let hgYgTYSamplesMasked = h.expandingShape(at: 1) + gYSamplesMasked + gTYSamplesMasked
          let normalizingConstant = hgYgTYSamplesMasked.logSumExp(squeezingAxes: -1).mean(squeezingAxes: 1)

          return (-((h + gYMasked + gTYMasked) * y).sum(squeezingAxes: -1) + normalizingConstant).mean()
        }
        optimizer.update(&predictor, along: gradient)
        accumulatedNLL += negativeLogLikelihood.scalarized()
        accumulatedSteps += 1
        if verbose {
          if let logStepCount = self.mStepLogCount, mStep % logStepCount == 0 || mStep == mStepCount - 1 {
            let nll = accumulatedNLL / Float(accumulatedSteps)
            let message = "M-Step \(String(format: "%5d", mStep)) " +
              "Negative Log-Likelihood: \(String(format: "%.8f", nll))"
            logger.info("\(message)")
            accumulatedNLL = 0.0
            accumulatedSteps = 0
          }
        }
      }

      // Check for early stopping.
      if let evaluationStepCount = self.evaluationStepCount, mStep % evaluationStepCount == 0 {
        var modelCopy = self
        modelCopy.performEStep(
          using: SubGraph(graph: graph, mapFromOriginalIndex: nil),
          randomSeed: randomSeed)
        let evaluationResult = evaluate(model: modelCopy, using: graph, usePrior: false)
        let result = evaluationResultsAccumulator.update(with: evaluationResult)
        if let bestResult = self.bestResult {
          if result.validationAccuracy > bestResult.validationAccuracy {
            let message = "Best predictor accuracy: \(evaluationResult.validationAccuracy) | " +
              "\(result.validationAccuracy)"
            logger.info("\(message)")
            self.bestPredictor = predictor
            self.bestResult = result
            convergenceStepCount = 0
          } else {
            convergenceStepCount += 1
            if let evaluationConvergenceStepCount = self.evaluationConvergenceStepCount,
               convergenceStepCount > evaluationConvergenceStepCount {
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

//  internal func labelsGibbsMarginalMAP() -> [Int32] {
//    var modelCopy = self
//    modelCopy.performGibbsEStep(using: predictor.graph)
//    let probabilities = modelCopy.labelProbabilities(for: [Int32](0..<Int32(predictor.graph.nodeCount)))
//    return probabilities.argmax(squeezingAxis: -1).scalars
//
////    let nodes = [Int](0..<predictor.graph.nodeCount).map(Int32.init)
////    var h = [Tensor<Float>]()
////    var q = [Tensor<Float>]()
////    for batch in Dataset(elements: Tensor<Int32>(nodes)).batched(batchSize) {
////      let predictions = predictor.labelProbabilitiesAndQualities(batch.scalars)
////      let qualities = predictions.qualities.unstacked(alongAxis: 0)
////      let neighborCounts = predictions.qualitiesMask
////        .sum(squeezingAxes: -1)
////        .unstacked(alongAxis: 0)
////        .map { Int($0.scalarized()) }
////      h.append(predictions.labelProbabilities)
////      q.append(contentsOf: zip(qualities, neighborCounts).map { $0[0..<$1] })
////    }
////    let hScalars = Tensor<Float>(concatenating: h, alongAxis: 0).scalars
////    let labelLogits = LabelLogits(
////      logits: hScalars,
////      nodeCount: predictor.graph.nodeCount,
////      labelCount: predictor.graph.classCount)
////    let qualityLogits = q.map {
////      QualityLogits(
////        logits: $0.scalars,
////        nodeCount: $0.shape[0],
////        labelCount: predictor.graph.classCount)
////    }
////    return gibbsMarginalMAP(
////      labelLogits: labelLogits,
////      qualityLogits: qualityLogits,
////      graph: predictor.graph)
//  }
//
//  internal func labelsApproximateMAP(maxStepCount: Int = 100) -> [Int32] {
//    let nodes = [Int](0..<predictor.graph.nodeCount).map(Int32.init)
//    var h = [Tensor<Float>]()
//    var q = [Tensor<Float>]()
//    for batch in Dataset(elements: Tensor<Int32>(nodes)).batched(batchSize) {
//      let predictions = predictor.labelProbabilitiesAndQualities(batch.scalars)
//      let qualities = predictions.qualities.unstacked(alongAxis: 0)
//      let neighborCounts = predictions.qualitiesMask
//        .sum(squeezingAxes: -1)
//        .unstacked(alongAxis: 0)
//        .map { Int($0.scalarized()) }
//      h.append(predictions.labelProbabilities)
//      q.append(contentsOf: zip(qualities, neighborCounts).map { $0[0..<$1] })
//    }
//    let hScalars = h.count > 1 ? Tensor<Float>(concatenating: h, alongAxis: 0).scalars : h[0].scalars
//    let labelLogits = LabelLogits(
//      logits: hScalars,
//      nodeCount: predictor.graph.nodeCount,
//      labelCount: predictor.graph.classCount)
//    let qualityLogits = q.map {
//      QualityLogits(
//        logits: $0.scalars,
//        nodeCount: $0.shape[0],
//        labelCount: predictor.graph.classCount)
//    }
//    return iteratedConditionalModes(
//      labelLogits: labelLogits,
//      qualityLogits: qualityLogits,
//      graph: predictor.graph,
//      maxStepCount: maxStepCount)
//  }
//
//  internal func labelsMAP() -> [Int32] {
//    let nodes = [Int](0..<predictor.graph.nodeCount).map(Int32.init)
//    var h = [Tensor<Float>]()
//    var q = [Tensor<Float>]()
//    for batch in Dataset(elements: Tensor<Int32>(nodes)).batched(batchSize) {
//      let predictions = predictor.labelProbabilitiesAndQualities(batch.scalars)
//      let qualities = predictions.qualities.unstacked(alongAxis: 0)
//      let neighborCounts = predictions.qualitiesMask
//        .sum(squeezingAxes: -1)
//        .unstacked(alongAxis: 0)
//        .map { Int($0.scalarized()) }
//      h.append(predictions.labelProbabilities)
//      q.append(contentsOf: zip(qualities, neighborCounts).map { $0[0..<$1] })
//    }
//    let hScalars = Tensor<Float>(concatenating: h, alongAxis: 0).scalars
//    let labelLogits = LabelLogits(
//      logits: hScalars,
//      nodeCount: predictor.graph.nodeCount,
//      labelCount: predictor.graph.classCount)
//    let qualityLogits = q.map {
//      QualityLogits(
//        logits: $0.scalars,
//        nodeCount: $0.shape[0],
//        labelCount: predictor.graph.classCount)
//    }
//    return bestNodeAssignments(h: labelLogits, q: qualityLogits, G: predictor.graph)
//  }
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
