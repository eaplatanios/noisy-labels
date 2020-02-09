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

// 6: Modularity?
// 6: Computational power could have been a reason.
// 6: "..." when you fast forward in the time scale.
// 7: Trained using many examples is not necessarily the difference.
// 8: More like parametric vs nonparametric models.
// 8: I disagree that the meaning is lost. It's more about how you compress that meaning.
// 14: Only applies to context-free structures.
// 19: Function evaluation? Why is that interesting?
// 20: Tree-structured not chain-structured, right?
// 20: The one-hot embeddings are local representations, right?
// 22: How many examples do you need? How did you generate training data? What about curriculum?
// 27: This simply has to do with moving complexity around to programming language features.
// 28: Shouldn't it be the same stack for everyone?
// 33: Is Sympy good? How does a better solver like Mathematica do?
// 44: Make that a single animation.
// 45: Yoda speaking.
// 45: Why use the right to left notation in your slides?
// 45: I don't get this example in the end? Can you keep using a running example instead of letter symbols?
// 46: What about uncertainty in k_t? Isn't that a big challenge with symbolic systems too?
// 47: Do these representations really generalize? As in, what does a vector mean when it lies between three logical expressions?
// 49: Mizaitis and Mitchel are misspelled in the bottom.
// 50: Underscore?
// 50: How many did you have in total? Was the domain limited? How were the data collected?
// 53: Does no feedback also means no supervision at all?
// 57: Mizaitis in the bottom.
// 58: Mitchel in the bottom.
// 58: Make it prettier.
// 61: P(Y|X) and P(Y, X) are not needed. Just explain what the two terms mean intuitively.
// 62: Weird jump back to the math part and the motivation for that.



// Symbolic solvers disadvantage: slow because of discrete search with heuristics.
// Maybe mention big challenges that symbolic solvers cannot handle.

// How do these math problems relate to difficult real-world problems with noisy structures?
// Why talk about the math stuff if there are no future directions or connections to future directions for that part?







// 16: Can be compressed by just evaluating that expression. How much information do you want to keep? Do you want to be able to reconstruct the original expression?
// 23: What is the representation of the prediction? Is that a structured prediction problem?
// 25: Are the mistakes between LSTM and Sympy the same? What do they look like?
// 25: How do you determine failure in the symbolic solver? Is it time based?
// 23: Bi-directional LSTM? Transformer? That would be more about directionality rather than about hierarchical structure in the data.




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
//    let neighborY = exp(predictions.neighborLabelLogits)                                            // [BatchSize, MaxNeighborCount, ClassCount]
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
        let graphDepth = (emStep / 10) + 1
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
          softmaxCrossEntropy(
            logits: predictor.labelLogitsHelper(forNodes: batch.nodes, using: graph),
            probabilities: labels)
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

  private mutating func performEStep(using subGraph: SubGraph, randomSeed: Int64) {
    // Set the labeled node labels to their true labels.
    for batch in Dataset(elements: subGraph.labeledData).batched(batchSize) {
      expectedLabels = _Raw.tensorScatterUpdate(
        expectedLabels,
        indices: batch.nodes.expandingShape(at: -1),
        updates: Tensor<Float>(
          oneHotAtIndices: batch.labels,
          depth: subGraph.classCount,
          onValue: 0,
          offValue: -100000000))
    }

    // Compute expectations for the labels of the unlabeled nodes.
    let unlabeledData = Dataset(elements: subGraph.unlabeledNodesTensor)
      .shuffled(sampleCount: 10000, randomSeed: randomSeed)
      .batched(batchSize)
    for batch in unlabeledData {
      let predictions = predictor.predictions(forNodes: batch, using: subGraph.graph)
      let neighborY = exp(expectedLabels.gathering(atIndices: predictions.neighborIndices))         // [BatchSize, MaxNeighborCount, ClassCount]
      let h = predictions.labelLogits                                                               // [BatchSize, ClassCount]
      let g = predictions.qualityLogits                                                             // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
      let gMask = predictions.neighborMask                                                          // [BatchSize, MaxNeighborCount]
      let gY = (g * neighborY.expandingShape(at: 2)).sum(squeezingAxes: -1)                         // [BatchSize, MaxNeighborCount, ClassCount]
      let gYMasked = (gY * gMask.expandingShape(at: -1)).sum(squeezingAxes: 1)                      // [BatchSize, ClassCount]
      expectedLabels = _Raw.tensorScatterUpdate(
        expectedLabels,
        indices: batch.expandingShape(at: -1),
        updates: logSoftmax(h + gYMasked, alongAxis: -1))
    }
  }

  public mutating func performMStep(using subGraph: SubGraph, randomSeed: Int64, onlyH: Bool = false, onlyG: Bool = false) {
    let expectedLabels = self.expectedLabels
    var dataIterator = Dataset(elements: subGraph.nodesTensor).repeated()
      .shuffled(sampleCount: 10000, randomSeed: randomSeed)
      .batched(batchSize)
      .prefetched(count: 10)
      .makeIterator()
    var convergenceStepCount = 0
    bestResult = nil
    evaluationResultsAccumulator.reset()
    if !useWarmStarting && !onlyH && !onlyG {
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
          let gMask = predictions.neighborMask                                                      // [BatchSize, MaxNeighborCount]
          let gY = (g * neighborY.expandingShape(at: 2)).sum(squeezingAxes: -1)                     // [BatchSize, MaxNeighborCount, ClassCount]
          let gYMasked = (gY * gMask.expandingShape(at: -1)).sum(squeezingAxes: 1)                  // [BatchSize, ClassCount]
          if onlyH {
            let compilerBug = gYMasked.sum() * 0
            return compilerBug + softmaxCrossEntropy(
              logits: h + withoutDerivative(at: gYMasked) { $0 },
              probabilities: y,
              reduction: { $0.mean() })
          } else if onlyG {
            let compilerBug = h.sum() * 0
            return compilerBug + softmaxCrossEntropy(
              logits: withoutDerivative(at: h) { $0 } + gYMasked,
              probabilities: y,
              reduction: { $0.mean() })
          }
          return softmaxCrossEntropy(
            logits: h + gYMasked,
            probabilities: y,
            reduction: { $0.mean() })
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
        let usePrior = onlyH
        let evaluationResult = { () -> Result in
          if usePrior {
            return evaluate(model: self, using: graph, usePrior: true)
          } else {
            var modelCopy = self
            modelCopy.performEStep(
              using: SubGraph(graph: graph, mapFromOriginalIndex: nil),
              randomSeed: randomSeed)
            return evaluate(model: modelCopy, using: graph, usePrior: false)
          }
        }()
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
