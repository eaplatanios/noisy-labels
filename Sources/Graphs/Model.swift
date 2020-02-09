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

  public func initialSample(forGraph graph: Graph) -> [Int32] {
    switch self {
    case .groundTruth: return graph.labels.sorted { $0.key < $1.key }.map { Int32($0.value) }
    case .random:
      var sample = (0..<graph.nodeCount).map { _ in Int32.random(in: 0..<Int32(graph.classCount)) }
      for node in graph.trainNodes { sample[Int(node)] = Int32(graph.labels[node]!) }
      return sample
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

      return labelScores.argmax(squeezingAxis: -1).scalars
    }
  }
}

@differentiable(wrt: predictions)
public func estimateNormalizationConstant(
  using subGraph: SubGraph,
  predictions: Predictions,
  samples: [[[Int32]]]
) -> Tensor<Float> {
  _vjpEstimateNormalizationConstant(
    using: subGraph,
    predictions: predictions,
    samples: samples
  ).value
}

@derivative(of: estimateNormalizationConstant)
@usableFromInline
internal func _vjpEstimateNormalizationConstant(
  using subGraph: SubGraph,
  predictions: Predictions,
  samples: [[[Int32]]]
) -> (value: Tensor<Float>, pullback: (Tensor<Float>) -> Predictions.TangentVector) {
  let samples = samples.flatMap({ $0 })
  var value = Tensor<Float>(0)
  var gradient = Predictions.TangentVector.zero
  for sample in samples {
    let sampleTensor = Tensor<Int32>(subGraph.transformOriginalSample(sample))
    let (sampleValue, sampleGradient) = valueWithGradient(at: predictions) {
      predictions -> Tensor<Float> in
      let h = predictions.labelLogits                                                               // [BatchSize, ClassCount]
      let g = predictions.qualityLogits                                                             // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
      let gMask = predictions.neighborMask                                                          // [BatchSize, MaxNeighborCount]
      let gY = g.batchGathering(
        atIndices: sampleTensor.gathering(
          atIndices: predictions.neighborIndices
        ).expandingShape(at: -1),
        alongAxis: 3,
        batchDimensionCount: 2).squeezingShape(at: -1)                                              // [BatchSize, MaxNeighborCount, ClassCount]
      let gYMasked = (gY * gMask.expandingShape(at: -1)).sum(squeezingAxes: 1)                      // [BatchSize, ClassCount]
      return (h + gYMasked).batchGathering(
        atIndices: sampleTensor.expandingShape(at: -1),
        alongAxis: 1,
        batchDimensionCount: 1).sum() / Float(samples.count)
    }
    value += sampleValue
    gradient += sampleGradient
  }
  return (value: value, pullback: { _ in gradient })
}

public struct Model<Predictor: GraphPredictor, Optimizer: TensorFlow.Optimizer>
where Optimizer.Model == Predictor {
  public let randomSeed: Int64
  /// TODO: !!! Clarify that this is currently only a pre-training batch size.
  public let batchSize: Int
  public let useIncrementalNeighborhoodExpansion: Bool
  public let initializationMethod: ModelInitializationMethod
  public let stepCount: Int
  public let preTrainingStepCount: Int
  public let gibbsLikelihoodChainCount: Int
  public let gibbsLikelihoodSampleCount: Int
  public let gibbsLikelihoodBurnInSampleCount: Int
  public let gibbsLikelihoodThinningSampleCount: Int
  public let gibbsNormalizationChainCount: Int
  public let gibbsNormalizationSampleCount: Int
  public let gibbsNormalizationBurnInSampleCount: Int
  public let gibbsNormalizationThinningSampleCount: Int
  public let evaluationStepCount: Int?
  public let evaluationConvergenceStepCount: Int?
  public let stepCallback: (Model) -> ()
  public let logStepCount: Int?
  public let verbose: Bool

  public var predictor: Predictor
  public var optimizer: Optimizer
  public var evaluationResultsAccumulator: Accumulator

  private let graph: Graph

  /// The last labels sample is an array that contains arrays with the labels sampled in the last
  /// step, for each node in the graph and for each Gibbs sampling chain.
  private var lastLikelihoodSamples: [[Int32]]
  private var lastNormalizationSamples: [[Int32]]
  private var bestPredictor: Predictor
  private var bestResult: Result?

  public init(
    graph: Graph,
    predictor: Predictor,
    optimizer: Optimizer,
    randomSeed: Int64,
    batchSize: Int = 128,
    useIncrementalNeighborhoodExpansion: Bool = false,
    initializationMethod: ModelInitializationMethod = .labelPropagation,
    stepCount: Int = 1000,
    preTrainingStepCount: Int = 1000,
    gibbsLikelihoodChainCount: Int = 5,
    gibbsLikelihoodSampleCount: Int = 5,
    gibbsLikelihoodBurnInSampleCount: Int = 0,
    gibbsLikelihoodThinningSampleCount: Int = 0,
    gibbsNormalizationChainCount: Int = 5,
    gibbsNormalizationSampleCount: Int = 5,
    gibbsNormalizationBurnInSampleCount: Int = 10,
    gibbsNormalizationThinningSampleCount: Int = 0,
    evaluationStepCount: Int? = 1,
    evaluationConvergenceStepCount: Int? = 10,
    evaluationResultsAccumulator: Accumulator = ExactAccumulator(),
    stepCallback: @escaping (Model) -> () = { _ in () },
    logStepCount: Int? = 100,
    verbose: Bool = false
  ) {
    self.graph = graph
    self.predictor = predictor
    self.optimizer = optimizer
    self.randomSeed = randomSeed
    self.batchSize = batchSize
    self.useIncrementalNeighborhoodExpansion = useIncrementalNeighborhoodExpansion
    self.initializationMethod = initializationMethod
    self.stepCount = stepCount
    self.preTrainingStepCount = preTrainingStepCount
    self.gibbsLikelihoodChainCount = gibbsLikelihoodChainCount
    self.gibbsLikelihoodSampleCount = gibbsLikelihoodSampleCount
    self.gibbsLikelihoodBurnInSampleCount = gibbsLikelihoodBurnInSampleCount
    self.gibbsLikelihoodThinningSampleCount = gibbsLikelihoodThinningSampleCount
    self.gibbsNormalizationChainCount = gibbsNormalizationChainCount
    self.gibbsNormalizationSampleCount = gibbsNormalizationSampleCount
    self.gibbsNormalizationBurnInSampleCount = gibbsNormalizationBurnInSampleCount
    self.gibbsNormalizationThinningSampleCount = gibbsNormalizationThinningSampleCount
    self.evaluationStepCount = evaluationStepCount
    self.evaluationConvergenceStepCount = evaluationConvergenceStepCount
    self.evaluationResultsAccumulator = evaluationResultsAccumulator
    self.stepCallback = stepCallback
    self.logStepCount = logStepCount
    self.verbose = verbose
    self.lastLikelihoodSamples = [[Int32]]()
    self.lastLikelihoodSamples.reserveCapacity(gibbsLikelihoodChainCount)
    self.lastLikelihoodSamples.append(initializationMethod.initialSample(forGraph: graph))
    for _ in 1..<gibbsLikelihoodChainCount {
      self.lastLikelihoodSamples.append(ModelInitializationMethod.random.initialSample(
        forGraph: graph))
    }
    self.lastNormalizationSamples = [[Int32]]()
    self.lastNormalizationSamples.reserveCapacity(gibbsNormalizationChainCount)
    for _ in 0..<gibbsNormalizationChainCount {
      self.lastNormalizationSamples.append(ModelInitializationMethod.random.initialSample(
        forGraph: graph))
    }
    self.bestPredictor = predictor
    self.bestResult = nil
  }

  public func labelLogits(forNodes nodes: [Int32], usePrior: Bool = false) -> Tensor<Float> {
    let nodes = Tensor<Int32>(nodes)
    if usePrior { return predictor.labelLogits(forNodes: nodes, using: graph) }

    // Estimate MAP using iterated conditional modes.
    // The following is inefficient.
    let predictions = InMemoryPredictions(
      fromPredictions: predictor.predictionsHelper(forNodes: graph.nodesTensor, using: graph),
      using: graph)
    return Tensor<Float>(
      oneHotAtIndices: Tensor<Int32>(iteratedConditionalModes(
        labelLogits: predictions.labelLogits,
        qualityLogits: predictions.qualityLogits,
        graph: graph,
        maxStepCount: 100)
      ).gathering(atIndices: nodes),
      depth: graph.classCount)
  }

  public mutating func train() {
    if preTrainingStepCount > 0 {
      if verbose { logger.info("Starting model pre-training.") }
      preTrainLabelsPredictor()
    }

    if verbose { logger.info("Starting model training.") }
    // trainLabelPredictors()
    stepCallback(self)

    // Burn some samples before starting to use them for estimating the likelihood function.
    if gibbsLikelihoodBurnInSampleCount > 0 {
      let predictions = predictor.predictionsHelper(forNodes: graph.nodesTensor, using: graph)
      let inMemoryPredictions = withoutDerivative(at: predictions) {
        InMemoryPredictions(fromPredictions: $0, using: graph)
      }
      for _ in 0..<gibbsLikelihoodBurnInSampleCount {
        lastLikelihoodSamples = sampleLabels(
          using: SubGraph(graph: graph, mapFromOriginalIndex: nil),
          predictions: inMemoryPredictions,
          previousSamples: lastLikelihoodSamples,
          sampleTrainLabels: false)
      }
    }

    var previousGraphExpansion = 0
    var subGraph = SubGraph(graph: graph, mapFromOriginalIndex: nil)
    var nodes = subGraph.nodesTensor
    var convergenceStepCount = 0
    bestResult = nil
    evaluationResultsAccumulator.reset()
    var accumulatedNLL = Float(0.0)
    var accumulatedSteps = 0
    for step in 0..<stepCount {
      if useIncrementalNeighborhoodExpansion {
        // TODO: !!! Make this schedule configurable.
        let graphDepth = (step / 10) + 1
        if previousGraphExpansion < graphDepth {
          previousGraphExpansion = graphDepth
          subGraph = graph.subGraph(upToDepth: graphDepth)
          nodes = subGraph.nodesTensor
          logger.info("Training on \(subGraph.nodeCount) / \(graph.nodeCount) nodes.")
        }
      }
      let (negativeLogLikelihood, gradient) = valueWithGradient(at: predictor) {
        predictor -> Tensor<Float> in
        let predictions = predictor.predictionsHelper(forNodes: nodes, using: subGraph.graph)
        let inMemoryPredictions = withoutDerivative(at: predictions) {
          InMemoryPredictions(fromPredictions: $0, using: subGraph.graph)
        }

        let h = predictions.labelLogits
        let g = predictions.qualityLogits                                                           // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
        let gMask = predictions.neighborMask                                                        // [BatchSize, MaxNeighborCount]

        let likelihoodSamples = sampleMultipleLabels(
          using: subGraph,
          predictions: inMemoryPredictions,
          previousSamples: lastLikelihoodSamples,
          sampleTrainLabels: false,
          sampleCount: gibbsLikelihoodSampleCount,
          burnInSampleCount: gibbsLikelihoodBurnInSampleCount,
          thinningSampleCount: gibbsLikelihoodThinningSampleCount)
        var negativeLogLikelihood = (h * exp(h)).sum() * 0 // TODO: !!!
        for samples in likelihoodSamples {
          for sample in samples {
            let y = Tensor<Int32>(subGraph.transformOriginalSample(sample))
            let neighborY = y.gathering(atIndices: predictions.neighborIndices)                     // [BatchSize, MaxNeighborCount, ClassCount]
            let gY = g.batchGathering(
              atIndices: neighborY.expandingShape(at: -1),
              alongAxis: 3,
              batchDimensionCount: 2).squeezingShape(at: -1)                                        // [BatchSize, MaxNeighborCount, ClassCount]
            let gYMasked = (gY * gMask.expandingShape(at: -1)).sum(squeezingAxes: 1)                // [BatchSize, ClassCount]
            negativeLogLikelihood = negativeLogLikelihood - h.batchGathering(
              atIndices: y.expandingShape(at: -1),
              alongAxis: 1,
              batchDimensionCount: 1).sum()
            negativeLogLikelihood = negativeLogLikelihood - gYMasked.batchGathering(
              atIndices: y.expandingShape(at: -1),
              alongAxis: 1,
              batchDimensionCount: 1).sum()
          }
          lastLikelihoodSamples = samples
        }

        let likelihoodSampleCount = gibbsLikelihoodSampleCount * gibbsLikelihoodChainCount
        negativeLogLikelihood = negativeLogLikelihood / Float(likelihoodSampleCount)

        // Compute the normalization term in the negative log-likelihood.
        let normalizationSamples = sampleMultipleLabels(
          using: subGraph,
          predictions: inMemoryPredictions,
          previousSamples: lastNormalizationSamples,
          sampleTrainLabels: true,
          sampleCount: gibbsNormalizationSampleCount,
          burnInSampleCount: gibbsNormalizationBurnInSampleCount,
          thinningSampleCount: gibbsNormalizationThinningSampleCount)
        negativeLogLikelihood = negativeLogLikelihood + estimateNormalizationConstant(
          using: subGraph,
          predictions: predictions,
          samples: normalizationSamples)
        lastNormalizationSamples = normalizationSamples.last!

        // Update the last sample.
        for _ in 0..<gibbsLikelihoodThinningSampleCount {
          lastLikelihoodSamples = sampleLabels(
            using: subGraph,
            predictions: inMemoryPredictions,
            previousSamples: lastLikelihoodSamples,
            sampleTrainLabels: false)
        }

        return negativeLogLikelihood
      }

      optimizer.update(&predictor, along: gradient)
      accumulatedNLL += negativeLogLikelihood.scalarized()
      accumulatedSteps += 1
      if verbose {
        if let logStepCount = self.logStepCount, step % logStepCount == 0 || step == stepCount - 1 {
          let nll = accumulatedNLL / Float(accumulatedSteps)
          let message = "Step \(String(format: "%5d", step)) " +
            "Negative Log-Likelihood: \(String(format: "%.8f", nll))"
          logger.info("\(message)")
          accumulatedNLL = 0.0
          accumulatedSteps = 0
        }
      }

      // Check for early stopping.
      if let evaluationStepCount = self.evaluationStepCount, step % evaluationStepCount == 0 {
        let evaluationResult = evaluate(model: self, using: graph, usePrior: true)
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

      stepCallback(self)
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
          if let logStepCount = self.logStepCount, step % logStepCount == 0 ||
            step == stepCount - 1 {
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

  private func sampleLabels(
    using subGraph: SubGraph,
    predictions: InMemoryPredictions,
    previousSamples: [[Int32]],
    sampleTrainLabels: Bool = true
  ) -> [[Int32]] {
    let nodes = sampleTrainLabels ? subGraph.nodes : subGraph.unlabeledNodes
    var currentSamples = previousSamples
    for chain in 0..<previousSamples.count {
      for node in nodes.shuffled() {
        let neighbors = subGraph.neighbors[Int(node)]
        let g = predictions.qualityLogits[Int(node)]
        let gT = predictions.qualityLogitsTransposed[Int(node)]
        var labelProbabilities = predictions.labelLogits.labelLogits(forNode: Int(node))
        for (neighborIndex, neighbor) in neighbors.enumerated() {
          let originalNeighborIndex = Int(subGraph.originalIndex(ofNode: neighbor))
          let neighborLabel = Int(currentSamples[chain][originalNeighborIndex])
          for k in 0..<subGraph.classCount {
            labelProbabilities[k] += g.qualityLogit(
              forNeighbor: Int(neighborIndex),
              nodeLabel: k,
              neighborLabel: neighborLabel) / Float(neighbors.count)
            labelProbabilities[k] += gT.qualityLogit(
              forNeighbor: Int(neighborIndex),
              nodeLabel: neighborLabel,
              neighborLabel: k) / Float(neighbors.count)
          }
        }
        let labelProbabilitiesMax = labelProbabilities.max()!
        var sum = Float(0)
        for k in 0..<subGraph.classCount {
          labelProbabilities[k] = exp(labelProbabilities[k] - labelProbabilitiesMax)
          sum += labelProbabilities[k]
        }
        // TODO: !!! Seed / random number generator.
        let random = sum > 0 ?
          Float.random(in: 0..<sum) :
          Float.random(in: 0..<Float(subGraph.classCount))
        var accumulator: Float = 0
        for k in 0..<subGraph.classCount {
          accumulator += sum > 0 ? labelProbabilities[k] : 1
          if random < accumulator {
            currentSamples[chain][Int(subGraph.originalIndex(ofNode: node))] = Int32(k)
            break
          }
        }
      }
    }
    return currentSamples
  }

  private func sampleMultipleLabels(
    using subGraph: SubGraph,
    predictions: InMemoryPredictions,
    previousSamples: [[Int32]],
    sampleTrainLabels: Bool = true,
    sampleCount: Int = 10,
    burnInSampleCount: Int = 10,
    thinningSampleCount: Int = 0
  ) -> [[[Int32]]] {
    var samples = [[[Int32]]](repeating: [[Int32]](), count: previousSamples.count)
    var currentSamples = previousSamples
    var currentSampleCount = 0
    while samples[0].count < sampleCount {
      currentSamples = sampleLabels(
        using: subGraph,
        predictions: predictions,
        previousSamples: currentSamples,
        sampleTrainLabels: sampleTrainLabels)
      currentSampleCount += 1
      if currentSampleCount > burnInSampleCount &&
           currentSampleCount.isMultiple(of: thinningSampleCount + 1) {
        for chain in 0..<previousSamples.count {
          samples[chain].append(currentSamples[chain])
        }
      }
    }
    return samples
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
