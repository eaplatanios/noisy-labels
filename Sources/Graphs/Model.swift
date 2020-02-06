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

  public func initialSample(forGraph graph: Graph) -> Tensor<Int32> {
    switch self {
    case .groundTruth: return Tensor<Int32>(
      graph.labels.sorted { $0.key < $1.key }.map { Int32($0.value) })
    case .random:
      var sample = Tensor<Int32>(
        randomUniform: [graph.nodeCount],
        lowerBound: Tensor<Int32>(0),
        upperBound: Tensor<Int32>(Int32(graph.classCount)))

      // Start with the labeled nodes.
      for node in graph.trainNodes {
        sample = _Raw.tensorScatterUpdate(
          sample,
          indices: Tensor<Int32>([node]).expandingShape(at: -1),
          updates: Tensor<Int32>([Int32(graph.labels[node]!)]))
      }

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

      return labelScores.argmax(squeezingAxis: -1)
    }
  }
}

@differentiable(wrt: predictions)
public func estimateNormalizationConstant(
  using predictions: Predictions,
  samples: [Tensor<Int32>]
) -> Tensor<Float> {
  _vjpEstimateNormalizationConstant(using: predictions, samples: samples).value
}

@derivative(of: estimateNormalizationConstant)
@usableFromInline
internal func _vjpEstimateNormalizationConstant(
  using predictions: Predictions,
  samples: [Tensor<Int32>]
) -> (value: Tensor<Float>, pullback: (Tensor<Float>) -> Predictions.TangentVector) {
  var value = Tensor<Float>(0)
  var gradient = Predictions.TangentVector.zero
  for sample in samples {
    let (sampleValue, sampleGradient) = valueWithGradient(at: predictions) {
      predictions -> Tensor<Float> in
      let h = predictions.labelLogits                                                               // [BatchSize, ClassCount]
      let g = predictions.qualityLogits                                                             // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
      let gMask = predictions.neighborMask                                                          // [BatchSize, MaxNeighborCount]
      let gY = g.batchGathering(
        atIndices: sample.gathering(atIndices: predictions.neighborIndices).expandingShape(at: -1),
        alongAxis: 3,
        batchDimensionCount: 2).squeezingShape(at: -1)                                              // [BatchSize, MaxNeighborCount, ClassCount]
      let gYMasked = (gY * gMask.expandingShape(at: -1)).sum(squeezingAxes: 1)                      // [BatchSize, ClassCount]
      return (h + gYMasked).batchGathering(
        atIndices: sample.expandingShape(at: -1),
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
  public let batchSize: Int
  public let useIncrementalNeighborhoodExpansion: Bool
  public let initializationMethod: ModelInitializationMethod
  public let stepCount: Int
  public let evaluationStepCount: Int?
  public let evaluationConvergenceStepCount: Int?
  public let stepCallback: (Model) -> ()
  public let logStepCount: Int?
  public let verbose: Bool

  public var predictor: Predictor
  public var optimizer: Optimizer
  public var evaluationResultsAccumulator: Accumulator

  /// The last labels sample is a tensor with shape `[NodeCount]` that contains the label sampled
  /// in the last step, for each node in the graph.
  private var lastSample: Tensor<Int32>
  private var bestPredictor: Predictor
  private var bestResult: Result?

  public var graph: Graph { predictor.graph }

  public init(
    predictor: Predictor,
    optimizer: Optimizer,
    randomSeed: Int64,
    batchSize: Int = 128,
    useIncrementalNeighborhoodExpansion: Bool = false,
    initializationMethod: ModelInitializationMethod = .labelPropagation,
    stepCount: Int = 1000,
    evaluationStepCount: Int? = 1,
    evaluationConvergenceStepCount: Int? = 10,
    evaluationResultsAccumulator: Accumulator = ExactAccumulator(),
    stepCallback: @escaping (Model) -> () = { _ in () },
    logStepCount: Int? = 100,
    verbose: Bool = false
  ) {
    self.predictor = predictor
    self.optimizer = optimizer
    self.randomSeed = randomSeed
    self.batchSize = batchSize
    self.useIncrementalNeighborhoodExpansion = useIncrementalNeighborhoodExpansion
    self.initializationMethod = initializationMethod
    self.stepCount = stepCount
    self.evaluationStepCount = evaluationStepCount
    self.evaluationConvergenceStepCount = evaluationConvergenceStepCount
    self.evaluationResultsAccumulator = evaluationResultsAccumulator
    self.stepCallback = stepCallback
    self.logStepCount = logStepCount
    self.verbose = verbose
    self.lastSample = initializationMethod.initialSample(forGraph: predictor.graph)
    self.bestPredictor = predictor
    self.bestResult = nil
  }

  public func labelLogits(forNodes nodes: [Int32], usePrior: Bool = false) -> Tensor<Float> {
    let nodes = Tensor<Int32>(nodes)
    if usePrior { return predictor.labelLogits(nodes) }

    // Estimate MAP using iterated conditional modes.
    // The following is inefficient.
    let predictions = predictor.predictions(forNodes: graph.allNodes)
    let neighborCounts = predictions.neighborMask
      .sum(squeezingAxes: -1)
      .unstacked(alongAxis: 0)
      .map { Int($0.scalarized()) }
    let h = predictions.labelLogits.scalars
    let q = zip(predictions.qualityLogits.unstacked(alongAxis: 0), neighborCounts).map {
      $0[0..<$1]
    }
    let labelLogits = LabelLogits(
      logits: h,
      nodeCount: graph.nodeCount,
      labelCount: graph.classCount)
    let qualityLogits = q.map {
      QualityLogits(
        logits: $0.scalars,
        nodeCount: $0.shape[0],
        labelCount: predictor.graph.classCount)
    }
    return Tensor<Float>(
      oneHotAtIndices: Tensor<Int32>(iteratedConditionalModes(
        labelLogits: labelLogits,
        qualityLogits: qualityLogits,
        graph: graph,
        maxStepCount: 100)
      ).gathering(atIndices: nodes),
      depth: graph.classCount)
  }

  public mutating func train() {
    if verbose { logger.info("Starting model training.") }
    // trainLabelPredictors()
    stepCallback(self)

    let nodes = graph.allNodes
    var convergenceStepCount = 0
    bestResult = nil
    evaluationResultsAccumulator.reset()
    var accumulatedNLL = Float(0.0)
    var accumulatedSteps = 0
    for step in 0..<stepCount {
      let y = lastSample                                                                            // [BatchSize]
      let (negativeLogLikelihood, gradient) = valueWithGradient(at: predictor) {
        predictor -> Tensor<Float> in
        let predictions = predictor.predictions(forNodes: nodes)

        // Compute the first term in the negative log-likelihood.
        let h = predictions.labelLogits
        let g = predictions.qualityLogits                                                           // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
        let gMask = predictions.neighborMask                                                        // [BatchSize, MaxNeighborCount]
        let neighborY = lastSample.gathering(atIndices: predictions.neighborIndices)                // [BatchSize, MaxNeighborCount, ClassCount]
        let gY = g.batchGathering(
          atIndices: neighborY.expandingShape(at: -1),
          alongAxis: 3,
          batchDimensionCount: 2).squeezingShape(at: -1)                                            // [BatchSize, MaxNeighborCount, ClassCount]
        let gYMasked = (gY * gMask.expandingShape(at: -1)).sum(squeezingAxes: 1)                    // [BatchSize, ClassCount]
        var negativeLogLikelihood = -h.batchGathering(
          atIndices: y.expandingShape(at: -1),
          alongAxis: 1,
          batchDimensionCount: 1).sum()
        negativeLogLikelihood = negativeLogLikelihood - gYMasked.batchGathering(
          atIndices: y.expandingShape(at: -1),
          alongAxis: 1,
          batchDimensionCount: 1).sum()

        // Compute the normalization term in the negative log-likelihood.
        negativeLogLikelihood = negativeLogLikelihood + estimateNormalizationConstant(
          using: predictions,
          samples: withoutDerivative(at: predictions) {
            sampleMultipleLabels(using: $0, previousSample: lastSample, sampleTrainLabels: true)
          })

        // Update the last sample.
        lastSample = withoutDerivative(at: predictions) {
          for _ in 0..<10 {
            lastSample = sampleLabels(
              using: $0,
              previousSample: lastSample,
              sampleTrainLabels: false)
          }
          return lastSample
        }

        // This is due to a compiler bug related to automatic differentiation.
        negativeLogLikelihood = negativeLogLikelihood + predictions.qualityLogitsTransposed.sum() * 0

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
        let evaluationResult = evaluate(model: self, usePrior: true)
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

  private func sampleLabels(
    using predictions: Predictions,
    previousSample: Tensor<Int32>,
    sampleTrainLabels: Bool = true
  ) -> Tensor<Int32> {
    var currentSample = previousSample
    let data = sampleTrainLabels ?
      [graph.allNodes] :
      useIncrementalNeighborhoodExpansion ?
        graph.leveledData.suffix(from: 1).map { Tensor<Int32>($0) } :
        [graph.unlabeledData]
    for batch in data {
      let h = predictions.labelLogits.gathering(atIndices: batch)
      let g = predictions.qualityLogits.gathering(atIndices: batch)                                 // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
      let gT = predictions.qualityLogitsTransposed.gathering(atIndices: batch)                      // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
      let gMask = predictions.neighborMask.gathering(atIndices: batch)                              // [BatchSize, MaxNeighborCount]
      let neighborY = currentSample.gathering(
        atIndices: predictions.neighborIndices.gathering(atIndices: batch)
      ).expandingShape(at: -1)                                                                      // [BatchSize, MaxNeighborCount, 1]
      let gY = g.batchGathering(
        atIndices: neighborY,
        alongAxis: 3,
        batchDimensionCount: 2).squeezingShape(at: -1)                                              // [BatchSize, MaxNeighborCount, ClassCount]
      let gTY = gT.batchGathering(
        atIndices: neighborY,
        alongAxis: 2,
        batchDimensionCount: 2).squeezingShape(at: -2)                                              // [BatchSize, MaxNeighborCount, ClassCount]
      let gYMasked = (gY * gMask.expandingShape(at: -1)).sum(squeezingAxes: 1)                      // [BatchSize, ClassCount]
      let gTYMasked = (gTY * gMask.expandingShape(at: -1)).sum(squeezingAxes: 1)                    // [BatchSize, ClassCount]
      currentSample = _Raw.tensorScatterUpdate(
        currentSample,
        indices: batch.expandingShape(at: -1),
        updates: Tensor<Int32>(
          randomCategorialLogits: h + gYMasked + gTYMasked,
          sampleCount: 1).squeezingShape(at: -1))
    }
    return currentSample
  }

  private func sampleMultipleLabels(
    using predictions: Predictions,
    previousSample: Tensor<Int32>? = nil,
    sampleTrainLabels: Bool = true,
    sampleCount: Int = 10,
    burnInSampleCount: Int = 10,
    thinningSampleCount: Int = 0
  ) -> [Tensor<Int32>] {
    var currentSample = previousSample ?? initializationMethod.initialSample(forGraph: graph)
    var currentSampleCount = 0
    var samples = [Tensor<Int32>]()
    samples.reserveCapacity(sampleCount)
    while samples.count < sampleCount {
      currentSample = sampleLabels(
        using: predictions,
        previousSample: currentSample,
        sampleTrainLabels: sampleTrainLabels)
      currentSampleCount += 1
      if currentSampleCount > burnInSampleCount &&
           currentSampleCount.isMultiple(of: thinningSampleCount + 1) {
        samples.append(currentSample)
      }
    }
    return samples
  }

//  private mutating func performMStep(data: Dataset<LabeledData>, emStep: Int) {
//    bestResult = nil
//    resultAccumulator.reset()
//    if !useWarmStarting {
//      predictor.reset()
//      optimizer = optimizerFn()
//    }
//    var accumulatedLoss = Float(0.0)
//    var accumulatedSteps = 0
//    var dataIterator = data.repeated()
//      .shuffled(sampleCount: 10000, randomSeed: randomSeed &+ Int64(emStep))
//      .batched(batchSize)
//      .prefetched(count: 10)
//      .makeIterator()
//    for mStep in 0..<mStepCount {
//      let batch = dataIterator.next()!
//      let labels = (1 - labelSmoothing) * Tensor<Float>(
//        oneHotAtIndices: batch.nodeLabels,
//        depth: predictor.classCount
//      ) + labelSmoothing / Float(predictor.classCount)
//      withLearningPhase(.training) {
//        let (loss, gradient) = valueWithGradient(at: predictor) { predictor -> Tensor<Float> in
//          let predictions = predictor.labelProbabilitiesAndQualities(batch.nodeIndices.scalars)
//          let crossEntropy = softmaxCrossEntropy(
//            logits: predictions.labelProbabilities,
//            probabilities: labels)
//          let loss = crossEntropy +
//            predictions.qualities.sum() * 0.0 +
//            predictions.qualitiesTranspose.sum() * 0.0
//          return loss / Float(predictions.labelProbabilities.shape[0])
//        }
//        optimizer.update(&predictor, along: gradient)
//        accumulatedLoss += loss.scalarized()
//        accumulatedSteps += 1
//        if verbose {
//          if let logSteps = mStepLogCount, mStep % logSteps == 0 || mStep == mStepCount - 1 {
//            let nll = accumulatedLoss / Float(accumulatedSteps)
//            let message = "Supervised M-Step \(String(format: "%5d", mStep)) | " +
//              "Loss: \(String(format: "%.8f", nll))"
//            logger.info("\(message)")
//            accumulatedLoss = 0.0
//            accumulatedSteps = 0
//          }
//        }
//      }
//      if let c = evaluationStepCount, mStep % c == 0 {
//        let result = resultAccumulator.update(
//          with: evaluate(model: self, using: predictor.graph, usePrior: true))
//        if let bestResult = self.bestResult {
//          if result.validationAccuracy > bestResult.validationAccuracy {
//            self.bestExpectedLabels = expectedLabels
//            self.bestPredictor = predictor
//            self.bestResult = result
//          }
//        } else {
//          self.bestExpectedLabels = expectedLabels
//          self.bestPredictor = predictor
//          self.bestResult = result
//        }
//      }
//    }
//    if evaluationStepCount != nil {
//      expectedLabels = bestExpectedLabels
//      predictor = bestPredictor
//    }
//  }
//
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
