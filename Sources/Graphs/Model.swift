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

public struct Model<Predictor: GraphPredictor, Optimizer: TensorFlow.Optimizer>
where Optimizer.Model == Predictor {
  public let entropyWeight: Float
  public let qualitiesRegularizationWeight: Float

  public let optimizerFn: () -> Optimizer
  public let randomSeed: Int64
  public let batchSize: Int
  public let useWarmStarting: Bool
  public let useThresholdedExpectations: Bool
  public let useIncrementalNeighborhoodExpansion: Bool
  public let labelSmoothing: Float
  public let mStepCount: Int
  public let emStepCount: Int
  public let evaluationStepCount: Int?
  public let mStepLogCount: Int?
  public let mConvergenceEvaluationCount: Int?
  public let emStepCallback: (Model) -> Bool
  public let verbose: Bool

  public var resultAccumulator: Accumulator

  public var predictor: Predictor
  public var optimizer: Optimizer
  public var lastLabelsSample: Tensor<Int32>   // [NodeCount]
  public var expectedLabels: Tensor<Float>     // [NodeCount, ClassCount]

  public var bestExpectedLabels: Tensor<Float>
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
    useIncrementalNeighborhoodExpansion: Bool = false,
    labelSmoothing: Float = 0.0,
    resultAccumulator: Accumulator = ExactAccumulator(),
    mStepCount: Int = 1000,
    emStepCount: Int = 100,
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
    self.useIncrementalNeighborhoodExpansion = useIncrementalNeighborhoodExpansion
    self.labelSmoothing = labelSmoothing
    self.mStepCount = mStepCount
    self.emStepCount = emStepCount
    self.evaluationStepCount = evaluationStepCount
    self.mStepLogCount = mStepLogCount
    self.mConvergenceEvaluationCount = mConvergenceEvaluationCount
    self.emStepCallback = emStepCallback
    self.verbose = verbose
    self.lastLabelsSample = Tensor<Int32>(zeros: [predictor.nodeCount])
    self.expectedLabels = Tensor<Float>(zeros: [predictor.nodeCount, predictor.classCount])
    self.bestExpectedLabels = expectedLabels
    self.bestPredictor = predictor
    self.bestResult = nil
  }

  public mutating func train(using graph: Graph) {
    var mStepData = Dataset(elements: graph.allUnlabeledData)

    if verbose { logger.info("Initialization") }
    initialize(using: graph)
//    performMStep(data: Dataset(elements: graph.labeledData), emStep: 0)
    let _ = emStepCallback(self)

    for emStep in 0..<emStepCount {
      if useIncrementalNeighborhoodExpansion {
        mStepData = Dataset(elements: graph.data(atDepth: emStep + 1))
      }

      // M-Step
      if verbose { logger.info("Iteration \(emStep) - Running M-Step") }
      performMStep(data: mStepData, emStep: emStep)

      // E-Step
      if verbose { logger.info("Iteration \(emStep) - Running E-Step") }
      for _ in 0..<10 { performEStep(using: graph) }

      if emStepCallback(self) { break }
    }
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
    expectedLabels = Tensor<Float>(zeros: [predictor.nodeCount, predictor.classCount])
    lastLabelsSample = Tensor<Int32>(
      randomUniform: [predictor.nodeCount],
      lowerBound: Tensor<Int32>(0),
      upperBound: Tensor<Int32>(Int32(predictor.classCount)))

    // Start with the labeled nodes.
    for node in graph.trainNodes {
      expectedLabels = _Raw.tensorScatterUpdate(
        expectedLabels,
        indices: Tensor<Int32>([node]).expandingShape(at: -1),
        updates: Tensor<Float>(
          oneHotAtIndices: Tensor<Int32>(Int32(graph.labels[node]!)),
          depth: predictor.classCount).expandingShape(at: 0))
    }

    for batch in Dataset(elements: graph.unlabeledData).batched(batchSize) {
      expectedLabels = _Raw.tensorScatterUpdate(
        expectedLabels,
        indices: batch.expandingShape(at: -1),
        updates: Tensor<Float>(
          repeating: 1 / Float(graph.classCount),
          shape: [batch.shape[0], graph.classCount]))
    }
  }

//  private mutating func initialize(using graph: Graph) {
//    expectedLabels = Tensor<Float>(zeros: [predictor.nodeCount, predictor.classCount])
//    var labeledNodes = Set<Int32>()
//    var nodesToLabel = Set<Int32>()
//
//    // Start with the labeled nodes.
//    for node in graph.trainNodes {
//      let label = graph.labels[node]!
//      expectedLabels = _Raw.tensorScatterAdd(
//        expectedLabels,
//        indices: Tensor<Int32>([node]).expandingShape(at: -1),
//        updates: Tensor<Float>(
//          oneHotAtIndices: Tensor<Int32>(Int32(label)),
//          depth: predictor.classCount).expandingShape(at: 0))
//      labeledNodes.update(with: node)
//      nodesToLabel.remove(node)
//      graph.neighbors[Int(node)].forEach {
//        if !labeledNodes.contains($0) {
//          nodesToLabel.update(with: $0)
//        }
//      }
//    }
//
//    // Proceed with label propagation for the unlabeled nodes.
//    while !nodesToLabel.isEmpty {
//      for node in nodesToLabel {
//        let labeledNeighbors = graph.neighbors[Int(node)].filter(labeledNodes.contains)
//        var probabilities = Tensor<Float>(zeros: [graph.classCount])
//        for neighbor in labeledNeighbors {
//          probabilities += expectedLabels[Int(neighbor)]
//        }
//        probabilities /= probabilities.sum()
//        expectedLabels = _Raw.tensorScatterAdd(
//          expectedLabels,
//          indices: Tensor<Int32>([node]).expandingShape(at: -1),
//          updates: probabilities.expandingShape(at: 0))
//      }
//      for node in nodesToLabel {
//        labeledNodes.update(with: node)
//        nodesToLabel.remove(node)
//        graph.neighbors[Int(node)].forEach {
//          if !labeledNodes.contains($0) {
//            nodesToLabel.update(with: $0)
//          }
//        }
//      }
//    }
//
//    for node in (0..<Int32(graph.nodeCount)).filter({ !labeledNodes.contains($0) }) {
//      expectedLabels = _Raw.tensorScatterAdd(
//        expectedLabels,
//        indices: Tensor<Int32>([node]).expandingShape(at: -1),
//        updates: Tensor<Float>(
//          repeating: 1 / Float(graph.classCount),
//          shape: [1, graph.classCount]))
//    }
//
//    // expectedLabels = useThresholdedExpectations ?
//    //   Tensor<Float>(expectedLabels .== expectedLabels.max(alongAxes: -1)) :
//    //   expectedLabels
//
//    lastLabelsSample = expectedLabels.argmax(squeezingAxis: -1)
//  }

  private mutating func performEStep(using graph: Graph) {
    // Use the provided labels for labeled nodes.
    for batch in Dataset(elements: graph.labeledData).batched(batchSize) {
      expectedLabels = _Raw.tensorScatterUpdate(
        expectedLabels,
        indices: batch.nodeIndices.expandingShape(at: -1),
        updates: Tensor<Float>(oneHotAtIndices: batch.nodeLabels, depth: predictor.classCount))
    }

    let leveledData = graph.leveledData
      .suffix(from: 1)
      .map { Dataset(elements: Tensor<Int32>($0)) }
    for batch in Dataset(elements: graph.unlabeledData).batched(batchSize) {
//    for level in leveledData {
//      for batch in level.batched(batchSize) {
        let predictions = predictor(batch.scalars)
        let h = predictions.labelProbabilities
        let q = predictions.qualities                                                               // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
        let qTranspose = predictions.qualitiesTranspose                                             // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
        let qMask = predictions.qualitiesMask.expandingShape(at: -1)                                // [BatchSize, MaxNeighborCount, 1]
        let yHat = expectedLabels.gathering(atIndices: predictions.neighborIndices)                 // [BatchSize, MaxNeighborCount, ClassCount]
        let qYHat = ((q * yHat.expandingShape(at: 2)).sum(squeezingAxes: -1) * qMask).sum(squeezingAxes: 1)                   // [BatchSize, ClassCount]
        let qTransposeYHat = ((qTranspose * yHat.expandingShape(at: 3)).sum(squeezingAxes: -2) * qMask).sum(squeezingAxes: 1) // [BatchSize, ClassCount]
        expectedLabels = _Raw.tensorScatterUpdate(
          expectedLabels,
          indices: batch.expandingShape(at: -1),
          updates: softmax(h + qYHat + qTransposeYHat))
//      }
    }
  }

  private mutating func performGibbsEStep(using graph: Graph) {
    // Use the provided labels for labeled nodes.
    for batch in Dataset(elements: graph.labeledData).batched(batchSize) {
      expectedLabels = _Raw.tensorScatterUpdate(
        expectedLabels,
        indices: batch.nodeIndices.expandingShape(at: -1),
        updates: Tensor<Float>(oneHotAtIndices: batch.nodeLabels, depth: predictor.classCount))
    }

    let leveledData = graph.leveledData
      .suffix(from: 1)
      .map { Dataset(elements: Tensor<Int32>($0)) }
    let sampleCount = 100
    let burnInSampleCount = 100
    let thinningSampleCount = 10
    var progressBar = ProgressBar(
      count: burnInSampleCount + (sampleCount - 1) * thinningSampleCount,
      configuration: [
        ProgressString(string: "E-Step Gibbs Sampling Progress"),
        ProgressIndex(),
        ProgressBarLine(),
        ProgressTimeEstimates()])
    var samples = [Tensor<Int32>]()
    var currentSample = lastLabelsSample
    var currentSampleCount = 0
    while samples.count < sampleCount {
      for level in leveledData {
        for batch in level.batched(batchSize) {
          let predictions = predictor(batch.scalars)
          let h = predictions.labelProbabilities                                                    // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
          let q = predictions.qualities                                                             // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
          let qTranspose = predictions.qualitiesTranspose                                           // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
          let qMask = predictions.qualitiesMask.expandingShape(at: -1)                              // [BatchSize, MaxNeighborCount, 1]
          let yHat = Tensor<Float>(
            oneHotAtIndices: currentSample.gathering(atIndices: predictions.neighborIndices),
            depth: graph.classCount)                                                                // [BatchSize, MaxNeighborCount, ClassCount]
          let qYHat = ((q * yHat.expandingShape(at: 2)).sum(squeezingAxes: -1) * qMask).sum(squeezingAxes: 1)                   // [BatchSize, ClassCount]
          let qTransposeYHat = ((qTranspose * yHat.expandingShape(at: 3)).sum(squeezingAxes: -2) * qMask).sum(squeezingAxes: 1) // [BatchSize, ClassCount]
          let sample = Tensor<Int32>(
            randomCategorialLogits: h + qYHat + qTransposeYHat,
            sampleCount: 1
          ).squeezingShape(at: -1)
          currentSample = _Raw.tensorScatterUpdate(
            currentSample,
            indices: batch.expandingShape(at: -1),
            updates: sample)
        }
        currentSampleCount += 1
        progressBar.setValue(currentSampleCount)
        if currentSampleCount > burnInSampleCount &&
           currentSampleCount.isMultiple(of: thinningSampleCount) {
          samples.append(currentSample)
        }
      }
    }
    lastLabelsSample = currentSample
    let stackedSamples = Tensor<Float>(// [NodeCount, ClassCount, SampleCount]
      stacking: samples.map { Tensor<Float>(oneHotAtIndices: $0, depth: predictor.classCount) },
      alongAxis: -1)
    expectedLabels = stackedSamples.mean(squeezingAxes: -1)
    // if labelSmoothing > 0.0 {
    //   let term1 = (1 - labelSmoothing) * expectedLabels
    //   let term2 = labelSmoothing / Float(predictor.classCount)
    //   expectedLabels = term1 + term2
    // }
    // expectedLabels = useThresholdedExpectations ?
    //   Tensor<Float>(expectedLabels .== expectedLabels.max(alongAxes: -1)) :
    //   expectedLabels
    //
    // expectedNodePairs = [Int32: Tensor<Float>]()
    // for node in 0..<graph.nodeCount {
    //   let neighborSamples = stackedSamples.gathering(
    //     atIndices: Tensor<Int32>(graph.neighbors[node]))
    //     .expandingShape(at: 2)
    //     .tiled(multiples: Tensor<Int32>([1, 1, Int32(predictor.classCount), 1]))
    //   let nodeSample = stackedSamples[node]
    //     .expandingShape(at: 0, 1)
    //     .tiled(multiples: Tensor<Int32>([1, Int32(predictor.classCount), 1, 1]))
    //   expectedNodePairs[Int32(node)] = (Tensor<Float>(neighborSamples) * Tensor<Float>(nodeSample)).mean(squeezingAxes: -1)
    //   // let nominator = (Tensor<Float>(neighborSamples) * Tensor<Float>(nodeSample)).mean(squeezingAxes: -1)
    //   // let denominator = Tensor<Float>(nominator.sum(alongAxes: -2))
    //   // let broadcastedDenominator = denominator.broadcasted(like: nominator)
    //   // let safeNominator = nominator.replacing(
    //   //   with: Tensor<Float>(onesLike: broadcastedDenominator) / Float(predictor.classCount),
    //   //   where: broadcastedDenominator .== 0)
    //   // let safeDenominator = denominator.replacing(
    //   //   with: Tensor<Float>(onesLike: denominator),
    //   //   where: denominator .== 0)
    //   // expectedNodePairs[Int32(node)] = safeNominator / safeDenominator
    // }
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
          let predictions = predictor.labelProbabilitiesAndQualities(batch.nodeIndices.scalars)
          let crossEntropy = softmaxCrossEntropy(
            logits: predictions.labelProbabilities,
            probabilities: labels)
          let loss = crossEntropy +
            predictions.qualities.sum() * 0.0 +
            predictions.qualitiesTranspose.sum() * 0.0
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
            self.bestExpectedLabels = expectedLabels
            self.bestPredictor = predictor
            self.bestResult = result
          }
        } else {
          self.bestExpectedLabels = expectedLabels
          self.bestPredictor = predictor
          self.bestResult = result
        }
      }
    }
    if evaluationStepCount != nil {
      expectedLabels = bestExpectedLabels
      predictor = bestPredictor
    }
  }

  private mutating func performMStep(data: Dataset<Tensor<Int32>>, emStep: Int) {
    var convergenceStepCount = 0
    bestResult = nil
    resultAccumulator.reset()
    if !useWarmStarting && emStep > 0 {
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
      let yTilde = expectedLabels.gathering(atIndices: batch)
      withLearningPhase(.training) {
        let (negativeLogLikelihood, gradient) = valueWithGradient(at: predictor) {
          [expectedLabels, entropyWeight, qualitiesRegularizationWeight] predictor -> Tensor<Float> in
          let predictions = predictor.labelProbabilitiesAndQualities(batch.scalars)
          let h = predictions.labelProbabilities
          let q = predictions.qualities                                                             // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
          let qTranspose = predictions.qualitiesTranspose                                           // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
          let qMask = predictions.qualitiesMask.expandingShape(at: -1)                              // [BatchSize, MaxNeighborCount, 1]
          let yHat = expectedLabels.gathering(atIndices: predictions.neighborIndices)               // [BatchSize, MaxNeighborCount, 1, ClassCount]
          let qYHat = ((q * yHat.expandingShape(at: 2)).sum(squeezingAxes: -1) * qMask).sum(squeezingAxes: 1)                   // [BatchSize, ClassCount]
          let qTransposeYHat = ((qTranspose * yHat.expandingShape(at: 3)).sum(squeezingAxes: -2) * qMask).sum(squeezingAxes: 1) // [BatchSize, ClassCount]
          let logits = h + qYHat + qTransposeYHat
          let negativeLogLikelihood = -(yTilde * logits).sum()
          let normalizingConstant = logits.logSumExp(alongAxes: -1).sum()
          let hEntropy = entropyWeight * (exp(h) * h).sum()
          let qEntropy = qualitiesRegularizationWeight * (exp(q) * q).sum()
          let loss = hEntropy + qEntropy + negativeLogLikelihood + normalizingConstant
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
        for _ in 0..<10 { modelCopy.performEStep(using: predictor.graph) }
//        let predictionsMAP = labelsApproximateMAP(maxStepCount: 10)
//        let predictionsMAP = labelsGibbsMarginalMAP()
//        let evaluationResult = evaluate(predictions: predictionsMAP, using: predictor.graph)
        let evaluationResult = evaluate(model: modelCopy, using: predictor.graph, usePrior: false)
        let result = resultAccumulator.update(with: evaluationResult)
        if let bestResult = self.bestResult {
          if result.validationAccuracy > bestResult.validationAccuracy {
            logger.info("Best predictor accuracy: \(evaluationResult.validationAccuracy) | \(result.validationAccuracy)")
            self.bestExpectedLabels = expectedLabels
            self.bestPredictor = predictor
            self.bestResult = result
            convergenceStepCount = 0
          } else {
            convergenceStepCount += 1
            if let c = self.mConvergenceEvaluationCount, convergenceStepCount > c {
              expectedLabels = bestExpectedLabels
              predictor = bestPredictor
              return
            }
          }
        } else {
          self.bestExpectedLabels = expectedLabels
          self.bestPredictor = predictor
          self.bestResult = result
        }
      }
    }
    if evaluationStepCount != nil {
      expectedLabels = bestExpectedLabels
      predictor = bestPredictor
    }
  }

  internal func labelsGibbsMarginalMAP() -> [Int32] {
    var modelCopy = self
    modelCopy.performGibbsEStep(using: predictor.graph)
    let probabilities = modelCopy.labelProbabilities(for: [Int32](0..<Int32(predictor.graph.nodeCount)))
    return probabilities.argmax(squeezingAxis: -1).scalars

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
//    return gibbsMarginalMAP(
//      labelLogits: labelLogits,
//      qualityLogits: qualityLogits,
//      graph: predictor.graph)
  }

  internal func labelsApproximateMAP(maxStepCount: Int = 100) -> [Int32] {
    let nodes = [Int](0..<predictor.graph.nodeCount).map(Int32.init)
    var h = [Tensor<Float>]()
    var q = [Tensor<Float>]()
    for batch in Dataset(elements: Tensor<Int32>(nodes)).batched(batchSize) {
      let predictions = predictor.labelProbabilitiesAndQualities(batch.scalars)
      let qualities = predictions.qualities.unstacked(alongAxis: 0)
      let neighborCounts = predictions.qualitiesMask
        .sum(squeezingAxes: -1)
        .unstacked(alongAxis: 0)
        .map { Int($0.scalarized()) }
      h.append(predictions.labelProbabilities)
      q.append(contentsOf: zip(qualities, neighborCounts).map { $0[0..<$1] })
    }
    let hScalars = Tensor<Float>(concatenating: h, alongAxis: 0).scalars
    let labelLogits = LabelLogits(
      logits: hScalars,
      nodeCount: predictor.graph.nodeCount,
      labelCount: predictor.graph.classCount)
    let qualityLogits = q.map {
      QualityLogits(
        logits: $0.scalars,
        nodeCount: $0.shape[0],
        labelCount: predictor.graph.classCount)
    }
    return iteratedConditionalModes(
      labelLogits: labelLogits,
      qualityLogits: qualityLogits,
      graph: predictor.graph,
      maxStepCount: maxStepCount)
  }

  internal func labelsMAP() -> [Int32] {
    let nodes = [Int](0..<predictor.graph.nodeCount).map(Int32.init)
    var h = [Tensor<Float>]()
    var q = [Tensor<Float>]()
    for batch in Dataset(elements: Tensor<Int32>(nodes)).batched(batchSize) {
      let predictions = predictor.labelProbabilitiesAndQualities(batch.scalars)
      let qualities = predictions.qualities.unstacked(alongAxis: 0)
      let neighborCounts = predictions.qualitiesMask
        .sum(squeezingAxes: -1)
        .unstacked(alongAxis: 0)
        .map { Int($0.scalarized()) }
      h.append(predictions.labelProbabilities)
      q.append(contentsOf: zip(qualities, neighborCounts).map { $0[0..<$1] })
    }
    let hScalars = Tensor<Float>(concatenating: h, alongAxis: 0).scalars
    let labelLogits = LabelLogits(
      logits: hScalars,
      nodeCount: predictor.graph.nodeCount,
      labelCount: predictor.graph.classCount)
    let qualityLogits = q.map {
      QualityLogits(
        logits: $0.scalars,
        nodeCount: $0.shape[0],
        labelCount: predictor.graph.classCount)
    }
    return bestNodeAssignments(h: labelLogits, q: qualityLogits, G: predictor.graph)
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
