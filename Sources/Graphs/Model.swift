
import Foundation
import Progress
import TensorFlow

public struct Model<Predictor: GraphPredictor, Optimizer: TensorFlow.Optimizer>
  where Optimizer.Model == Predictor {
  public let optimizerFn: () -> Optimizer
  public let randomSeed: Int64
  public let batchSize: Int
  public let useWarmStarting: Bool
  public let useIncrementalNeighborhoodExpansion: Bool
  public let mStepCount: Int
  public let emStepCount: Int
  public let evaluationStepCount: Int?
  public let mStepLogCount: Int?
  public let mConvergenceEvaluationCount: Int?
  public let emStepCallback: (Model) -> Bool
  public let verbose: Bool

  public var predictor: Predictor
  public var optimizer: Optimizer
  public var yHatYExpected: [Tensor<Float>] // [NodeCount, ClassCount, ClassCount]

  public var bestPredictor: Predictor
  public var bestResult: Result?

  public init(
    predictor: Predictor,
    optimizerFn: @escaping () -> Optimizer,
    randomSeed: Int64,
    batchSize: Int = 128,
    useWarmStarting: Bool = false,
    useIncrementalNeighborhoodExpansion: Bool = false,
    mStepCount: Int = 1000,
    emStepCount: Int = 100,
    evaluationStepCount: Int? = 1,
    mStepLogCount: Int? = 100,
    mConvergenceEvaluationCount: Int? = 10,
    emStepCallback: @escaping (Model) -> Bool = { _ in false },
    verbose: Bool = false
  ) {
    self.predictor = predictor
    self.optimizer = optimizerFn()
    self.optimizerFn = optimizerFn
    self.randomSeed = randomSeed
    self.batchSize = batchSize
    self.useWarmStarting = useWarmStarting
    self.useIncrementalNeighborhoodExpansion = useIncrementalNeighborhoodExpansion
    self.mStepCount = mStepCount
    self.emStepCount = emStepCount
    self.evaluationStepCount = evaluationStepCount
    self.mStepLogCount = mStepLogCount
    self.mConvergenceEvaluationCount = mConvergenceEvaluationCount
    self.emStepCallback = emStepCallback
    self.verbose = verbose
    self.yHatYExpected = [Tensor<Float>]()
    self.bestPredictor = predictor
    self.bestResult = nil

    if verbose { logger.info("Initialization") }
    initialize(using: predictor.graph)
    // performMStep(data: Dataset(elements: predictor.graph.labeledData), emStep: 0)
    let _ = emStepCallback(self)
  }

  public mutating func train() {
    var mStepData = Dataset(elements: predictor.graph.allUnlabeledData)
    for emStep in 0..<emStepCount {
      if useIncrementalNeighborhoodExpansion {
        mStepData = Dataset(elements: predictor.graph.data(atDepth: emStep + 1))
      }

      // M-Step
      if verbose { logger.info("Iteration \(emStep) - Running M-Step") }
      performMStep(data: mStepData, emStep: emStep)

      // E-Step
      if verbose { logger.info("Iteration \(emStep) - Running E-Step") }
      performEStep(using: predictor.graph)

      if emStepCallback(self) { break }
    }
  }

  public func labelLogits(forNodes nodes: [Int32], usePrior: Bool = false) -> Tensor<Float> {
    var batches = [Tensor<Float>]()
    for batch in Dataset(elements: Tensor<Int32>(nodes)).batched(batchSize) {
      // TODO: Posterior.
      batches.append(predictor.labelLogits(batch.scalars))
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
    for node in graph.trainNodes {
      let neighborCount = Int32(graph.neighbors[Int(node)].count)
      yHatYExpected[Int(node)] = Tensor<Float>(
        oneHotAtIndices: Tensor<Int32>(Int(graph.labels[node])), depth: graph.classCount
      ).expandingShape(at: 0, -1).tiled(multiples: Tensor<Int32>([1, 1, neighborCount]))
    }

    let leveledData = graph.leveledData
      .suffix(from: 1)
      .map { Dataset(elements: Tensor<Int32>($0)) }
    // for batch in Dataset(elements: graph.unlabeledData).batched(batchSize) {
    for level in leveledData {
      for batch in level.batched(batchSize) {
        let nodes = batch.scalars
        let yHatYConditionalLogits = predictor.yHatYConditionalLogits(nodes)
        for (node, logits) in zip(nodes, yHatYConditionalLogits) {
          yHatYExpected[Int(node)] = exp(logSoftmax(logits, alongAxis: -2))
        }
      }
    }
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
