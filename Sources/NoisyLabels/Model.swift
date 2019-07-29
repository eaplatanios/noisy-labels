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

import TensorFlow

public protocol EMModel {
  mutating func prepareForEStep()
  mutating func executeEStep(using data: TrainingData, majorityVote: Bool)
  mutating func finalizeEStep(majorityVote: Bool)

  mutating func prepareForMStep()

  /// - Returns: Negative log-likelihood.
  mutating func executeMStep(using data: TrainingData, majorityVote: Bool) -> Float
  mutating func finalizeMStep(majorityVote: Bool)

  /// - Returns: Negative log-likelihood.
  mutating func executeMarginalStep(using data: TrainingData) -> Float

  func negativeLogLikelihood(for data: TrainingData) -> Float

  func labelProbabilities(forInstances instances: Tensor<Int32>) -> [Tensor<Float>]

  func qualities(
    forInstances instances: Tensor<Int32>,
    predictors: Tensor<Int32>,
    labels: Tensor<Int32>
  ) -> [Tensor<Float>]
}

public extension EMModel {
  mutating func prepareForEStep() {}
  mutating func finalizeEStep(majorityVote: Bool) {}
  mutating func prepareForMStep() {}
  mutating func finalizeMStep(majorityVote: Bool) {}
}

public struct MultiLabelEMModel<
  Predictor: MultiLabelPredictor,
  Optimizer: TensorFlow.Optimizer
>: EMModel where Optimizer.Model == Predictor {
  public let instanceCount: Int
  public let predictorCount: Int
  public let labelCount: Int
  public let entropyWeight: Float
  public let useSoftMajorityVote: Bool
  public let useSoftPredictions: Bool

  public private(set) var predictor: Predictor
  public private(set) var optimizer: Optimizer

  private var eStepAccumulator: Tensor<Float>

  public init(
    predictor: Predictor,
    optimizer: Optimizer,
    entropyWeight: Float = 0.0,
    useSoftMajorityVote: Bool = true,
    useSoftPredictions: Bool = true
  ) {
    self.predictor = predictor
    self.optimizer = optimizer
    self.instanceCount = predictor.instanceCount
    self.predictorCount = predictor.predictorCount
    self.labelCount = predictor.labelCount
    self.entropyWeight = entropyWeight
    self.useSoftMajorityVote = useSoftMajorityVote
    self.useSoftPredictions = useSoftPredictions
    self.eStepAccumulator = Tensor<Float>(zeros: [instanceCount, labelCount, 2])
  }

  public mutating func prepareForEStep() {
    eStepAccumulator = Tensor<Float>(zeros: [instanceCount, labelCount, 2])
  }

  public mutating func executeEStep(using data: TrainingData, majorityVote: Bool) {
    // yHatProvided shape: [BatchSize, 2]
    // yHat shape: [BatchSize, 2]
    // qLog shape: [BatchSize, 2, 2]
    // qLogYHat shape: [BatchSize, 2]
    let yHatProvided = Tensor<Float>(stacking: [1.0 - data.values, data.values], alongAxis: -1)
    let yHat = useSoftMajorityVote ? yHatProvided : Tensor<Float>(yHatProvided .>= 0.5)
    let qLog = predictor.qualities(
      forInstances: data.instances,
      predictors: data.predictors,
      labels: data.labels)
    let qLogYHat = (qLog * yHat.expandingShape(at: 1)).sum(squeezingAxes: -1)
    eStepAccumulator = Raw.tensorScatterAdd(
      eStepAccumulator,
      indices: Tensor<Int32>(
        stacking: [data.instances, data.labels],
        alongAxis: -1),
      updates: majorityVote ? yHat : qLogYHat)
  }

  public mutating func finalizeEStep(majorityVote: Bool) {
    if majorityVote {
      eStepAccumulator = log(eStepAccumulator)
    }
  }

  public mutating func prepareForMStep() {
    predictor.reset()
  }

  public mutating func executeMStep(using data: TrainingData, majorityVote: Bool) -> Float {
    // majorityVote shape: []
    // yHatProvided shape: [BatchSize, 2]
    // yHat shape: [BatchSize, 2]
    // yAccumulated shape: [BatchSize, 2]
    let majorityVote = Tensor<Float>(majorityVote ? 0 : 1)
    let yHatProvided = Tensor<Float>(stacking: [1.0 - data.values, data.values], alongAxis: -1)
    let yHat = useSoftMajorityVote ? yHatProvided : Tensor<Float>(yHatProvided .>= 0.5)
    let yAccumulated = Raw.gatherNd(
      params: eStepAccumulator,
      indices: Tensor<Int32>(stacking: [data.instances, data.labels], alongAxis: -1))

    let (negativeLogLikelihood, gradient) = predictor.valueWithGradient { predictor -> Tensor<Float> in
      // hLog shape: [BatchSize, 2]
      // yAccumulatedHLog shape: [BatchSize, 2]
      // yExpected shape: [BatchSize, 2]
      // qLog shape: [BatchSize, 2, 2]
      // qLogYHat shape: [BatchSize, 2]
      let predictions = predictor.predictions(
        forInstances: data.instances,
        predictors: data.predictors,
        labels: data.labels)
      let hLog1 = predictions.labelProbabilities // min(predictions.labelProbabilities, 1e-6)
        .batchGathering(atIndices: data.labels.expandingShape(at: -1))
      let hLog = log1mexp(hLog1).concatenated(with: hLog1, alongAxis: -1)
      let yAccumulatedHLog = yAccumulated + hLog * majorityVote
      let yExpected = exp(yAccumulatedHLog - yAccumulatedHLog.logSumExp(alongAxes: -1))
      // TODO: Do we need the above? Or should we maybe replace it with the below?
      // let yAccumulatedHLog = predictions.includePredictionsPrior ?
      //   yAccumulated + hLog :
      //   yAccumulated + log(0.5)
      let qLog = predictions.qualities
      let qLogYHat = (qLog * yHat.expandingShape(at: 1)).sum(squeezingAxes: -1)
      let term1 = -(yExpected * hLog).sum()
      let term2 = -(yExpected * qLogYHat).sum()
      let term3 = predictions.regularizationTerm
      let term4 = self.entropyWeight * (exp(hLog) * hLog).sum()
      return term1 + term2 + term3 + term4
    }

    optimizer.update(&predictor, along: gradient)
    return negativeLogLikelihood.scalarized()
  }

  public mutating func executeMarginalStep(using data: TrainingData) -> Float {
    // yHatProvided shape: [BatchSize, 2]
    // yHat shape: [BatchSize, 2]
    let yHatProvided = Tensor<Float>(stacking: [1.0 - data.values, data.values], alongAxis: -1)
    let yHat = useSoftMajorityVote ? yHatProvided : Tensor<Float>(yHatProvided .>= 0.5)
    
    let (negativeLogLikelihood, gradient) = predictor.valueWithGradient { predictor -> Tensor<Float> in
      // hLog shape: [BatchSize, 2]
      // qLogYHat shape: [BatchSize, 2]
      let predictions = predictor.predictions(
        forInstances: data.instances,
        predictors: data.predictors,
        labels: data.labels)
      let hLog1 = predictions.labelProbabilities // min(predictions.labelProbabilities, 1e-6)
        .batchGathering(atIndices: data.labels.expandingShape(at: -1))
      let hLog = log1mexp(hLog1).concatenated(with: hLog1, alongAxis: -1)
      let qLogYHat = predictions.qualities * yHat.expandingShape(at: 1)      
      let term1 = -(qLogYHat.sum(squeezingAxes: -1) + hLog).logSumExp(squeezingAxes: -1).sum()
      let term2 = predictions.regularizationTerm
      let term3 = self.entropyWeight * (exp(hLog) * hLog).sum()
      return term1 + term2 + term3
    }

    optimizer.update(&predictor, along: gradient)
    return negativeLogLikelihood.scalarized()
  }

  public func negativeLogLikelihood(for data: TrainingData) -> Float {
    // yHatProvided shape: [BatchSize, 2]
    // yHat shape: [BatchSize, 2]
    // hLog shape: [BatchSize, 2]
    // qLogYHat shape: [BatchSize, 2]
    let predictions = predictor.predictions(
      forInstances: data.instances,
      predictors: data.predictors,
      labels: data.labels)
    let yHatProvided = Tensor<Float>(stacking: [1.0 - data.values, data.values], alongAxis: -1)
    let yHat = useSoftMajorityVote ? yHatProvided : Tensor<Float>(yHatProvided .>= 0.5)
    let hLog1 = predictions.labelProbabilities // min(predictions.labelProbabilities, 1e-6)
      .batchGathering(atIndices: data.labels.expandingShape(at: -1))
    let hLog = log1mexp(hLog1).concatenated(with: hLog1, alongAxis: -1)
    let qLogYHat = predictions.qualities * yHat.expandingShape(at: 1)
    let term1 = -(qLogYHat.sum(squeezingAxes: -1) + hLog).logSumExp(squeezingAxes: -1).sum()
    let term2 = predictions.regularizationTerm
    let term3 = entropyWeight * (exp(hLog) * hLog).sum()
    return (term1 + term2 + term3).scalarized()
  }

  public func labelProbabilities(forInstances instances: Tensor<Int32>) -> [Tensor<Float>] {
    return [exp(predictor.labelProbabilities(forInstances: instances)).squeezingShape(at: -1)]
  }

  public func qualities(
    forInstances instances: Tensor<Int32>,
    predictors: Tensor<Int32>,
    labels: Tensor<Int32>
  ) -> [Tensor<Float>] {
    let predictions = predictor.predictions(
      forInstances: instances,
      predictors: predictors,
      labels: labels)
    let hLog1 = predictions.labelProbabilities // min(predictions.labelProbabilities, 1e-6)
      .batchGathering(atIndices: labels.expandingShape(at: -1))
    let hLog = log1mexp(hLog1).concatenated(with: hLog1, alongAxis: -1)
    let qLogHLog = predictions.qualities + hLog.expandingShape(at: -1)
    let qualities = Tensor<Float>(stacking: [
      qLogHLog[0..., 1, 1],
      qLogHLog[0..., 0, 0]
    ], alongAxis: -1).logSumExp(squeezingAxes: -1)
    return [qualities]
  }
}

public struct MultiClassMultiLabelEMModel<
  Predictor: MultiClassMultiLabelPredictor,
  Optimizer: TensorFlow.Optimizer
>: EMModel where Optimizer.Model == Predictor {
  public let instanceCount: Int
  public let predictorCount: Int
  public let labelCount: Int
  public let classCounts: [Int]
  public let entropyWeight: Float
  public let useSoftMajorityVote: Bool
  public let useSoftPredictions: Bool

  public private(set) var predictor: Predictor
  public private(set) var optimizer: Optimizer

  private var eStepAccumulators: [Tensor<Float>]

  public init(
    predictor: Predictor,
    optimizer: Optimizer,
    entropyWeight: Float = 0.0,
    useSoftMajorityVote: Bool = true,
    useSoftPredictions: Bool = true
  ) {
    self.predictor = predictor
    self.optimizer = optimizer
    self.instanceCount = predictor.instanceCount
    self.predictorCount = predictor.predictorCount
    self.labelCount = predictor.labelCount
    self.classCounts = predictor.classCounts
    self.entropyWeight = entropyWeight
    self.useSoftMajorityVote = useSoftMajorityVote
    self.useSoftPredictions = useSoftPredictions
    self.eStepAccumulators = predictor.classCounts.map {
      Tensor<Float>(zeros: [predictor.instanceCount, $0])
    }
  }

  public mutating func prepareForEStep() {
    eStepAccumulators = classCounts.map {
      Tensor<Float>(zeros: [instanceCount, $0])
    }
  }

  public mutating func executeEStep(using data: TrainingData, majorityVote: Bool) {
    // Array of one-hot encodings of the label that corresponds to each batch element.
    let labelsOneHot = Tensor<Int32>(
      oneHotAtIndices: data.labels,
      depth: labelCount
    ).unstacked(alongAxis: 1)

    // Array of tensors (one for each label), where:
    // qLogs[l].shape == [batchSize, classCounts[l], classCounts[l]].
    let qLogs = predictor.qualities(
      forInstances: data.instances,
      predictors: data.predictors,
      labels: data.labels)

    // Update the E-step accumulator for each label.
    for l in 0..<labelCount {
      let labelMask = labelsOneHot[l] .> 0
      let classCount = classCounts[l]
      let instances = data.instances.gathering(where: labelMask)
      let values = data.values.gathering(where: labelMask)
      let qLog = qLogs[l].gathering(where: labelMask)
      
      // yHat.shape == [batchSize for l, classCounts[l]].
      let yHat = useSoftMajorityVote ?
        Tensor<Float>(stacking: [1.0 - values, values], alongAxis: -1) :
        Tensor<Float>(oneHotAtIndices: Tensor<Int32>(data.values), depth: classCount)

      // qLogYHat.shape == [batchSize for l, classCounts[l]].
      let qLogYHat = (qLog * yHat.expandingShape(at: 1)).sum(squeezingAxes: -1)

      eStepAccumulators[l] = Raw.tensorScatterAdd(
        eStepAccumulators[l],
        indices: instances.expandingShape(at: -1),
        updates: majorityVote ? yHat : qLogYHat)
    }
  }

  public mutating func finalizeEStep(majorityVote: Bool) {
    if majorityVote {
      eStepAccumulators = eStepAccumulators.map(log)
    }
  }

  public mutating func prepareForMStep() {
    predictor.reset()
  }

  public mutating func executeMStep(using data: TrainingData, majorityVote: Bool) -> Float {
    let majorityVote = Tensor<Float>(majorityVote ? 0 : 1)

    // Array of one-hot encodings of the label that corresponds to each batch element.
    let labelMasks = Tensor<Int32>(
      oneHotAtIndices: data.labels,
      depth: labelCount
    ).unstacked(alongAxis: 1)

    let (negativeLogLikelihood, gradient) = predictor.valueWithGradient { [
      eStepAccumulators,
      useSoftMajorityVote,
      entropyWeight
    ] predictor -> Tensor<Float> in
      let predictions = predictor.predictions(
        forInstances: data.instances,
        predictors: data.predictors,
        labels: data.labels)
      return modelZip(
        labelMasks: labelMasks,
        eStepAccumulators: eStepAccumulators,
        labelProbabilities: predictions.labelProbabilities,
        qualities: predictions.qualities
      ).differentiableMap { parameters -> Tensor<Float> in
        let labelMask = parameters.labelMask .> 0
        let instances = data.instances.gathering(where: labelMask)
        let values = data.values.gathering(where: labelMask)
        let qLog = parameters.qualities.gathering(where: labelMask)

        // hLog.shape == [batchSize for l, classCounts[l]].
        let hLog = parameters.labelProbabilities.gathering(where: labelMask)

        // yAccumulated.shape == [batchSize for l, classCounts[l]].
        let yAccumulated = parameters.eStepAccumulator.gathering(atIndices: instances)
        let yAccumulatedHLog = yAccumulated + hLog * majorityVote
        let yExpected = withoutDerivative(at: yAccumulatedHLog) {
          exp($0 - $0.logSumExp(alongAxes: -1))
        }
        // TODO: Do we need the above? Or should we maybe replace it with the below?
        // let yAccumulatedHLog = predictions.includePredictionsPrior ?
        //   yAccumulated + hLog :
        //   yAccumulated + log(0.5)

        // yHat.shape == [batchSize for l, classCounts[l]].
        let yHat = useSoftMajorityVote ?
          Tensor<Float>(stacking: [1.0 - values, values], alongAxis: -1) :
          Tensor<Float>(oneHotAtIndices: Tensor<Int32>(data.values), depth: hLog.shape[1])

        // qLogYHat.shape == [batchSize for l, classCounts[l]].
        let qLogYHat = (qLog * yHat.expandingShape(at: 1)).sum(squeezingAxes: -1)

        return entropyWeight * (exp(hLog) * hLog).sum() - 
          (yExpected * hLog).sum() -
          (yExpected * qLogYHat).sum()
      }.differentiableReduce(predictions.regularizationTerm, { $0 + $1 })
    }

    optimizer.update(&predictor, along: gradient)
    return negativeLogLikelihood.scalarized()
  }

  public mutating func executeMarginalStep(using data: TrainingData) -> Float {
    // Array of one-hot encodings of the label that corresponds to each batch element.
    let labelMasks = Tensor<Int32>(
      oneHotAtIndices: data.labels,
      depth: labelCount
    ).unstacked(alongAxis: 1)

    let (negativeLogLikelihood, gradient) = predictor.valueWithGradient { [
      eStepAccumulators,
      useSoftMajorityVote,
      entropyWeight
    ] predictor -> Tensor<Float> in
      let predictions = predictor.predictions(
        forInstances: data.instances,
        predictors: data.predictors,
        labels: data.labels)
      return modelZip(
        labelMasks: labelMasks,
        eStepAccumulators: eStepAccumulators,
        labelProbabilities: predictions.labelProbabilities,
        qualities: predictions.qualities
      ).differentiableMap { parameters -> Tensor<Float> in
        let labelMask = parameters.labelMask .> 0
        let values = data.values.gathering(where: labelMask)
        let qLog = parameters.qualities.gathering(where: labelMask)

        // hLog.shape == [batchSize for l, classCounts[l]].
        let hLog = parameters.labelProbabilities.gathering(where: labelMask)

        // yHat.shape == [batchSize for l, classCounts[l]].
        let yHat = useSoftMajorityVote ?
          Tensor<Float>(stacking: [1.0 - values, values], alongAxis: -1) :
          Tensor<Float>(oneHotAtIndices: Tensor<Int32>(data.values), depth: hLog.shape[1])

        // qLogYHat.shape == [batchSize for l, classCounts[l]].
        let qLogYHat = (qLog * yHat.expandingShape(at: 1)).sum(squeezingAxes: -1)

        return entropyWeight * (exp(hLog) * hLog).sum() - 
          (qLogYHat.sum(squeezingAxes: -1) + hLog).logSumExp(squeezingAxes: -1).sum()
      }.differentiableReduce(Tensor<Float>(zeros: []), { $0 + $1 })
    }

    optimizer.update(&predictor, along: gradient)
    return negativeLogLikelihood.scalarized()
  }

  public func negativeLogLikelihood(for data: TrainingData) -> Float {
    // Array of one-hot encodings of the label that corresponds to each batch element.
    let labelMasks = Tensor<Int32>(
      oneHotAtIndices: data.labels,
      depth: labelCount
    ).unstacked(alongAxis: 1)

    let predictions = predictor.predictions(
      forInstances: data.instances,
      predictors: data.predictors,
      labels: data.labels)
    let qLogs = predictions.qualities

    // Sum the loss terms contributed by each label.
    var loss = predictions.regularizationTerm
    for l in 0..<labelCount {
      let labelMask = labelMasks[l] .> 0
      let values = data.values.gathering(where: labelMask)
      let qLog = qLogs[l].gathering(where: labelMask)

      // hLog.shape == [batchSize, classCounts[l]].
      let hLog = predictions.labelProbabilities[l].gathering(where: labelMask)

      // yHat.shape == [batchSize, classCounts[l]].
      let yHat = self.useSoftMajorityVote ?
        Tensor<Float>(stacking: [1.0 - values, values], alongAxis: -1) :
        Tensor<Float>(oneHotAtIndices: Tensor<Int32>(data.values), depth: self.classCounts[l])

      // qLogYHat.shape == [batchSize, classCounts[l]].
      let qLogYHat = (qLog * yHat.expandingShape(at: 1)).sum(squeezingAxes: -1)

      loss = loss - (qLogYHat.sum(squeezingAxes: -1) + hLog).logSumExp(squeezingAxes: -1).sum()
      loss = loss + self.entropyWeight * (exp(hLog) * hLog).sum()
    }
    return loss.scalarized()
  }

  public func labelProbabilities(forInstances instances: Tensor<Int32>) -> [Tensor<Float>] {
    predictor.labelProbabilities(forInstances: instances).map(exp)
  }

  public func qualities(
    forInstances instances: Tensor<Int32>,
    predictors: Tensor<Int32>,
    labels: Tensor<Int32>
  ) -> [Tensor<Float>] {
    // Array of one-hot encodings of the label that corresponds to each batch element.
    let labelMasks = Tensor<Int32>(
      oneHotAtIndices: labels,
      depth: labelCount
    ).unstacked(alongAxis: 1)

    let predictions = predictor.predictions(
      forInstances: instances,
      predictors: predictors,
      labels: labels)
    let qLogs = predictions.qualities

    // Compute the qualities for each label separately.
    return (0..<labelCount).map { l -> Tensor<Float> in
      let labelMask = labelMasks[l] .> 0
      let hLog = predictions.labelProbabilities[l].gathering(where: labelMask)
      let qLogHLog = qLogs[l].gathering(where: labelMask) * hLog.expandingShape(at: -1)
      return Raw.matrixDiagPart(qLogHLog).logSumExp(squeezingAxes: -1)
    }
  }
}
