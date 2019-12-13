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

// TODO: !!!! Remove `Optimizer.Scalar == Float` once formal support for learning rate decay lands.
public struct EMModel<Predictor: NoisyLabels.Predictor, Optimizer: TensorFlow.Optimizer>
where Optimizer.Model == Predictor, Optimizer.Scalar == Float {
  public let instanceCount: Int
  public let predictorCount: Int
  public let labelCount: Int
  public let classCounts: [Int]
  public let entropyWeight: Float
  public let useSoftPredictions: Bool
  public let initialLearningRate: Float
  public let learningRateDecayFactor: Float

  public private(set) var predictor: Predictor
  public private(set) var optimizer: Optimizer

  public private(set) var eStepAccumulators: [Tensor<Float>]
  public private(set) var expectedLabels: [Tensor<Float>] = []

  public init(
    predictor: Predictor,
    optimizer: Optimizer,
    entropyWeight: Float = 0.0,
    useSoftPredictions: Bool = true,
    learningRateDecayFactor: Float = 1.0
  ) {
    self.predictor = predictor
    self.optimizer = optimizer
    self.instanceCount = predictor.instanceCount
    self.predictorCount = predictor.predictorCount
    self.labelCount = predictor.labelCount
    self.classCounts = predictor.classCounts
    self.entropyWeight = entropyWeight
    self.useSoftPredictions = useSoftPredictions
    self.initialLearningRate = optimizer.learningRate
    self.learningRateDecayFactor = learningRateDecayFactor
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
    let labelMasks = self.labelMasks(for: data.labels)
    let qLogs = predictor.qualities(data.instances, data.predictors, data.labels)
    for l in 0..<labelCount {
      let qLog = logSoftmax(qLogs[l].gathering(where: labelMasks[l]), alongAxis: 1)
      let values = data.values.gathering(where: labelMasks[l])
      let yHat = useSoftPredictions && classCounts[l] == 2 ?
        Tensor<Float>(stacking: [1.0 - values, values], alongAxis: -1) :
        Tensor<Float>(oneHotAtIndices: Tensor<Int32>(values), depth: classCounts[l])
      let qLogYHat = (qLog * yHat.expandingShape(at: 1)).sum(squeezingAxes: -1)
      eStepAccumulators[l] = _Raw.tensorScatterAdd(
        eStepAccumulators[l],
        indices: data.instances.gathering(where: labelMasks[l]).expandingShape(at: -1),
        updates: majorityVote ? yHat : qLogYHat)
    }
  }

  public mutating func finalizeEStep(majorityVote: Bool) {
    if majorityVote { eStepAccumulators = eStepAccumulators.map(log) }
    expectedLabels = eStepAccumulators.map { exp($0 - $0.logSumExp(alongAxes: -1)) }
  }

  public mutating func prepareForMStep() {
    predictor.reset()
    optimizer.learningRate = initialLearningRate
  }

  public mutating func executeMStep(using data: TrainingData) -> Float {
    let labelMasks = self.labelMasks(for: data.labels)
    let (negativeLogLikelihood, gradient) = predictor.valueWithGradient { [
      expectedLabels, useSoftPredictions, entropyWeight
    ] predictor -> Tensor<Float> in
      let predictions = predictor(data.instances, data.predictors, data.labels)
      return modelZip(
        labelMasks: labelMasks,
        expectedLabels: expectedLabels,
        labelProbabilities: predictions.labelProbabilities,
        qualities: predictions.qualities
      ).differentiableMap { parameters -> Tensor<Float> in
        let hLog = parameters.labelProbabilities.gathering(where: parameters.labelMask)
        let qLog = logSoftmax(
          parameters.qualities.gathering(where: parameters.labelMask),
          alongAxis: 1)
        let values = data.values.gathering(where: parameters.labelMask)
        let classCount = hLog.shape[1]
        let yHat = { useSoftPredictions && classCount == 2 ?
          Tensor<Float>(stacking: [1.0 - values, values], alongAxis: -1) :
          Tensor<Float>(oneHotAtIndices: Tensor<Int32>(values), depth: classCount) }()
        let qLogYHat = (qLog * yHat.expandingShape(at: 1)).sum(squeezingAxes: -1)
        let yExpected = parameters.expectedLabels.gathering(
          atIndices: data.instances.gathering(where: parameters.labelMask))
        return entropyWeight * (exp(hLog) * hLog).sum() -
          (yExpected * hLog).sum() -
          (yExpected * qLogYHat).sum()
      }.differentiableReduce(predictions.regularizationTerm, { $0 + $1 })
    }
    optimizer.update(&predictor, along: gradient)
    optimizer.learningRate *= learningRateDecayFactor
    return negativeLogLikelihood.scalarized()
  }

  public mutating func executeMarginalStep(using data: TrainingData) -> Float {
    let labelMasks = self.labelMasks(for: data.labels)
    let (negativeLogLikelihood, gradient) = predictor.valueWithGradient { [
      expectedLabels, useSoftPredictions, entropyWeight
    ] predictor -> Tensor<Float> in
      let predictions = predictor(data.instances, data.predictors, data.labels)
      return modelZip(
        labelMasks: labelMasks,
        expectedLabels: expectedLabels, // TODO: Remove this from here.
        labelProbabilities: predictions.labelProbabilities,
        qualities: predictions.qualities
      ).differentiableMap { parameters -> Tensor<Float> in
        let hLog = parameters.labelProbabilities.gathering(where: parameters.labelMask)
        let qLog = logSoftmax(
          parameters.qualities.gathering(where: parameters.labelMask),
          alongAxis: 1)
        let values = data.values.gathering(where: parameters.labelMask)
        let classCount = hLog.shape[1]
        let yHat = { useSoftPredictions && classCount == 2 ?
          Tensor<Float>(stacking: [1.0 - values, values], alongAxis: -1) :
          Tensor<Float>(oneHotAtIndices: Tensor<Int32>(values), depth: classCount) }()
        let qLogYHat = (qLog * yHat.expandingShape(at: 1)).sum(squeezingAxes: -1)
        let logLikelihood = (qLogYHat + hLog).logSumExp(squeezingAxes: -1).sum()
        let entropy = (exp(hLog) * hLog).sum()
        return entropyWeight * entropy - logLikelihood
      }.differentiableReduce(Tensor(0.0), { $0 + $1 })
    }
    optimizer.update(&predictor, along: gradient)
    optimizer.learningRate *= learningRateDecayFactor
    return negativeLogLikelihood.scalarized()
  }

  public func negativeLogLikelihood(for data: TrainingData) -> Float {
    let labelMasks = self.labelMasks(for: data.labels)
    let predictions = predictor(data.instances, data.predictors, data.labels)
    var negativeLogLikelihood = predictions.regularizationTerm
    for l in 0..<labelCount {
      let hLog = predictions.labelProbabilities[l].gathering(where: labelMasks[l])
      let qLog = logSoftmax(predictions.qualities[l].gathering(where: labelMasks[l]), alongAxis: 1)
      let values = data.values.gathering(where: labelMasks[l])
      let yHat = useSoftPredictions && classCounts[l] == 2 ?
        Tensor<Float>(stacking: [1.0 - values, values], alongAxis: -1) :
        Tensor<Float>(oneHotAtIndices: Tensor<Int32>(values), depth: classCounts[l])
      let qLogYHat = (qLog * yHat.expandingShape(at: 1)).sum(squeezingAxes: -1)
      let logLikelihood = (qLogYHat + hLog).logSumExp(squeezingAxes: -1).sum()
      let entropy = (exp(hLog) * hLog).sum()
      negativeLogLikelihood += entropyWeight * entropy - logLikelihood
    }
    return negativeLogLikelihood.scalarized()
  }

  public func labelProbabilities(_ instances: Tensor<Int32>) -> [Tensor<Float>] {
    expectedLabels.map { $0.gathering(atIndices: instances) }
//    predictor.labelProbabilities(instances).map(exp)
  }

  public func qualities(
    _ instances: Tensor<Int32>,
    _ predictors: Tensor<Int32>,
    _ labels: Tensor<Int32>
  ) -> [Tensor<Float>] {
    let labelMasks = self.labelMasks(for: labels)
    let predictions = predictor(instances, predictors, labels)
    return (0..<labelCount).map { l -> Tensor<Float> in
      let hLog = predictions.labelProbabilities[l].gathering(where: labelMasks[l])
      let qLog = logSoftmax(predictions.qualities[l].gathering(where: labelMasks[l]), alongAxis: 1)
      let qLogHLog = _Raw.matrixDiagPart(qLog) + hLog
      return qLogHLog.logSumExp(squeezingAxes: -1)
    }
  }
}

extension EMModel {
  /// Returns an array of one-hot encodings of the label that corresponds to each batch element.
  @inlinable
  internal func labelMasks(for labels: Tensor<Int32>) -> [Tensor<Bool>] {
    (Tensor<Int32>(oneHotAtIndices: labels, depth: labelCount) .> 0).unstacked(alongAxis: 1)
  }
}
