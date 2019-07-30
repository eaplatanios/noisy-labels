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
    let labelMasks = self.labelMasks(for: data.labels)
    let qLogs = predictor.qualities(data.instances, data.predictors, data.labels)
    for l in 0..<labelCount {
      let qLog = qLogs[l].gathering(where: labelMasks[l])
      let values = data.values.gathering(where: labelMasks[l])
      let yHat = useSoftMajorityVote ?
        Tensor<Float>(stacking: [1.0 - values, values], alongAxis: -1) :
        Tensor<Float>(oneHotAtIndices: Tensor<Int32>(values), depth: classCounts[l])
      let qLogYHat = (qLog * yHat.expandingShape(at: 1)).sum(squeezingAxes: -1)
      eStepAccumulators[l] = Raw.tensorScatterAdd(
        eStepAccumulators[l],
        indices: data.instances.gathering(where: labelMasks[l]).expandingShape(at: -1),
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
    let labelMasks = self.labelMasks(for: data.labels)
    let (negativeLogLikelihood, gradient) = predictor.valueWithGradient { [
      eStepAccumulators, useSoftMajorityVote, entropyWeight
    ] predictor -> Tensor<Float> in
      let predictions = predictor(data.instances, data.predictors, data.labels)
      let includePredictionsPrior = withoutDerivative(at: predictions.includePredictionsPrior)
      return modelZip(
        labelMasks: labelMasks,
        eStepAccumulators: eStepAccumulators,
        labelProbabilities: predictions.labelProbabilities,
        qualities: predictions.qualities
      ).differentiableMap { parameters -> Tensor<Float> in
        let hLog = parameters.labelProbabilities.gathering(where: parameters.labelMask)
        let qLog = parameters.qualities.gathering(where: parameters.labelMask)
        let values = data.values.gathering(where: parameters.labelMask)
        let yHat = useSoftMajorityVote ?
          Tensor<Float>(stacking: [1.0 - values, values], alongAxis: -1) :
          Tensor<Float>(oneHotAtIndices: Tensor<Int32>(values), depth: hLog.shape[1])
        let qLogYHat = (qLog * yHat.expandingShape(at: 1)).sum(squeezingAxes: -1)
        let yAccumulated = parameters.eStepAccumulator.gathering(
          atIndices: data.instances.gathering(where: parameters.labelMask))
        let yAccumulatedHLog = includePredictionsPrior ?
          yAccumulated + withoutDerivative(at: hLog) * majorityVote :
          yAccumulated + log(0.5) * majorityVote
        let yExpected = exp(yAccumulatedHLog - yAccumulatedHLog.logSumExp(alongAxes: -1))
        return entropyWeight * (exp(hLog) * hLog).sum() - 
          (yExpected * hLog).sum() -
          (yExpected * qLogYHat).sum()
      }.differentiableReduce(predictions.regularizationTerm, { $0 + $1 })
    }
    optimizer.update(&predictor, along: gradient)
    // TODO: !!!! More formal support for learning rate decay.
    optimizer.learningRate *= 0.995
    return negativeLogLikelihood.scalarized()
  }

  public mutating func executeMarginalStep(using data: TrainingData) -> Float {
    let labelMasks = self.labelMasks(for: data.labels)
    let (negativeLogLikelihood, gradient) = predictor.valueWithGradient { [
      eStepAccumulators, useSoftMajorityVote, entropyWeight
    ] predictor -> Tensor<Float> in
      let predictions = predictor(data.instances, data.predictors, data.labels)
      return modelZip(
        labelMasks: labelMasks,
        eStepAccumulators: eStepAccumulators,
        labelProbabilities: predictions.labelProbabilities,
        qualities: predictions.qualities
      ).differentiableMap { parameters -> Tensor<Float> in
        let hLog = parameters.labelProbabilities.gathering(where: parameters.labelMask)
        let qLog = parameters.qualities.gathering(where: parameters.labelMask)
        let values = data.values.gathering(where: parameters.labelMask)
        let yHat = useSoftMajorityVote ?
          Tensor<Float>(stacking: [1.0 - values, values], alongAxis: -1) :
          Tensor<Float>(oneHotAtIndices: Tensor<Int32>(values), depth: hLog.shape[1])
        let qLogYHat = (qLog * yHat.expandingShape(at: 1)).sum(squeezingAxes: -1)
        let logLikelihood = (qLogYHat + hLog).logSumExp(squeezingAxes: -1).sum()
        let entropy = (exp(hLog) * hLog).sum()
        return entropyWeight * entropy - logLikelihood
    // TODO: !!!! More formal support for learning rate decay.
      }.differentiableReduce(Tensor(0.0), { $0 + $1 })
    }
    optimizer.update(&predictor, along: gradient)
    optimizer.learningRate *= 0.995
    return negativeLogLikelihood.scalarized()
  }

  public func negativeLogLikelihood(for data: TrainingData) -> Float {
    let labelMasks = self.labelMasks(for: data.labels)
    let predictions = predictor(data.instances, data.predictors, data.labels)
    var negativeLogLikelihood = predictions.regularizationTerm
    for l in 0..<labelCount {
      let hLog = predictions.labelProbabilities[l].gathering(where: labelMasks[l])
      let qLog = predictions.qualities[l].gathering(where: labelMasks[l])
      let values = data.values.gathering(where: labelMasks[l])
      let yHat = useSoftMajorityVote ?
        Tensor<Float>(stacking: [1.0 - values, values], alongAxis: -1) :
        Tensor<Float>(oneHotAtIndices: Tensor<Int32>(values), depth: classCounts[l])
      let qLogYHat = (qLog * yHat.expandingShape(at: 1)).sum(squeezingAxes: -2, -1)
      let logLikelihood = (qLogYHat + hLog).logSumExp(squeezingAxes: -1).sum()
      let entropy = (exp(hLog) * hLog).sum()
      negativeLogLikelihood += entropyWeight * entropy - logLikelihood
    }
    return negativeLogLikelihood.scalarized()
  }

  public func labelProbabilities(_ instances: Tensor<Int32>) -> [Tensor<Float>] {
    predictor.labelProbabilities(instances).map(exp)
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
      let qLog = predictions.qualities[l].gathering(where: labelMasks[l])
      let qLogHLog = qLog * hLog.expandingShape(at: -1)
      return Raw.matrixDiagPart(qLogHLog).logSumExp(squeezingAxes: -1)
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
