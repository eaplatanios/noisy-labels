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
import Python
import TensorFlow

// TODO: The Python interface is not thread-safe.
let serialQueue = DispatchQueue(label: "PythonQueue")

let metrics = Python.import("sklearn.metrics")
let np = Python.import("numpy")

public func computeMADErrorRank(
  estimatedQualities: Tensor<Float>,
  trueQualities: Tensor<Float>
) -> Float {
  var result: Float? = nil
  serialQueue.sync {
    let estimatedQualities = estimatedQualities.makeNumpyArray()
    let trueQualities = trueQualities.makeNumpyArray()
    let p = np.argsort(estimatedQualities, axis: -1)
    let t = np.argsort(trueQualities, axis: -1)
    result = Float(np.mean(np.abs(p - t)))!
  }
  return result!
}

public func computeMADError(
  estimatedQualities: Tensor<Float>,
  trueQualities: Tensor<Float>
) -> Float {
  return abs(estimatedQualities - trueQualities).mean().scalarized()
}

public func computeAccuracy(
  estimatedLabelProbabilities: Tensor<Float>,
  trueLabels: Tensor<Int32>
) -> Float {
  let estimatedLabels = estimatedLabelProbabilities.rank > 1 ?
    Tensor<Int32>(estimatedLabelProbabilities.argmax(squeezingAxis: -1)) :
    Tensor<Int32>(estimatedLabelProbabilities .>= 0.5)
  return Tensor<Float>(estimatedLabels .== trueLabels).mean().scalarized()
}

public func computeAUC(
  estimatedLabelProbabilities: Tensor<Float>,
  trueLabels: Tensor<Int32>
) -> Float {
  var result: Float? = nil
  serialQueue.sync {
    var trueLabels = trueLabels.makeNumpyArray()
    if estimatedLabelProbabilities.rank > 1 {
      let trueLabelsOneHot = np.zeros(estimatedLabelProbabilities.shape)
      trueLabelsOneHot[np.arange(trueLabels.shape[0]), trueLabels] = 1
      trueLabels = trueLabelsOneHot
    }
    result = Float(metrics.average_precision_score(
      y_true: trueLabels,
      y_score: estimatedLabelProbabilities.makeNumpyArray()))!
  }
  return result!
}

public struct EvaluationResult {
  public let madErrorRank: Float
  public let madError: Float
  public let accuracy: Float
  public let auc: Float

  public init(madErrorRank: Float, madError: Float, accuracy: Float, auc: Float) {
    self.madErrorRank = madErrorRank
    self.madError = madError
    self.accuracy = accuracy
    self.auc = auc
  }

  public init(merging results: [EvaluationResult]) {
    self.madErrorRank = results.map{ $0.madErrorRank }.reduce(0, +) / Float(results.count)
    self.madError = results.map{ $0.madError }.reduce(0, +) / Float(results.count)
    self.accuracy = results.map{ $0.accuracy }.reduce(0, +) / Float(results.count)
    self.auc = results.map{ $0.auc }.reduce(0, +) / Float(results.count)
  }
}

extension EvaluationResult: CustomStringConvertible {
  public var description: String {
    return """
    MAD Error Rank = \(String(format: "%7.4f", madErrorRank)), \
    MAD Error = \(String(format: "%6.4f", madError)), \
    Accuracy = \(String(format: "%6.4f", accuracy)), \
    AUC = \(String(format: "%6.4f", auc))
    """
  }
}

public extension Learner {
  func evaluatePerLabel<Instance, Predictor, Label>(
    using data: Data<Instance, Predictor, Label>
  ) -> [EvaluationResult] {
    // predictedLabelProbabilities is an array of tensors with shape: [BatchSize, ClassCount]
    // predictedQualities shape: [LabelCount, PredictorCount]
    // trueQualities shape: [LabelCount, PredictorCount]
    let predictedLabelProbabilities = labelProbabilities(
      forInstances: data.instanceIndices)
    let predictedQualities = qualities(
      forInstances: data.instanceIndices,
      predictors: data.predictorIndices,
      labels: data.labelIndices)
    // We need to handle missing entries in the predicted qualities, which are marked as -1.0.
    let predictedQualitiesMask = Tensor<Float>(predictedQualities .!= -1.0)
    let predictedQualitiesMean = (
      predictedQualitiesMask * predictedQualities
    ).sum(squeezingAxes: 0) / predictedQualitiesMask.sum(squeezingAxes: 0)
    let trueQualities = data.computeBinaryQualities()
    var results = [EvaluationResult]()
    for label in 0..<predictedLabelProbabilities.count {
      let instances = data.trueLabels[label]!.keys
      let trueLabels = Tensor<Int32>(instances.map { Int32(data.trueLabels[label]![$0]!) })
      let predictedLabelProbabilities = predictedLabelProbabilities[label].gathering(
        atIndices: Tensor<Int32>(instances.map(Int32.init)))
      let predictedQualities = predictedQualitiesMean[label]
      let trueQualities = trueQualities[label]
      results.append(EvaluationResult(
        madErrorRank: computeMADErrorRank(
          estimatedQualities: predictedQualities,
          trueQualities: trueQualities),
        madError: computeMADError(
          estimatedQualities: predictedQualities,
          trueQualities: trueQualities),
        accuracy: computeAccuracy(
          estimatedLabelProbabilities: predictedLabelProbabilities,
          trueLabels: trueLabels),
        auc: computeAUC(
          estimatedLabelProbabilities: predictedLabelProbabilities,
          trueLabels: trueLabels)))
    }
    return results
  }
}
