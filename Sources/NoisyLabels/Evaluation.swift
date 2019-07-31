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
import TensorFlow

public func computeMADErrorRank(
  estimatedQualities: Tensor<Float>,
  trueQualities: Tensor<Float>
) -> Float {
  let e = estimatedQualities.scalars
  let t = trueQualities.scalars
  let eRanks = e.indices.sorted(by: { e[$0] > e[$1] })
  let tRanks = t.indices.sorted(by: { t[$0] > t[$1] })
  var madErrorRank: Float = 0.0
  for (eRank, tRank) in zip(eRanks, tRanks) {
    madErrorRank += Float(abs(eRank - tRank))
  }
  return madErrorRank / Float(eRanks.count)
}

public func computeMADError(
  estimatedQualities: Tensor<Float>,
  trueQualities: Tensor<Float>
) -> Float {
  abs(estimatedQualities - trueQualities).mean().scalarized()
}

public func computeAccuracy(
  estimatedLabelProbabilities: Tensor<Float>,
  trueLabels: Tensor<Int32>
) -> Float {
  // We add a small random perturbation in order to randomly break ties.
  let estimates = estimatedLabelProbabilities + Tensor<Float>(
    randomUniform: estimatedLabelProbabilities.shape,
    lowerBound: Tensor<Float>(-1e-3),
    upperBound: Tensor<Float>(1e-3))
  let estimatedLabels = estimates.rank > 1 ?
    Tensor<Int32>(estimates.argmax(squeezingAxis: -1)) :
    Tensor<Int32>(estimates .>= 0.5)
  return Tensor<Float>(estimatedLabels .== trueLabels).mean().scalarized()
}

public func computeAUC(
  estimatedLabelProbabilities: Tensor<Float>,
  trueLabels: Tensor<Int32>
) -> Float {
  let perLabel = estimatedLabelProbabilities.rank > 1 ?
    estimatedLabelProbabilities.unstacked(alongAxis: 1) :
    [estimatedLabelProbabilities]
  let perLabelScalars = perLabel.map { $0.scalars }
  let trueLabelScalars = trueLabels.scalars
  var auc: Float = 0.0
  for l in perLabelScalars.indices {
    let predictions = perLabelScalars[l]
    var scores = [Float]()
    var i = 0
    while i < predictions.count {
      var tiesCount = 0
      var positiveTiesCount = 0
      repeat {
        tiesCount += 1
        if trueLabelScalars[i] == l {
          positiveTiesCount += 1
        }
        i += 1
      } while i < predictions.count && predictions[i - 1] == predictions[i]
      for _ in 0..<(i - scores.count) {
        scores.append(Float(positiveTiesCount) / Float(tiesCount))
      }
    }
    var tp: Float = 0.0 // True positives.
    var fp: Float = 0.0 // False positives.
    var fn: Float = 0.0 // False negatives.
    var previousPrecision: Float = 0.0
    var previousRecall: Float = 1.0
    var currentPrecision: Float = 0.0
    var currentRecall: Float = 0.0
    for i in predictions.indices {
      if trueLabelScalars[i] == l {
        fn += scores[i]
      }
    }
    for i in predictions.indices {
      let score = scores[i]
      if trueLabelScalars[i] == l {
        fn -= score
        tp += score
      } else {
        fp += score
      }
      currentPrecision = tp + fn > 0 ? tp / (tp + fn) : 1.0
      currentRecall = tp + fp > 0 ? tp / (tp + fp) : 1.0
      auc += 0.5 * (currentPrecision - previousPrecision) * (currentRecall + previousRecall)
      previousPrecision = currentPrecision
      previousRecall = currentRecall
    }
  }
  return min(auc / Float(perLabelScalars.count), 1.0)
}

public enum Metric: String {
  case madError, madErrorRank, accuracy, auc
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
    """
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
    let predictedLabelProbabilities = labelProbabilities(data.instanceIndices)
    let predictedQualities = qualities(
      data.instanceIndices,
      data.predictorIndices,
      data.labelIndices)
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
