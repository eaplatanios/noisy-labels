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

public protocol MultiLabelPredictor: Differentiable, KeyPathIterable
where AllDifferentiableVariables: KeyPathIterable {
  var instanceCount: Int { get }
  var predictorCount: Int { get }
  var labelCount: Int { get }
  var classCounts: [Int] { get }

  @differentiable
  func predictions(
    forInstances instances: Tensor<Int32>,
    predictors: Tensor<Int32>,
    labels: Tensor<Int32>
  ) -> MultiLabelPredictions

  @differentiable
  func labelProbabilities(forInstances instances: Tensor<Int32>) -> [Tensor<Float>]

  @differentiable
  func qualities(
    forInstances instances: Tensor<Int32>,
    predictors: Tensor<Int32>,
    labels: Tensor<Int32>
  ) -> [Tensor<Float>]

  mutating func reset()
}

public struct MultiLabelPredictions: Differentiable {
  @differentiable public var labelProbabilities: [Tensor<Float>]
  @differentiable public var qualities: [Tensor<Float>]
  @differentiable public var regularizationTerm: Tensor<Float>
  @noDerivative public let includePredictionsPrior: Bool

  @inlinable
  @differentiable
  public init(
    labelProbabilities: [Tensor<Float>],
    qualities: [Tensor<Float>],
    regularizationTerm: Tensor<Float>,
    includePredictionsPrior: Bool
  ) {
    self.labelProbabilities = labelProbabilities
    self.qualities = qualities
    self.regularizationTerm = regularizationTerm
    self.includePredictionsPrior = includePredictionsPrior
  }
}

/// Model proposed in "Regularized Minimax Conditional Entropy for Crowdsourcing".
///
/// Source: https://arxiv.org/pdf/1503.07240.pdf.
public struct MinimaxConditionalEntropyPredictor: MultiLabelPredictor {
  @noDerivative public let instanceCount: Int
  @noDerivative public let predictorCount: Int
  @noDerivative public let labelCount: Int
  @noDerivative public let classCounts: [Int]
  @noDerivative public let alpha: Float
  @noDerivative public let beta: Float

  /// Instance embeddings (one for each label) used for the label prediction function.
  public var pInstanceEmbeddings: [Tensor<Float>]

  /// Instance embeddings (one for each label) used for the quality prediction function.
  public var qInstanceEmbeddings: [Tensor<Float>]

  /// Predictor embeddings (one for each label) used for the quality prediction function.
  public var qPredictorEmbeddings: [Tensor<Float>]

  public init(
    instanceCount: Int,
    predictorCount: Int,
    labelCount: Int,
    classCounts: [Int],
    avgLabelsPerPredictor: Float,
    avgLabelsPerItem: Float,
    gamma: Float = 0.25
  ) {
    self.instanceCount = instanceCount
    self.predictorCount = predictorCount
    self.labelCount = labelCount
    self.classCounts = classCounts
    self.alpha = 0.5 * gamma * pow(Float(2 * labelCount), 2.0)
    self.beta = alpha * avgLabelsPerPredictor / avgLabelsPerItem
    pInstanceEmbeddings = classCounts.map { Tensor(glorotUniform: [instanceCount, $0]) }
    qInstanceEmbeddings = classCounts.map { Tensor(glorotUniform: [instanceCount, $0, $0]) }
    qPredictorEmbeddings = classCounts.map { Tensor(glorotUniform: [predictorCount, $0, $0]) }
  }

  @differentiable
  public func predictions(
    forInstances instances: Tensor<Int32>,
    predictors: Tensor<Int32>,
    labels: Tensor<Int32>
  ) -> MultiLabelPredictions {
    let qI = qInstanceEmbeddings.differentiableMap { $0.gathering(atIndices: instances) }
    let qP = qPredictorEmbeddings.differentiableMap { $0.gathering(atIndices: predictors) }
    let qualities = differentiableZip(qI, qP).differentiableMap {
      logSoftmax($0.first + $0.second)
    }
    let alpha = self.alpha
    let beta = self.beta
    let regularizationTerms = differentiableZip(
      qI.differentiableMap{ $0.squared().sum() },
      qP.differentiableMap{ $0.squared().sum() }
    ).differentiableMap { beta * $0.first + alpha * $0.second }
    let regularizationTerm = regularizationTerms.differentiableReduce(Tensor(0.0), { $0 + $1 })
    return MultiLabelPredictions(
      labelProbabilities: labelProbabilities(forInstances: instances),
      qualities: qualities,
      regularizationTerm: regularizationTerm,
      includePredictionsPrior: false)
  }

  @differentiable
  public func labelProbabilities(forInstances instances: Tensor<Int32>) -> [Tensor<Float>] {
    pInstanceEmbeddings.differentiableMap {
      logSigmoid($0.gathering(atIndices: instances))
    }
  }

  @differentiable
  public func qualities(
    forInstances instances: Tensor<Int32>,
    predictors: Tensor<Int32>,
    labels: Tensor<Int32>
  ) -> [Tensor<Float>] {
    differentiableZip(qInstanceEmbeddings, qPredictorEmbeddings).differentiableMap {
      logSoftmax(
        $0.first.gathering(atIndices: instances) +
        $0.second.gathering(atIndices: predictors))
    }
  }

  public mutating func reset() {
    pInstanceEmbeddings = classCounts.map { Tensor(glorotUniform: [instanceCount, $0]) }
    qInstanceEmbeddings = classCounts.map { Tensor(glorotUniform: [instanceCount, $0, $0]) }
    qPredictorEmbeddings = classCounts.map { Tensor(glorotUniform: [predictorCount, $0, $0]) }
  }
}
