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
  @differentiable(wrt: self)
  func predictions(
    forInstances instances: Tensor<Int32>,
    predictors: Tensor<Int32>,
    labels: Tensor<Int32>
  ) -> MultiLabelPredictions

  @differentiable(wrt: self)
  func labelProbabilities(forInstances instances: Tensor<Int32>) -> Tensor<Float>

  @differentiable(wrt: self)
  func qualities(
    forInstances instances: Tensor<Int32>,
    predictors: Tensor<Int32>,
    labels: Tensor<Int32>
  ) -> Tensor<Float>

  mutating func reset()
}

public struct MultiLabelPredictions: Differentiable {
  public var labelProbabilities: Tensor<Float>
  public var qualities: Tensor<Float>
  public var regularizationTerm: Tensor<Float>
  @noDerivative public let includePredictionsPrior: Bool
}

/// Model proposed in "Regularized Minimax Conditional Entropy for Crowdsourcing".
///
/// Source: https://arxiv.org/pdf/1503.07240.pdf.
public struct MinimaxConditionalEntropyPredictor: MultiLabelPredictor {
  @noDerivative public let instanceCount: Int
  @noDerivative public let predictorCount: Int
  @noDerivative public let labelCount: Int
  @noDerivative public let alpha: Float
  @noDerivative public let beta: Float

  public var pInstanceEmbeddings: Tensor<Float>
  public var qInstanceEmbeddings: Tensor<Float>
  public var qPredictorEmbeddings: Tensor<Float>

  public init(
    instanceCount: Int,
    predictorCount: Int,
    labelCount: Int,
    avgLabelsPerPredictor: Float,
    avgLabelsPerItem: Float,
    gamma: Float = 0.25
  ) {
    self.instanceCount = instanceCount
    self.predictorCount = predictorCount
    self.labelCount = labelCount
    self.alpha = 0.5 * gamma * pow(Float(2 * labelCount), 2.0)
    self.beta = alpha * avgLabelsPerPredictor / avgLabelsPerItem

    pInstanceEmbeddings = Tensor<Float>(glorotUniform: [instanceCount, labelCount]) // TODO: seed.
    qInstanceEmbeddings = Tensor<Float>(glorotUniform: [instanceCount, 2, 2]) // TODO: seed.
    qPredictorEmbeddings = Tensor<Float>(glorotUniform: [predictorCount, 2, 2]) // TODO: seed.
  }

  @differentiable(wrt: self)
  public func predictions(
    forInstances instances: Tensor<Int32>,
    predictors: Tensor<Int32>,
    labels: Tensor<Int32>
  ) -> MultiLabelPredictions {
    // TODO: If I do: let labelProbabilities = self.labelProbabilities(data), then AD does not work.
    let qI = qInstanceEmbeddings.gathering(atIndices: instances)
    let qP = qPredictorEmbeddings.gathering(atIndices: predictors)
    let qualities = logSoftmax(qI + qP)
    let regularizationTerm = beta * (qI * qI).sum() + alpha * (qP * qP).sum()
    return MultiLabelPredictions(
      labelProbabilities: labelProbabilities(forInstances: instances),
      qualities: qualities,
      regularizationTerm: regularizationTerm,
      includePredictionsPrior: false)
  }

  @differentiable(wrt: self)
  public func labelProbabilities(forInstances instances: Tensor<Int32>) -> Tensor<Float> {
    return logSigmoid(pInstanceEmbeddings.gathering(atIndices: instances))
  }

  @differentiable(wrt: self)
  public func qualities(
    forInstances instances: Tensor<Int32>,
    predictors: Tensor<Int32>,
    labels: Tensor<Int32>
  ) -> Tensor<Float> {
    let qI = qInstanceEmbeddings.gathering(atIndices: instances)
    let qP = qPredictorEmbeddings.gathering(atIndices: predictors)
    return logSoftmax(qI + qP)
  }

  public mutating func reset() {
    pInstanceEmbeddings = Tensor<Float>(glorotUniform: [instanceCount, labelCount]) // TODO: seed.
    qInstanceEmbeddings = Tensor<Float>(glorotUniform: [instanceCount, 4]) // TODO: seed.
    qPredictorEmbeddings = Tensor<Float>(glorotUniform: [predictorCount, 4]) // TODO: seed.
  }
}

public struct MultiClassMinimaxConditionalEntropyPredictor: MultiLabelPredictor {
  @noDerivative public let instanceCount: Int
  @noDerivative public let predictorCount: Int
  @noDerivative public let labelCount: Int
  @noDerivative public let numClasses: [Int]
  @noDerivative public let alpha: Float
  @noDerivative public let beta: Float

  public var pInstanceEmbeddings: [Tensor<Float>]
  public var qInstanceEmbeddings: Tensor<Float>
  public var qPredictorEmbeddings: Tensor<Float>

  public init(
    instanceCount: Int,
    predictorCount: Int,
    labelCount: Int,
    numClasses: [Int],
    avgLabelsPerPredictor: Float,
    avgLabelsPerItem: Float,
    gamma: Float = 0.25
  ) {
    self.instanceCount = instanceCount
    self.predictorCount = predictorCount
    self.labelCount = labelCount
    self.numClasses = numClasses
    self.alpha = 0.5 * gamma * pow(Float(2 * labelCount), 2.0)
    self.beta = alpha * avgLabelsPerPredictor / avgLabelsPerItem

    pInstanceEmbeddings = numClasses.map { Tensor<Float>(glorotUniform: [instanceCount, $0]) } // TODO: seed.
    qInstanceEmbeddings = Tensor<Float>(glorotUniform: [instanceCount, 2, 2]) // TODO: seed.
    qPredictorEmbeddings = Tensor<Float>(glorotUniform: [predictorCount, 2, 2]) // TODO: seed.
  }

  @differentiable(wrt: self)
  public func predictions(
    forInstances instances: Tensor<Int32>,
    predictors: Tensor<Int32>,
    labels: Tensor<Int32>
  ) -> MultiLabelPredictions {
    // TODO: If I do: let labelProbabilities = self.labelProbabilities(data), then AD does not work.
    let qI = qInstanceEmbeddings.gathering(atIndices: instances)
    let qP = qPredictorEmbeddings.gathering(atIndices: predictors)
    let qualities = logSoftmax(qI + qP)
    let regularizationTerm = beta * (qI * qI).sum() + alpha * (qP * qP).sum()
    return MultiLabelPredictions(
      labelProbabilities: labelProbabilities(forInstances: instances),
      qualities: qualities,
      regularizationTerm: regularizationTerm,
      includePredictionsPrior: false)
  }

  @differentiable(wrt: self)
  public func labelProbabilities(forInstances instances: Tensor<Int32>) -> [Tensor<Float>] {
    let numLabels = numClasses.count
    for labelId in 0..<numLabels {
      logSigmoid(pInstanceEmbeddings[labelId].gathering(atIndices: instances))
    }
    return logSigmoid(pInstanceEmbeddings.gathering(atIndices: instances))
  }

  @differentiable(wrt: self)
  public func qualities(
    forInstances instances: Tensor<Int32>,
    predictors: Tensor<Int32>,
    labels: Tensor<Int32>
  ) -> Tensor<Float> {
    let qI = qInstanceEmbeddings.gathering(atIndices: instances)
    let qP = qPredictorEmbeddings.gathering(atIndices: predictors)
    return logSoftmax(qI + qP)
  }

  public mutating func reset() {
    pInstanceEmbeddings = Tensor<Float>(glorotUniform: [instanceCount, labelCount]) // TODO: seed.
    qInstanceEmbeddings = Tensor<Float>(glorotUniform: [instanceCount, 4]) // TODO: seed.
    qPredictorEmbeddings = Tensor<Float>(glorotUniform: [predictorCount, 4]) // TODO: seed.
  }
}
