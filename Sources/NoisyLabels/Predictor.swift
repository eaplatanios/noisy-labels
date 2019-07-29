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
  func callAsFunction(
    _ instances: Tensor<Int32>,
    _ predictors: Tensor<Int32>,
    _ labels: Tensor<Int32>
  ) -> MultiLabelPredictions

  @differentiable
  func labelProbabilities(_ instances: Tensor<Int32>) -> [Tensor<Float>]

  @differentiable
  func qualities(
    _ instances: Tensor<Int32>,
    _ predictors: Tensor<Int32>,
    _ labels: Tensor<Int32>
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

  public init<Instance, Predictor, Label>(
    data: Data<Instance, Predictor, Label>,
    gamma: Float = 0.25
  ) {
    self.instanceCount = data.instances.count
    self.predictorCount = data.predictors.count
    self.labelCount = data.labels.count
    self.classCounts = data.classCounts
    self.alpha = 0.5 * gamma * pow(Float(2 * labelCount), 2.0)
    self.beta = alpha * data.avgLabelsPerPredictor / data.avgLabelsPerItem
    let instanceCount = data.instances.count
    let predictorCount = data.predictors.count
    pInstanceEmbeddings = classCounts.map { Tensor(glorotUniform: [instanceCount, $0]) }
    qInstanceEmbeddings = classCounts.map { Tensor(glorotUniform: [instanceCount, $0, $0]) }
    qPredictorEmbeddings = classCounts.map { Tensor(glorotUniform: [predictorCount, $0, $0]) }
  }

  @differentiable
  public func callAsFunction(
    _ instances: Tensor<Int32>,
    _ predictors: Tensor<Int32>,
    _ labels: Tensor<Int32>
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
      labelProbabilities: labelProbabilities(instances),
      qualities: qualities,
      regularizationTerm: regularizationTerm,
      includePredictionsPrior: false)
  }

  @inlinable
  @differentiable
  public func labelProbabilities(_ instances: Tensor<Int32>) -> [Tensor<Float>] {
    pInstanceEmbeddings.differentiableMap {
      logSoftmax($0.gathering(atIndices: instances))
    }
  }

  @inlinable
  @differentiable
  public func qualities(
    _ instances: Tensor<Int32>,
    _ predictors: Tensor<Int32>,
    _ labels: Tensor<Int32>
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

public struct LNLPredictor: MultiLabelPredictor {
  @noDerivative public let instanceCount: Int
  @noDerivative public let predictorCount: Int
  @noDerivative public let labelCount: Int
  @noDerivative public let classCounts: [Int]
  @noDerivative public let alpha: Float
  @noDerivative public let beta: Float
  @noDerivative public let instanceEmbeddingSize: Int?
  @noDerivative public let predictorEmbeddingSize: Int?
  @noDerivative public let instanceHiddenUnitCounts: [Int]
  @noDerivative public let predictorHiddenUnitCounts: [Int]
  @noDerivative public let confusionLatentSize: Int

  public var instanceFeatures: Tensor<Float>
  public var predictorFeatures: Tensor<Float>
  public var instanceProcessingLayers: [Dense<Float>]
  public var predictorProcessingLayers: [Dense<Float>]
  public var predictionLayers: [Dense<Float>]
  public var difficultyLayers: [Sequential<Dense<Float>, Reshape<Float>>]
  public var competenceLayers: [Sequential<Dense<Float>, Reshape<Float>>]

  public init<Instance, Predictor, Label>(
    data: Data<Instance, Predictor, Label>,
    instanceEmbeddingSize: Int?,
    predictorEmbeddingSize: Int?,
    instanceHiddenUnitCounts: [Int],
    predictorHiddenUnitCounts: [Int],
    confusionLatentSize: Int,
    gamma: Float = 0.25
  ) {
    self.instanceCount = data.instances.count
    self.predictorCount = data.predictors.count
    self.labelCount = data.labels.count
    self.classCounts = data.classCounts
    self.alpha = 0.5 * gamma * pow(Float(2 * labelCount), 2.0)
    self.beta = alpha * data.avgLabelsPerPredictor / data.avgLabelsPerItem
    self.instanceEmbeddingSize = instanceEmbeddingSize
    self.predictorEmbeddingSize = predictorEmbeddingSize
    self.instanceHiddenUnitCounts = instanceHiddenUnitCounts
    self.predictorHiddenUnitCounts = predictorHiddenUnitCounts
    self.confusionLatentSize = confusionLatentSize
    self.instanceFeatures = instanceEmbeddingSize == nil ?
      Tensor(stacking: data.instanceFeatures!, alongAxis: 0) :
      Tensor(glorotUniform: [instanceCount, instanceEmbeddingSize!])
    self.predictorFeatures = predictorEmbeddingSize == nil ?
      Tensor(stacking: data.predictorFeatures!, alongAxis: 0) :
      Tensor(glorotUniform: [predictorCount, predictorEmbeddingSize!])
    
    // Create the instance processing layers.
    var inputSize = instanceFeatures.shape[1]
    self.instanceProcessingLayers = [Dense<Float>]()
    for hiddenUnitCount in instanceHiddenUnitCounts {
      self.instanceProcessingLayers.append(Dense<Float>(
        inputSize: inputSize,
        outputSize: hiddenUnitCount,
        activation: relu))
      inputSize = hiddenUnitCount
    }
    self.predictionLayers = classCounts.map { Dense<Float>(inputSize: inputSize, outputSize: $0) }
    self.difficultyLayers = classCounts.map { count in
      let resultShape = Tensor<Int32>([-1, Int32(count), Int32(count), Int32(confusionLatentSize)])
      return Sequential {
        Dense<Float>(inputSize: inputSize, outputSize: count * count * confusionLatentSize)
        Reshape<Float>(shape: resultShape)
      }
    }

    // Create the predictor processing layers.
    inputSize = predictorFeatures.shape[1]
    self.predictorProcessingLayers = [Dense<Float>]()
    for hiddenUnitCount in predictorHiddenUnitCounts {
      self.predictorProcessingLayers.append(Dense<Float>(
        inputSize: inputSize,
        outputSize: hiddenUnitCount,
        activation: relu))
      inputSize = hiddenUnitCount
    }
    self.competenceLayers = classCounts.map { count in
      let resultShape = Tensor<Int32>([-1, Int32(count), Int32(count), Int32(confusionLatentSize)])
      return Sequential {
        Dense<Float>(inputSize: inputSize, outputSize: count * count * confusionLatentSize)
        Reshape<Float>(shape: resultShape)
      }
    }
  }

  @differentiable
  public func callAsFunction(
    _ instances: Tensor<Int32>,
    _ predictors: Tensor<Int32>,
    _ labels: Tensor<Int32>
  ) -> MultiLabelPredictions {
    let instances = instanceProcessingLayers.differentiableReduce(
      instanceFeatures.gathering(atIndices: instances),
      { (instances, layer) in layer(instances) })
    let predictors = predictorProcessingLayers.differentiableReduce(
      predictorFeatures.gathering(atIndices: predictors),
      { (predictors, layer) in layer(predictors) })
    let labelProbabilities = predictionLayers.differentiableMap(instances) { logSoftmax($1($0)) }
    let difficulties = difficultyLayers.differentiableMap(instances) { $1($0) }
    let competences = competenceLayers.differentiableMap(predictors) { $1($0) }
    let qualities = differentiableZip(difficulties, competences).differentiableMap {
      logSoftmax($0.first + $0.second, alongAxis: -2).logSumExp(squeezingAxes: -1)
    }
    let alpha = self.alpha
    let beta = self.beta
    let regularizationTerms = differentiableZip(
      difficulties.differentiableMap{ $0.squared().sum() },
      competences.differentiableMap{ $0.squared().sum() }
    ).differentiableMap { beta * $0.first + alpha * $0.second }
    let regularizationTerm = regularizationTerms.differentiableReduce(Tensor(0.0), { $0 + $1 })
    return MultiLabelPredictions(
      labelProbabilities: labelProbabilities,
      qualities: qualities,
      regularizationTerm: regularizationTerm,
      includePredictionsPrior: true)
  }

  @inlinable
  @differentiable
  public func labelProbabilities(_ instances: Tensor<Int32>) -> [Tensor<Float>] {
    let instances = instanceProcessingLayers.differentiableReduce(
      instanceFeatures.gathering(atIndices: instances),
      { (instances, layer) in layer(instances) })
    return predictionLayers.differentiableMap(instances) { logSoftmax($1($0)) }
  }

  @inlinable
  @differentiable
  public func qualities(
    _ instances: Tensor<Int32>,
    _ predictors: Tensor<Int32>,
    _ labels: Tensor<Int32>
  ) -> [Tensor<Float>] {
    let instances = instanceProcessingLayers.differentiableReduce(
      instanceFeatures.gathering(atIndices: instances),
      { (instances, layer) in layer(instances) })
    let predictors = predictorProcessingLayers.differentiableReduce(
      predictorFeatures.gathering(atIndices: predictors),
      { (predictors, layer) in layer(predictors) })
    let difficulties = difficultyLayers.differentiableMap(instances) { $1($0) }
    let competences = competenceLayers.differentiableMap(predictors) { $1($0) }
    return differentiableZip(difficulties, competences).differentiableMap {
      logSoftmax($0.first + $0.second).logSumExp(squeezingAxes: -1)
    }
  }

  public mutating func reset() {
    if let embeddingSize = instanceEmbeddingSize {
      instanceFeatures = Tensor(glorotUniform: [instanceCount, embeddingSize])
    }

    if let embeddingSize = predictorEmbeddingSize {
      predictorFeatures = Tensor(glorotUniform: [predictorCount, embeddingSize])
    }

    // Create the instance processing layers.
    var inputSize = instanceFeatures.shape[1]
    instanceProcessingLayers = [Dense<Float>]()
    for hiddenUnitCount in instanceHiddenUnitCounts {
      instanceProcessingLayers.append(Dense<Float>(
        inputSize: inputSize,
        outputSize: hiddenUnitCount,
        activation: relu))
      inputSize = hiddenUnitCount
    }
    predictionLayers = classCounts.map { Dense<Float>(inputSize: inputSize, outputSize: $0) }
    difficultyLayers = classCounts.map { count in
      let resultShape = Tensor<Int32>([-1, Int32(count), Int32(count), Int32(confusionLatentSize)])
      return Sequential {
        Dense<Float>(inputSize: inputSize, outputSize: count * count * confusionLatentSize)
        Reshape<Float>(shape: resultShape)
      }
    }

    // Create the predictor processing layers.
    inputSize = predictorFeatures.shape[1]
    predictorProcessingLayers = [Dense<Float>]()
    for hiddenUnitCount in predictorHiddenUnitCounts {
      predictorProcessingLayers.append(Dense<Float>(
        inputSize: inputSize,
        outputSize: hiddenUnitCount,
        activation: relu))
      inputSize = hiddenUnitCount
    }
    competenceLayers = classCounts.map { count in
      let resultShape = Tensor<Int32>([-1, Int32(count), Int32(count), Int32(confusionLatentSize)])
      return Sequential {
        Dense<Float>(inputSize: inputSize, outputSize: count * count * confusionLatentSize)
        Reshape<Float>(shape: resultShape)
      }
    }
  }
}
