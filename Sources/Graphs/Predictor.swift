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

public protocol Predictor: Differentiable, KeyPathIterable {
  var nodeCount: Int { get }
  var classCount: Int { get }

  @differentiable
  func callAsFunction(_ nodes: Tensor<Float>, _ neighbors: Tensor<Float>) -> Predictions

  @differentiable
  func labelProbabilities(_ nodes: Tensor<Float>, _ neighbors: Tensor<Float>) -> Tensor<Float>

  @differentiable
  func qualities(_ nodes: Tensor<Float>, _ neighbors: Tensor<Float>) -> Tensor<Float>

  mutating func reset()
}

public struct Predictions: Differentiable {
  public var labelProbabilities: Tensor<Float>
  public var qualities: Tensor<Float>
  public var regularizationTerm: Tensor<Float>

  @inlinable
  @differentiable
  public init(
    labelProbabilities: Tensor<Float>,
    qualities: Tensor<Float>,
    regularizationTerm: Tensor<Float>
  ) {
    self.labelProbabilities = labelProbabilities
    self.qualities = qualities
    self.regularizationTerm = regularizationTerm
  }
}

public struct MLPPredictor: Predictor {
  @noDerivative public let nodeCount: Int
  @noDerivative public let maxNeighborCount: Int
  @noDerivative public let featureCount: Int
  @noDerivative public let classCount: Int
  @noDerivative public let alpha: Float
  @noDerivative public let beta: Float
  @noDerivative public let hiddenUnitCounts: [Int]
  @noDerivative public let confusionLatentSize: Int

  public var nodeProcessingLayers: [Dense<Float>]
  public var predictionLayer: Dense<Float>
  public var difficultyLayer: Sequential<Dense<Float>, Reshape<Float>>
  public var competenceLayer: Sequential<Reshape<Float>, Sequential<Dense<Float>, Reshape<Float>>>

  public init(data: Data, hiddenUnitCounts: [Int], confusionLatentSize: Int, gamma: Float = 0.25) {
    self.nodeCount = data.nodeCount
    self.maxNeighborCount = data.maxNeighborCount
    self.featureCount = data.featureCount
    self.classCount = data.classCount
    self.alpha = 0.5 * gamma * pow(4.0, 2.0)
    self.beta = alpha // * data.avgLabelsPerPredictor / data.avgLabelsPerItem
    self.hiddenUnitCounts = hiddenUnitCounts
    self.confusionLatentSize = confusionLatentSize

    // Create the instance processing layers.
    var inputSize = data.featureCount
    self.nodeProcessingLayers = [Dense<Float>]()
    for hiddenUnitCount in hiddenUnitCounts {
      self.nodeProcessingLayers.append(Dense<Float>(
        inputSize: inputSize,
        outputSize: hiddenUnitCount,
        activation: relu))
      inputSize = hiddenUnitCount
    }
    self.predictionLayer = Dense<Float>(inputSize: inputSize, outputSize: classCount)

    let difficultyShape = Tensor<Int32>(
      [-1, Int32(classCount), Int32(classCount), Int32(confusionLatentSize)])
    self.difficultyLayer = Sequential {
      Dense<Float>(
        inputSize: inputSize,
        outputSize: data.classCount * data.classCount * confusionLatentSize)
      Reshape<Float>(shape: difficultyShape)
    }

    let competenceShape = Tensor<Int32>(
      [-1, Int32(data.maxNeighborCount),
       Int32(classCount), Int32(classCount), Int32(confusionLatentSize)])
    self.competenceLayer = Sequential {
      Reshape<Float>(shape: Tensor<Int32>([-1, Int32(inputSize)]))
      Sequential {
        Dense<Float>(
          inputSize: inputSize,
          outputSize: data.classCount * data.classCount * confusionLatentSize)
        Reshape<Float>(shape: competenceShape)
      }
    }
  }

  @differentiable
  public func callAsFunction(_ nodes: Tensor<Float>, _ neighbors: Tensor<Float>) -> Predictions {
    let nodes = nodeProcessingLayers.differentiableReduce(nodes) { $1($0) }                         // [BatchSize, HiddenSize]
    let neighbors = nodeProcessingLayers.differentiableReduce(neighbors) { $1($0) }                 // [BatchSize, MaxNeighborCount, HiddenSize]
    let labelProbabilities = logSoftmax(predictionLayer(nodes))
    let difficulties = difficultyLayer(nodes).expandingShape(at: 1)                                 // [BatchSize,                1, ClassCount, ClassCount, ConfusionLatentSize]
    let competencies = competenceLayer(neighbors)                                                   // [BatchSize, MaxNeighborCount, ClassCount, ClassCount, ConfusionLatentSize]
    let qualities = logSoftmax(
      difficulties + competencies,
      alongAxis: -2
    ).logSumExp(squeezingAxes: -1)                                                                  // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
    let alpha = self.alpha
    let beta = self.beta
    let regularizationTerm =
      beta * difficulties.squared().sum() +
      alpha * competencies.squared().sum()
    return Predictions(
      labelProbabilities: labelProbabilities,
      qualities: qualities,
      regularizationTerm: regularizationTerm)
  }

  @inlinable
  @differentiable
  public func labelProbabilities(
    _ nodes: Tensor<Float>,
    _ neighbors: Tensor<Float>
  ) -> Tensor<Float> {
    let nodes = nodeProcessingLayers.differentiableReduce(nodes) { $1($0) }                         // [BatchSize, HiddenSize]
    return logSoftmax(predictionLayer(nodes))
  }

  @inlinable
  @differentiable
  public func qualities(_ nodes: Tensor<Float>, _ neighbors: Tensor<Float>) -> Tensor<Float> {
    let nodes = nodeProcessingLayers.differentiableReduce(nodes) { $1($0) }                         // [BatchSize, HiddenSize]
    let neighbors = nodeProcessingLayers.differentiableReduce(neighbors) { $1($0) }                 // [BatchSize, MaxNeighborCount, HiddenSize]
    let difficulties = difficultyLayer(nodes).expandingShape(at: 1)                                 // [BatchSize,                1, ClassCount, ClassCount, ConfusionLatentSize]
    let competencies = competenceLayer(neighbors)                                                   // [BatchSize, MaxNeighborCount, ClassCount, ClassCount, ConfusionLatentSize]
    return logSoftmax(
      difficulties + competencies,
      alongAxis: -2
    ).logSumExp(squeezingAxes: -1)                                                                  // [BatchSize, MaxNeighborCount, ClassCount, ClassCount]
  }

  public mutating func reset() {
    // Create the instance processing layers.
    var inputSize = featureCount
    self.nodeProcessingLayers = [Dense<Float>]()
    for hiddenUnitCount in hiddenUnitCounts {
      self.nodeProcessingLayers.append(Dense<Float>(
        inputSize: inputSize,
        outputSize: hiddenUnitCount,
        activation: relu))
      inputSize = hiddenUnitCount
    }
    self.predictionLayer = Dense<Float>(inputSize: inputSize, outputSize: classCount)

    let difficultyShape = Tensor<Int32>(
      [-1, Int32(classCount), Int32(classCount), Int32(confusionLatentSize)])
    self.difficultyLayer = Sequential {
      Dense<Float>(inputSize: inputSize, outputSize: classCount * classCount * confusionLatentSize)
      Reshape<Float>(shape: difficultyShape)
    }

    let competenceShape = Tensor<Int32>(
      [-1, Int32(data.maxNeighborCount),
       Int32(classCount), Int32(classCount), Int32(confusionLatentSize)])
    self.competenceLayer = Sequential {
      Reshape<Float>(shape: Tensor<Int32>(
        [-1, Int32(classCount), Int32(classCount), Int32(confusionLatentSize)]))
      Sequential {
        Dense<Float>(
          inputSize: inputSize * data.maxNeighborCount,
          outputSize: data.classCount * data.classCount * confusionLatentSize)
        Reshape<Float>(shape: competenceShape)
      }
    }
  }
}
