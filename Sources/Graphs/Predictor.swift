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
  var maxBatchNeighborCount: Int { get }

  @differentiable
  func callAsFunction(_ nodes: Tensor<Int32>, _ neighbors: Tensor<Int32>) -> Predictions

  @differentiable
  func labelProbabilities(_ nodes: Tensor<Int32>) -> Tensor<Float>

  @differentiable
  func qualities(_ nodes: Tensor<Int32>, _ neighbors: Tensor<Int32>) -> Tensor<Float>

  mutating func reset()
}

public struct Predictions: Differentiable {
  public var labelProbabilities: Tensor<Float>
  public var qualities: Tensor<Float>

  @inlinable
  @differentiable
  public init(labelProbabilities: Tensor<Float>, qualities: Tensor<Float>) {
    self.labelProbabilities = labelProbabilities
    self.qualities = qualities
  }
}

public struct MLPPredictor: Predictor {
  @noDerivative public let data: Data
  @noDerivative public let hiddenUnitCounts: [Int]
  @noDerivative public let confusionLatentSize: Int

  @noDerivative public let nodeFeatures: Tensor<Float>

  @noDerivative public var nodeCount: Int { data.nodeCount }
  @noDerivative public var maxNeighborCount: Int { data.maxNeighborCount }
  @noDerivative public var maxBatchNeighborCount: Int { data.maxBatchNeighborCount }
  @noDerivative public var featureCount: Int { data.featureCount }
  @noDerivative public var classCount: Int { data.classCount }

  public var nodeProcessingLayers: [Dense<Float>]
  public var predictionLayer: Dense<Float>
  public var nodeLatentLayer: Sequential<Dense<Float>, Reshape<Float>>
  public var neighborsFlattenLayer: Reshape<Float>
  public var neighborsUnflattenLayer: Reshape<Float>

  public init(data: Data, hiddenUnitCounts: [Int], confusionLatentSize: Int) {
    self.data = data
    self.hiddenUnitCounts = hiddenUnitCounts
    self.confusionLatentSize = confusionLatentSize
    self.nodeFeatures = Tensor<Float>(
      stacking: data.nodeFeatures.map(Tensor<Float>.init),
      alongAxis: 0)

    // Create the instance processing layers.
    var inputSize = data.featureCount
    self.nodeProcessingLayers = [Dense<Float>]()
    for hiddenUnitCount in hiddenUnitCounts {
      self.nodeProcessingLayers.append(Dense<Float>(
        inputSize: inputSize,
        outputSize: hiddenUnitCount,
        activation: { leakyRelu($0) }))
      inputSize = hiddenUnitCount
    }
    self.predictionLayer = Dense<Float>(inputSize: inputSize, outputSize: data.classCount)
    self.nodeLatentLayer = Sequential {
      Dense<Float>(
        inputSize: inputSize,
        outputSize: data.classCount * data.classCount * confusionLatentSize)
      Reshape<Float>(shape: Tensor<Int32>([
        -1, Int32(data.classCount), Int32(data.classCount), Int32(confusionLatentSize)]))
    }
    self.neighborsFlattenLayer = Reshape<Float>(shape: Tensor<Int32>([-1, Int32(inputSize)]))
    self.neighborsUnflattenLayer = Reshape<Float>(shape: Tensor<Int32>([
      -1, Int32(data.maxBatchNeighborCount),
      Int32(data.classCount), Int32(data.classCount),
      Int32(confusionLatentSize)]))
  }

  @differentiable
  public func callAsFunction(_ nodes: Tensor<Int32>, _ neighbors: Tensor<Int32>) -> Predictions {
    let nodeFeatures = self.nodeFeatures.gathering(atIndices: nodes)
    let neighborFeatures = self.nodeFeatures.gathering(atIndices: neighbors)
    let nodes = nodeProcessingLayers.differentiableReduce(nodeFeatures) { $1($0) }                  // [BatchSize, HiddenSize]
    let neighbors = nodeProcessingLayers.differentiableReduce(neighborFeatures) { $1($0) }          // [BatchSize, MaxBatchNeighborCount, HiddenSize]
    let labelProbabilities = logSoftmax(predictionLayer(nodes))
    let projectedNodes = nodeLatentLayer(nodes).expandingShape(at: 1)                               // [BatchSize,                     1, ClassCount, ClassCount, ConfusionLatentSize]
    let projectedNeighbors = neighborsUnflattenLayer(
      nodeLatentLayer(neighborsFlattenLayer(neighbors)))                                            // [BatchSize,                     1, ClassCount, ClassCount, ConfusionLatentSize]
    let qualities = logSoftmax(
      (projectedNodes + projectedNeighbors).logSumExp(squeezingAxes: -1),
      alongAxis: -2)                                                                                // [BatchSize, MaxBatchNeighborCount, ClassCount, ClassCount]
    return Predictions(labelProbabilities: labelProbabilities, qualities: qualities)
  }

  @inlinable
  @differentiable
  public func labelProbabilities(_ nodes: Tensor<Int32>) -> Tensor<Float> {
    let nodeFeatures = self.nodeFeatures.gathering(atIndices: nodes)
    let nodes = nodeProcessingLayers.differentiableReduce(nodeFeatures) { $1($0) }                  // [BatchSize, HiddenSize]
    return logSoftmax(predictionLayer(nodes))
  }

  @inlinable
  @differentiable
  public func qualities(_ nodes: Tensor<Int32>, _ neighbors: Tensor<Int32>) -> Tensor<Float> {
    let nodeFeatures = self.nodeFeatures.gathering(atIndices: nodes)
    let neighborFeatures = self.nodeFeatures.gathering(atIndices: neighbors)
    let nodes = nodeProcessingLayers.differentiableReduce(nodeFeatures) { $1($0) }                  // [BatchSize, HiddenSize]
    let neighbors = nodeProcessingLayers.differentiableReduce(neighborFeatures) { $1($0) }          // [BatchSize, MaxBatchNeighborCount, HiddenSize]
    let projectedNodes = nodeLatentLayer(nodes).expandingShape(at: 1)                               // [BatchSize,                     1, ClassCount, ClassCount, ConfusionLatentSize]
    let projectedNeighbors = neighborsUnflattenLayer(
      nodeLatentLayer(neighborsFlattenLayer(neighbors)))                                            // [BatchSize,                     1, ClassCount, ClassCount, ConfusionLatentSize]
    return logSoftmax(
      (projectedNodes + projectedNeighbors).logSumExp(squeezingAxes: -1),
      alongAxis: -2)                                                                                // [BatchSize, MaxBatchNeighborCount, ClassCount, ClassCount]
  }

  public mutating func reset() {
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
    self.nodeLatentLayer = Sequential {
      Dense<Float>(
        inputSize: inputSize,
        outputSize: data.classCount * data.classCount * confusionLatentSize)
      Reshape<Float>(shape: Tensor<Int32>([
        -1, Int32(data.classCount), Int32(data.classCount), Int32(confusionLatentSize)]))
    }
  }
}

public struct DecoupledMLPPredictor: Predictor {
  @noDerivative public let data: Data
  @noDerivative public let hiddenUnitCounts: [Int]
  @noDerivative public let confusionLatentSize: Int

  @noDerivative public let nodeFeatures: Tensor<Float>

  @noDerivative public var nodeCount: Int { data.nodeCount }
  @noDerivative public var maxNeighborCount: Int { data.maxNeighborCount }
  @noDerivative public var maxBatchNeighborCount: Int { data.maxBatchNeighborCount }
  @noDerivative public var featureCount: Int { data.featureCount }
  @noDerivative public var classCount: Int { data.classCount }

  public var nodeProcessingLayers: [Dense<Float>]
  public var neighborProcessingLayers: [Dense<Float>]
  public var predictionLayer: Dense<Float>
  public var nodeLatentLayer: Sequential<Dense<Float>, Reshape<Float>>
  public var neighborsFlattenLayer: Reshape<Float>
  public var neighborsUnflattenLayer: Reshape<Float>

  public init(data: Data, hiddenUnitCounts: [Int], confusionLatentSize: Int) {
    self.data = data
    self.hiddenUnitCounts = hiddenUnitCounts
    self.confusionLatentSize = confusionLatentSize
    self.nodeFeatures = Tensor<Float>(
      stacking: data.nodeFeatures.map(Tensor<Float>.init),
      alongAxis: 0)

    // Create the instance processing layers.
    var inputSize = data.featureCount
    self.nodeProcessingLayers = [Dense<Float>]()
    self.neighborProcessingLayers = [Dense<Float>]()
    for hiddenUnitCount in hiddenUnitCounts {
      self.nodeProcessingLayers.append(Dense<Float>(
        inputSize: inputSize,
        outputSize: hiddenUnitCount,
        activation: { leakyRelu($0) }))
      self.neighborProcessingLayers.append(Dense<Float>(
        inputSize: inputSize,
        outputSize: hiddenUnitCount,
        activation: { leakyRelu($0) }))
      inputSize = hiddenUnitCount
    }
    self.predictionLayer = Dense<Float>(inputSize: inputSize, outputSize: data.classCount)
    self.nodeLatentLayer = Sequential {
      Dense<Float>(
        inputSize: inputSize,
        outputSize: data.classCount * data.classCount * confusionLatentSize)
      Reshape<Float>(shape: Tensor<Int32>([
        -1, Int32(data.classCount), Int32(data.classCount), Int32(confusionLatentSize)]))
    }
    self.neighborsFlattenLayer = Reshape<Float>(shape: Tensor<Int32>([-1, Int32(inputSize)]))
    self.neighborsUnflattenLayer = Reshape<Float>(shape: Tensor<Int32>([
      -1, Int32(data.maxBatchNeighborCount),
      Int32(data.classCount), Int32(data.classCount),
      Int32(confusionLatentSize)]))
  }

  @differentiable
  public func callAsFunction(_ nodes: Tensor<Int32>, _ neighbors: Tensor<Int32>) -> Predictions {
    let nodeFeatures = self.nodeFeatures.gathering(atIndices: nodes)
    let neighborFeatures = self.nodeFeatures.gathering(atIndices: neighbors)
    let nodes = nodeProcessingLayers.differentiableReduce(nodeFeatures) { $1($0) }                  // [BatchSize, HiddenSize]
    let labelProbabilities = logSoftmax(predictionLayer(nodes))
    let nodesForNeighbors = neighborProcessingLayers.differentiableReduce(nodeFeatures) { $1($0) }  // [BatchSize, HiddenSize]
    let neighbors = neighborProcessingLayers.differentiableReduce(neighborFeatures) { $1($0) }      // [BatchSize, MaxBatchNeighborCount, HiddenSize]
    let projectedNodes = nodeLatentLayer(nodesForNeighbors).expandingShape(at: 1)                   // [BatchSize,                     1, ClassCount, ClassCount, ConfusionLatentSize]
    let projectedNeighbors = neighborsUnflattenLayer(
      nodeLatentLayer(neighborsFlattenLayer(neighbors)))                                            // [BatchSize,                     1, ClassCount, ClassCount, ConfusionLatentSize]
    let qualities = logSoftmax(
      projectedNodes + projectedNeighbors,
      alongAxis: -2
    ).logSumExp(squeezingAxes: -1)                                                                  // [BatchSize, MaxBatchNeighborCount, ClassCount, ClassCount]
    return Predictions(labelProbabilities: labelProbabilities, qualities: qualities)
  }

  @inlinable
  @differentiable
  public func labelProbabilities(_ nodes: Tensor<Int32>) -> Tensor<Float> {
    let nodeFeatures = self.nodeFeatures.gathering(atIndices: nodes)
    let nodes = nodeProcessingLayers.differentiableReduce(nodeFeatures) { $1($0) }                  // [BatchSize, HiddenSize]
    return logSoftmax(predictionLayer(nodes))
  }

  @inlinable
  @differentiable
  public func qualities(_ nodes: Tensor<Int32>, _ neighbors: Tensor<Int32>) -> Tensor<Float> {
    let nodeFeatures = self.nodeFeatures.gathering(atIndices: nodes)
    let neighborFeatures = self.nodeFeatures.gathering(atIndices: neighbors)
    let nodes = neighborProcessingLayers.differentiableReduce(nodeFeatures) { $1($0) }              // [BatchSize, HiddenSize]
    let neighbors = neighborProcessingLayers.differentiableReduce(neighborFeatures) { $1($0) }      // [BatchSize, MaxBatchNeighborCount, HiddenSize]
    let projectedNodes = nodeLatentLayer(nodes).expandingShape(at: 1)                               // [BatchSize,                     1, ClassCount, ClassCount, ConfusionLatentSize]
    let projectedNeighbors = neighborsUnflattenLayer(
      nodeLatentLayer(neighborsFlattenLayer(neighbors)))                                            // [BatchSize,                     1, ClassCount, ClassCount, ConfusionLatentSize]
    return logSoftmax(
      projectedNodes + projectedNeighbors,
      alongAxis: -2
    ).logSumExp(squeezingAxes: -1)                                                                  // [BatchSize, MaxBatchNeighborCount, ClassCount, ClassCount]
  }

  public mutating func reset() {
    var inputSize = featureCount
    self.nodeProcessingLayers = [Dense<Float>]()
    self.neighborProcessingLayers = [Dense<Float>]()
    for hiddenUnitCount in hiddenUnitCounts {
      self.nodeProcessingLayers.append(Dense<Float>(
        inputSize: inputSize,
        outputSize: hiddenUnitCount,
        activation: { leakyRelu($0) }))
      self.neighborProcessingLayers.append(Dense<Float>(
        inputSize: inputSize,
        outputSize: hiddenUnitCount,
        activation: { leakyRelu($0) }))
      inputSize = hiddenUnitCount
    }
    self.predictionLayer = Dense<Float>(inputSize: inputSize, outputSize: classCount)
    self.nodeLatentLayer = Sequential {
      Dense<Float>(
        inputSize: inputSize,
        outputSize: data.classCount * data.classCount * confusionLatentSize)
      Reshape<Float>(shape: Tensor<Int32>([
        -1, Int32(data.classCount), Int32(data.classCount), Int32(confusionLatentSize)]))
    }
  }
}
