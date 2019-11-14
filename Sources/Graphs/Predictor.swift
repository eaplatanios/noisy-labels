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

public typealias NeighborProjectionMatrix =
  (nodeIndices: Tensor<Int32>, adjacencyMatrix: Tensor<Float>)

public struct NodeIndexMap {
  public let graph: Graph
  public let nodeIndicesArray: [Int32]
  public let neighborIndicesArray: [[Int32]]
  public let uniqueNodeIndicesArray: [Int32]

  public let nodeIndices: Tensor<Int32>
  public let neighborIndices: Tensor<Int32>
  public let neighborsMask: Tensor<Float>
  public let uniqueNodeIndices: Tensor<Int32>

  public init(nodes: [Int32], graph: Graph) {
    self.graph = graph
    var nodeIndices = [Int32]()
    var neighborIndices = [[Int32]](repeating: [], count: nodes.count)
    var uniqueNodeIndices = [Int32]()
    for (i, node) in nodes.enumerated() {
      if let index = uniqueNodeIndices.firstIndex(of: node) {
        nodeIndices.append(Int32(index))
      } else {
        nodeIndices.append(Int32(uniqueNodeIndices.count))
        uniqueNodeIndices.append(node)
      }
      for neighbor in graph.neighbors[Int(node)] {
        if let index = uniqueNodeIndices.firstIndex(of: neighbor) {
          neighborIndices[i].append(Int32(index))
        } else {
          neighborIndices[i].append(Int32(uniqueNodeIndices.count))
          uniqueNodeIndices.append(neighbor)
        }
      }
    }
    let maxNeighborCount = neighborIndices.map { $0.count }.max()!
    self.nodeIndicesArray = nodeIndices
    self.neighborIndicesArray = neighborIndices
    self.uniqueNodeIndicesArray = uniqueNodeIndices
    self.nodeIndices = Tensor<Int32>(nodeIndices)
    self.neighborIndices = Tensor<Int32>(
      stacking: neighborIndices.map {
        Tensor<Int32>($0 + [Int32](repeating: 0, count: maxNeighborCount - $0.count))
      },
      alongAxis: 0)
    self.neighborsMask = Tensor<Float>(
      stacking: neighborIndices.map {
        Tensor<Float>(
          [Float](repeating: 1, count: $0.count) +
            [Float](repeating: 0, count: maxNeighborCount - $0.count))
      },
      alongAxis: 0)
    self.uniqueNodeIndices = Tensor<Int32>(uniqueNodeIndices)
  }

  public func neighborProjectionMatrices(depth: Int) -> [NeighborProjectionMatrix] {
    var results = [(Tensor<Int32>, Tensor<Float>)]()
    results.reserveCapacity(depth)
    var previousIndices = uniqueNodeIndicesArray
    for _ in 0..<depth {
      var currentIndices = previousIndices
      var currentAdjacency = [[Float]](
        repeating: [Float](
          repeating: 0,
          count: previousIndices.count),
        count: previousIndices.count)
      for (pi, previousIndex) in previousIndices.enumerated() {
        currentAdjacency[Int(pi)][Int(pi)] = 1
        for neighbor in graph.neighbors[Int(previousIndex)] {
          if let index = currentIndices.firstIndex(of: neighbor) {
            currentAdjacency[Int(index)][Int(pi)] = 1
          } else {
            var newAdjacencyRow = [Float](repeating: 0, count: previousIndices.count)
            newAdjacencyRow[Int(pi)] = 1
            currentAdjacency.append(newAdjacencyRow)
            currentIndices.append(neighbor)
          }
        }
      }
      previousIndices = currentIndices
      let nodeIndices = Tensor<Int32>(currentIndices)
      var adjacencyMatrix = Tensor<Float>(
        stacking: currentAdjacency.map(Tensor<Float>.init),
        alongAxis: 0).transposed()
      adjacencyMatrix /= adjacencyMatrix.sum(alongAxes: 0)
      results.append((nodeIndices: nodeIndices, adjacencyMatrix: adjacencyMatrix))
    }
    return results.reversed()
  }
}

public struct GraphPredictions: Differentiable {
  public var labelProbabilities: Tensor<Float>
  public var neighborLabelProbabilities: Tensor<Float>
  public var qualities: Tensor<Float>
  public var qualitiesMask: Tensor<Float>

  @inlinable
  @differentiable
  public init(
    labelProbabilities: Tensor<Float>,
    neighborLabelProbabilities: Tensor<Float>,
    qualities: Tensor<Float>,
    qualitiesMask: Tensor<Float>
  ) {
    self.labelProbabilities = labelProbabilities
    self.neighborLabelProbabilities = neighborLabelProbabilities
    self.qualities = qualities
    self.qualitiesMask = qualitiesMask
  }
}

public protocol GraphPredictor: Differentiable, KeyPathIterable {
  var graph: Graph { get }

  @differentiable
  func callAsFunction(_ nodes: [Int32]) -> GraphPredictions

  @differentiable
  func labelProbabilities(_ nodes: [Int32]) -> Tensor<Float>

  mutating func reset()
}

extension GraphPredictor {
  public var nodeCount: Int { graph.nodeCount }
  public var featureCount: Int { graph.featureCount }
  public var classCount: Int { graph.classCount }
}

public struct MLPPredictor: GraphPredictor {
  @noDerivative public let graph: Graph
  @noDerivative public let hiddenUnitCounts: [Int]
  @noDerivative public let confusionLatentSize: Int

  public var hiddenLayers: [Dense<Float>]
  public var predictionLayer: Dense<Float>
  public var nodeLatentLayer: Sequential<Dense<Float>, Reshape<Float>>

  public init(graph: Graph, hiddenUnitCounts: [Int], confusionLatentSize: Int) {
    self.graph = graph
    self.hiddenUnitCounts = hiddenUnitCounts
    self.confusionLatentSize = confusionLatentSize

    var inputSize = graph.featureCount
    self.hiddenLayers = [Dense<Float>]()
    for hiddenUnitCount in hiddenUnitCounts {
      self.hiddenLayers.append(Dense<Float>(
        inputSize: inputSize,
        outputSize: hiddenUnitCount,
        activation: { leakyRelu($0) }))
      inputSize = hiddenUnitCount
    }
    let C = Int32(graph.classCount)
    let L = Int32(confusionLatentSize)
    self.predictionLayer = Dense<Float>(inputSize: inputSize, outputSize: Int(C))
    self.nodeLatentLayer = Sequential {
      Dense<Float>(inputSize: inputSize, outputSize: Int(C * C * L))
      Reshape<Float>(shape: Tensor<Int32>([-1, C, C, L]))
    }
  }

  @differentiable
  public func callAsFunction(_ nodes: [Int32]) -> GraphPredictions {
    // We need features for all provided nodes and their neighbors.
    let indexMap = withoutDerivative(at: nodes) { NodeIndexMap(nodes: $0, graph: graph) }

    // Compute features, label probabilities, and qualities for all requested nodes.
    let allFeatures = graph.features.gathering(atIndices: indexMap.uniqueNodeIndices)
    let allLatent = hiddenLayers.differentiableReduce(allFeatures) { $1($0) }
    let allProbabilities = logSoftmax(predictionLayer(allLatent))
    let allLatentQ = nodeLatentLayer(allLatent)

    // Split up into the nodes and their neighbors.
    let labelProbabilities = allProbabilities.gathering(atIndices: indexMap.nodeIndices)
    let neighborLabelProbabilities = allProbabilities.gathering(atIndices: indexMap.neighborIndices)
    let nodesLatentQ = allLatentQ.gathering(atIndices: indexMap.nodeIndices).expandingShape(at: 1)
    let neighborsLatentQ = allLatentQ.gathering(atIndices: indexMap.neighborIndices)
    let qualities = logSoftmax(
      (nodesLatentQ + neighborsLatentQ).logSumExp(squeezingAxes: -1),
      alongAxis: -2)

    return GraphPredictions(
      labelProbabilities: labelProbabilities,
      neighborLabelProbabilities: neighborLabelProbabilities,
      qualities: qualities,
      qualitiesMask: indexMap.neighborsMask)
  }

  @differentiable
  public func labelProbabilities(_ nodes: [Int32]) -> Tensor<Float> {
    let nodeIndices = Tensor<Int32>(nodes)
    let nodeFeatures = graph.features.gathering(atIndices: nodeIndices)
    let nodeLatent = hiddenLayers.differentiableReduce(nodeFeatures) { $1($0) }
    return logSoftmax(predictionLayer(nodeLatent))
  }

  public mutating func reset() {
    var inputSize = graph.featureCount
    self.hiddenLayers = [Dense<Float>]()
    for hiddenUnitCount in hiddenUnitCounts {
      self.hiddenLayers.append(Dense<Float>(
        inputSize: inputSize,
        outputSize: hiddenUnitCount,
        activation: { leakyRelu($0) }))
      inputSize = hiddenUnitCount
    }
    let C = Int32(graph.classCount)
    let L = Int32(confusionLatentSize)
    self.predictionLayer = Dense<Float>(inputSize: inputSize, outputSize: Int(C))
    self.nodeLatentLayer = Sequential {
      Dense<Float>(inputSize: inputSize, outputSize: Int(C * C * L))
      Reshape<Float>(shape: Tensor<Int32>([-1, C, C, L]))
    }
  }
}

public struct GCNPredictor: GraphPredictor {
  @noDerivative public let graph: Graph
  @noDerivative public let hiddenUnitCounts: [Int]
  @noDerivative public let confusionLatentSize: Int

  public var hiddenLayers: [Dense<Float>]
  public var predictionLayer: Dense<Float>
  public var nodeLatentLayer: Sequential<Dense<Float>, Reshape<Float>>

  public init(graph: Graph, hiddenUnitCounts: [Int], confusionLatentSize: Int) {
    self.graph = graph
    self.hiddenUnitCounts = hiddenUnitCounts
    self.confusionLatentSize = confusionLatentSize

    var inputSize = graph.featureCount
    self.hiddenLayers = [Dense<Float>]()
    for hiddenUnitCount in hiddenUnitCounts {
      self.hiddenLayers.append(Dense<Float>(
        inputSize: inputSize,
        outputSize: hiddenUnitCount,
        activation: { leakyRelu($0) }))
      inputSize = hiddenUnitCount
    }
    let C = Int32(graph.classCount)
    let L = Int32(confusionLatentSize)
    self.predictionLayer = Dense<Float>(inputSize: inputSize, outputSize: Int(C))
    self.nodeLatentLayer = Sequential {
      Dense<Float>(inputSize: inputSize, outputSize: Int(C * C * L))
      Reshape<Float>(shape: Tensor<Int32>([-1, C, C, L]))
    }
  }

  @differentiable
  public func callAsFunction(_ nodes: [Int32]) -> GraphPredictions {
    // We need features for all provided nodes and their neighbors.
    let indexMap = withoutDerivative(at: nodes) { NodeIndexMap(nodes: $0, graph: graph) }
    let projectionMatrices = indexMap.neighborProjectionMatrices(depth: hiddenUnitCounts.count)

    // Compute features, label probabilities, and qualities for all requested nodes.
    var allFeatures = graph.features.gathering(atIndices: indexMap.uniqueNodeIndices)
    for i in 0..<hiddenUnitCounts.count {
      allFeatures = leakyRelu(hiddenLayers[i](matmul(
        projectionMatrices[i].adjacencyMatrix, transposed: true,
        allFeatures, transposed: false)))
    }
    let allProbabilities = logSoftmax(predictionLayer(allFeatures))
    let allLatentQ = nodeLatentLayer(allFeatures)

    // Split up into the nodes and their neighbors.
    let labelProbabilities = allProbabilities.gathering(atIndices: indexMap.nodeIndices)
    let neighborLabelProbabilities = allProbabilities.gathering(atIndices: indexMap.neighborIndices)
    let nodesLatentQ = allLatentQ.gathering(atIndices: indexMap.nodeIndices).expandingShape(at: 1)
    let neighborsLatentQ = allLatentQ.gathering(atIndices: indexMap.neighborIndices)
    let qualities = logSoftmax(
      (nodesLatentQ + neighborsLatentQ).logSumExp(squeezingAxes: -1),
      alongAxis: -2)

    return GraphPredictions(
      labelProbabilities: labelProbabilities,
      neighborLabelProbabilities: neighborLabelProbabilities,
      qualities: qualities,
      qualitiesMask: indexMap.neighborsMask)
  }

  @differentiable
  public func labelProbabilities(_ nodes: [Int32]) -> Tensor<Float> {
    // We need features for all provided nodes and their neighbors.
    let indexMap = withoutDerivative(at: nodes) { NodeIndexMap(nodes: $0, graph: graph) }
    let projectionMatrices = indexMap.neighborProjectionMatrices(depth: hiddenUnitCounts.count)

    // Compute features, label probabilities, and qualities for all requested nodes.
    var allFeatures = graph.features.gathering(atIndices: indexMap.uniqueNodeIndices)
    for i in 0..<hiddenUnitCounts.count {
      allFeatures = leakyRelu(hiddenLayers[i](matmul(
        projectionMatrices[i].adjacencyMatrix, transposed: true,
        allFeatures, transposed: false)))
    }
    let allProbabilities = logSoftmax(predictionLayer(allFeatures))

    // Split up the nodes from their neighbors.
    return allProbabilities.gathering(atIndices: indexMap.nodeIndices)
  }

  public mutating func reset() {
    var inputSize = graph.featureCount
    self.hiddenLayers = [Dense<Float>]()
    for hiddenUnitCount in hiddenUnitCounts {
      self.hiddenLayers.append(Dense<Float>(
        inputSize: inputSize,
        outputSize: hiddenUnitCount,
        activation: { leakyRelu($0) }))
      inputSize = hiddenUnitCount
    }
    let C = Int32(graph.classCount)
    let L = Int32(confusionLatentSize)
    self.predictionLayer = Dense<Float>(inputSize: inputSize, outputSize: Int(C))
    self.nodeLatentLayer = Sequential {
      Dense<Float>(inputSize: inputSize, outputSize: Int(C * C * L))
      Reshape<Float>(shape: Tensor<Int32>([-1, C, C, L]))
    }
  }
}

// public protocol Predictor: Differentiable, KeyPathIterable {
//   var nodeCount: Int { get }
//   var classCount: Int { get }
//   var maxBatchNeighborCount: Int { get }

//   @differentiable
//   func callAsFunction(_ nodes: Tensor<Int32>, _ neighbors: Tensor<Int32>) -> Predictions

//   @differentiable
//   func labelProbabilities(_ nodes: Tensor<Int32>) -> Tensor<Float>

//   @differentiable
//   func qualities(_ nodes: Tensor<Int32>, _ neighbors: Tensor<Int32>) -> Tensor<Float>

//   mutating func reset()
// }

// public struct Predictions: Differentiable {
//   public var labelProbabilities: Tensor<Float>
//   public var qualities: Tensor<Float>

//   @inlinable
//   @differentiable
//   public init(labelProbabilities: Tensor<Float>, qualities: Tensor<Float>) {
//     self.labelProbabilities = labelProbabilities
//     self.qualities = qualities
//   }
// }

// public struct MLPPredictor: Predictor {
//   @noDerivative public let graph: Graph
//   @noDerivative public let hiddenUnitCounts: [Int]
//   @noDerivative public let confusionLatentSize: Int

//   @noDerivative public var nodeCount: Int { graph.nodeCount }
//   @noDerivative public var maxNeighborCount: Int { graph.maxNeighborCount }
//   @noDerivative public var maxBatchNeighborCount: Int { graph.maxBatchNeighborCount }
//   @noDerivative public var featureCount: Int { graph.featureCount }
//   @noDerivative public var classCount: Int { graph.classCount }

//   public var nodeProcessingLayers: [Dense<Float>]
//   public var predictionLayer: Dense<Float>
//   public var nodeLatentLayer: Sequential<Dense<Float>, Reshape<Float>>
//   public var neighborsFlattenLayer: Reshape<Float>
//   public var neighborsUnflattenLayer: Reshape<Float>

//   public init(graph: Graph, hiddenUnitCounts: [Int], confusionLatentSize: Int) {
//     self.graph = graph
//     self.hiddenUnitCounts = hiddenUnitCounts
//     self.confusionLatentSize = confusionLatentSize

//     // Create the instance processing layers.
//     var inputSize = graph.featureCount
//     self.nodeProcessingLayers = [Dense<Float>]()
//     for hiddenUnitCount in hiddenUnitCounts {
//       self.nodeProcessingLayers.append(Dense<Float>(
//         inputSize: inputSize,
//         outputSize: hiddenUnitCount,
//         activation: { leakyRelu($0) }))
//       inputSize = hiddenUnitCount
//     }
//     self.predictionLayer = Dense<Float>(inputSize: inputSize, outputSize: graph.classCount)
//     self.nodeLatentLayer = Sequential {
//       Dense<Float>(
//         inputSize: inputSize,
//         outputSize: graph.classCount * graph.classCount * confusionLatentSize)
//       Reshape<Float>(shape: Tensor<Int32>([
//         -1, Int32(graph.classCount), Int32(graph.classCount), Int32(confusionLatentSize)]))
//     }
//     self.neighborsFlattenLayer = Reshape<Float>(shape: Tensor<Int32>([-1, Int32(inputSize)]))
//     self.neighborsUnflattenLayer = Reshape<Float>(shape: Tensor<Int32>([
//       -1, Int32(graph.maxBatchNeighborCount),
//       Int32(graph.classCount), Int32(graph.classCount),
//       Int32(confusionLatentSize)]))
//   }

//   @differentiable
//   public func callAsFunction(_ nodes: Tensor<Int32>, _ neighbors: Tensor<Int32>) -> Predictions {
//     let nodeFeatures = graph.features.gathering(atIndices: nodes)
//     let neighborFeatures = graph.features.gathering(atIndices: neighbors)
//     let nodes = nodeProcessingLayers.differentiableReduce(nodeFeatures) { $1($0) }                  // [BatchSize, HiddenSize]
//     let neighbors = nodeProcessingLayers.differentiableReduce(neighborFeatures) { $1($0) }          // [BatchSize, MaxBatchNeighborCount, HiddenSize]
//     let labelProbabilities = logSoftmax(predictionLayer(nodes))
//     let projectedNodes = nodeLatentLayer(nodes).expandingShape(at: 1)                               // [BatchSize,                     1, ClassCount, ClassCount, ConfusionLatentSize]
//     let projectedNeighbors = neighborsUnflattenLayer(
//       nodeLatentLayer(neighborsFlattenLayer(neighbors)))                                            // [BatchSize,                     1, ClassCount, ClassCount, ConfusionLatentSize]
//     let qualities = logSoftmax(
//       (projectedNodes + projectedNeighbors).logSumExp(squeezingAxes: -1),
//       alongAxis: -2)                                                                                // [BatchSize, MaxBatchNeighborCount, ClassCount, ClassCount]
//     return Predictions(labelProbabilities: labelProbabilities, qualities: qualities)
//   }

//   @inlinable
//   @differentiable
//   public func labelProbabilities(_ nodes: Tensor<Int32>) -> Tensor<Float> {
//     let nodeFeatures = graph.features.gathering(atIndices: nodes)
//     let nodes = nodeProcessingLayers.differentiableReduce(nodeFeatures) { $1($0) }                  // [BatchSize, HiddenSize]
//     return logSoftmax(predictionLayer(nodes))
//   }

//   @inlinable
//   @differentiable
//   public func qualities(_ nodes: Tensor<Int32>, _ neighbors: Tensor<Int32>) -> Tensor<Float> {
//     let nodeFeatures = graph.features.gathering(atIndices: nodes)
//     let neighborFeatures = graph.features.gathering(atIndices: neighbors)
//     let nodes = nodeProcessingLayers.differentiableReduce(nodeFeatures) { $1($0) }                  // [BatchSize, HiddenSize]
//     let neighbors = nodeProcessingLayers.differentiableReduce(neighborFeatures) { $1($0) }          // [BatchSize, MaxBatchNeighborCount, HiddenSize]
//     let projectedNodes = nodeLatentLayer(nodes).expandingShape(at: 1)                               // [BatchSize,                     1, ClassCount, ClassCount, ConfusionLatentSize]
//     let projectedNeighbors = neighborsUnflattenLayer(
//       nodeLatentLayer(neighborsFlattenLayer(neighbors)))                                            // [BatchSize,                     1, ClassCount, ClassCount, ConfusionLatentSize]
//     return logSoftmax(
//       (projectedNodes + projectedNeighbors).logSumExp(squeezingAxes: -1),
//       alongAxis: -2)                                                                                // [BatchSize, MaxBatchNeighborCount, ClassCount, ClassCount]
//   }

//   public mutating func reset() {
//     var inputSize = featureCount
//     self.nodeProcessingLayers = [Dense<Float>]()
//     for hiddenUnitCount in hiddenUnitCounts {
//       self.nodeProcessingLayers.append(Dense<Float>(
//         inputSize: inputSize,
//         outputSize: hiddenUnitCount,
//         activation: relu))
//       inputSize = hiddenUnitCount
//     }
//     self.predictionLayer = Dense<Float>(inputSize: inputSize, outputSize: classCount)
//     self.nodeLatentLayer = Sequential {
//       Dense<Float>(
//         inputSize: inputSize,
//         outputSize: graph.classCount * graph.classCount * confusionLatentSize)
//       Reshape<Float>(shape: Tensor<Int32>([
//         -1, Int32(graph.classCount), Int32(graph.classCount), Int32(confusionLatentSize)]))
//     }
//   }
// }

// public struct DecoupledMLPPredictor: Predictor {
//   @noDerivative public let graph: Graph
//   @noDerivative public let hiddenUnitCounts: [Int]
//   @noDerivative public let confusionLatentSize: Int

//   @noDerivative public var nodeCount: Int { graph.nodeCount }
//   @noDerivative public var maxNeighborCount: Int { graph.maxNeighborCount }
//   @noDerivative public var maxBatchNeighborCount: Int { graph.maxBatchNeighborCount }
//   @noDerivative public var featureCount: Int { graph.featureCount }
//   @noDerivative public var classCount: Int { graph.classCount }

//   public var nodeProcessingLayers: [Dense<Float>]
//   public var neighborProcessingLayers: [Dense<Float>]
//   public var predictionLayer: Dense<Float>
//   public var nodeLatentLayer: Sequential<Dense<Float>, Reshape<Float>>
//   public var neighborsFlattenLayer: Reshape<Float>
//   public var neighborsUnflattenLayer: Reshape<Float>

//   public init(graph: Graph, hiddenUnitCounts: [Int], confusionLatentSize: Int) {
//     self.graph = graph
//     self.hiddenUnitCounts = hiddenUnitCounts
//     self.confusionLatentSize = confusionLatentSize

//     // Create the instance processing layers.
//     var inputSize = graph.featureCount
//     self.nodeProcessingLayers = [Dense<Float>]()
//     self.neighborProcessingLayers = [Dense<Float>]()
//     for hiddenUnitCount in hiddenUnitCounts {
//       self.nodeProcessingLayers.append(Dense<Float>(
//         inputSize: inputSize,
//         outputSize: hiddenUnitCount,
//         activation: { leakyRelu($0) }))
//       self.neighborProcessingLayers.append(Dense<Float>(
//         inputSize: inputSize,
//         outputSize: hiddenUnitCount,
//         activation: { leakyRelu($0) }))
//       inputSize = hiddenUnitCount
//     }
//     self.predictionLayer = Dense<Float>(inputSize: inputSize, outputSize: graph.classCount)
//     self.nodeLatentLayer = Sequential {
//       Dense<Float>(
//         inputSize: inputSize,
//         outputSize: graph.classCount * graph.classCount * confusionLatentSize)
//       Reshape<Float>(shape: Tensor<Int32>([
//         -1, Int32(graph.classCount), Int32(graph.classCount), Int32(confusionLatentSize)]))
//     }
//     self.neighborsFlattenLayer = Reshape<Float>(shape: Tensor<Int32>([-1, Int32(inputSize)]))
//     self.neighborsUnflattenLayer = Reshape<Float>(shape: Tensor<Int32>([
//       -1, Int32(graph.maxBatchNeighborCount),
//       Int32(graph.classCount), Int32(graph.classCount),
//       Int32(confusionLatentSize)]))
//   }

//   @differentiable
//   public func callAsFunction(_ nodes: Tensor<Int32>, _ neighbors: Tensor<Int32>) -> Predictions {
//     let nodeFeatures = graph.features.gathering(atIndices: nodes)
//     let neighborFeatures = graph.features.gathering(atIndices: neighbors)
//     let nodes = nodeProcessingLayers.differentiableReduce(nodeFeatures) { $1($0) }                  // [BatchSize, HiddenSize]
//     let labelProbabilities = logSoftmax(predictionLayer(nodes))
//     let nodesForNeighbors = neighborProcessingLayers.differentiableReduce(nodeFeatures) { $1($0) }  // [BatchSize, HiddenSize]
//     let neighbors = neighborProcessingLayers.differentiableReduce(neighborFeatures) { $1($0) }      // [BatchSize, MaxBatchNeighborCount, HiddenSize]
//     let projectedNodes = nodeLatentLayer(nodesForNeighbors).expandingShape(at: 1)                   // [BatchSize,                     1, ClassCount, ClassCount, ConfusionLatentSize]
//     let projectedNeighbors = neighborsUnflattenLayer(
//       nodeLatentLayer(neighborsFlattenLayer(neighbors)))                                            // [BatchSize,                     1, ClassCount, ClassCount, ConfusionLatentSize]
//     let qualities = logSoftmax(
//       projectedNodes + projectedNeighbors,
//       alongAxis: -2
//     ).logSumExp(squeezingAxes: -1)                                                                  // [BatchSize, MaxBatchNeighborCount, ClassCount, ClassCount]
//     return Predictions(labelProbabilities: labelProbabilities, qualities: qualities)
//   }

//   @inlinable
//   @differentiable
//   public func labelProbabilities(_ nodes: Tensor<Int32>) -> Tensor<Float> {
//     let nodeFeatures = graph.features.gathering(atIndices: nodes)
//     let nodes = nodeProcessingLayers.differentiableReduce(nodeFeatures) { $1($0) }                  // [BatchSize, HiddenSize]
//     return logSoftmax(predictionLayer(nodes))
//   }

//   @inlinable
//   @differentiable
//   public func qualities(_ nodes: Tensor<Int32>, _ neighbors: Tensor<Int32>) -> Tensor<Float> {
//     let nodeFeatures = graph.features.gathering(atIndices: nodes)
//     let neighborFeatures = graph.features.gathering(atIndices: neighbors)
//     let nodes = neighborProcessingLayers.differentiableReduce(nodeFeatures) { $1($0) }              // [BatchSize, HiddenSize]
//     let neighbors = neighborProcessingLayers.differentiableReduce(neighborFeatures) { $1($0) }      // [BatchSize, MaxBatchNeighborCount, HiddenSize]
//     let projectedNodes = nodeLatentLayer(nodes).expandingShape(at: 1)                               // [BatchSize,                     1, ClassCount, ClassCount, ConfusionLatentSize]
//     let projectedNeighbors = neighborsUnflattenLayer(
//       nodeLatentLayer(neighborsFlattenLayer(neighbors)))                                            // [BatchSize,                     1, ClassCount, ClassCount, ConfusionLatentSize]
//     return logSoftmax(
//       projectedNodes + projectedNeighbors,
//       alongAxis: -2
//     ).logSumExp(squeezingAxes: -1)                                                                  // [BatchSize, MaxBatchNeighborCount, ClassCount, ClassCount]
//   }

//   public mutating func reset() {
//     var inputSize = featureCount
//     self.nodeProcessingLayers = [Dense<Float>]()
//     self.neighborProcessingLayers = [Dense<Float>]()
//     for hiddenUnitCount in hiddenUnitCounts {
//       self.nodeProcessingLayers.append(Dense<Float>(
//         inputSize: inputSize,
//         outputSize: hiddenUnitCount,
//         activation: { leakyRelu($0) }))
//       self.neighborProcessingLayers.append(Dense<Float>(
//         inputSize: inputSize,
//         outputSize: hiddenUnitCount,
//         activation: { leakyRelu($0) }))
//       inputSize = hiddenUnitCount
//     }
//     self.predictionLayer = Dense<Float>(inputSize: inputSize, outputSize: classCount)
//     self.nodeLatentLayer = Sequential {
//       Dense<Float>(
//         inputSize: inputSize,
//         outputSize: graph.classCount * graph.classCount * confusionLatentSize)
//       Reshape<Float>(shape: Tensor<Int32>([
//         -1, Int32(graph.classCount), Int32(graph.classCount), Int32(confusionLatentSize)]))
//     }
//   }
// }
