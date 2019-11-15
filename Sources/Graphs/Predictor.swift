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

public struct NodeIndexMap {
  public let graph: Graph

  public let nodeIndicesArray: [Int32]
  public let neighborIndicesArray: [[Int32]]
  public let uniqueNodeIndicesArray: [Int32]

  public let nodeIndices: Tensor<Int32>
  public let neighborIndices: Tensor<Int32>
  public let neighborsMask: Tensor<Float>
  public let uniqueNodeIndices: Tensor<Int32>

  @inlinable
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

  @inlinable
  public func neighborProjectionMatrices(
    depth: Int
  ) -> (nodeIndices: Tensor<Int32>, matrices: [Tensor<Float>]) {
    var projectionMatrices = [Tensor<Float>]()
    projectionMatrices.reserveCapacity(depth)
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
      let adjacencyMatrix = Tensor<Float>(
        stacking: currentAdjacency.map(Tensor<Float>.init),
        alongAxis: 0).transposed()
      projectionMatrices.append(adjacencyMatrix / adjacencyMatrix.sum(alongAxes: 1))
    }
    return (nodeIndices: Tensor<Int32>(previousIndices), matrices: projectionMatrices.reversed())
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
        activation: { relu($0) }))
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
    var allFeatures = graph.features.gathering(atIndices: projectionMatrices.nodeIndices)
    for i in 0..<hiddenUnitCounts.count {
      allFeatures = hiddenLayers[i](matmul(projectionMatrices.matrices[i], allFeatures))
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
    var allFeatures = graph.features.gathering(atIndices: projectionMatrices.nodeIndices)
    for i in 0..<hiddenUnitCounts.count {
      allFeatures = hiddenLayers[i](matmul(projectionMatrices.matrices[i], allFeatures))
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
