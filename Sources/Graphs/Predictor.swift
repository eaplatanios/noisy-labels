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

@usableFromInline
internal struct SparseTensor<Scalar: TensorFlowFloatingPoint> {
  let indices: Tensor<Int32>
  let values: Tensor<Scalar>
  let shape: Tensor<Int64>

  @usableFromInline
  init(indices: Tensor<Int32>, values: Tensor<Scalar>, shape: Tensor<Int64>) {
    self.indices = indices
    self.values = values
    self.shape = shape
  }

  @usableFromInline
  @differentiable(wrt: dense, vjp: _vjpMatmul(withDense:adjointA:adjointB:))
  func matmul(
    withDense dense: Tensor<Scalar>,
    adjointA: Bool = false,
    adjointB: Bool = false
  ) -> Tensor<Scalar> {
    _Raw.sparseTensorDenseMatMul(
      aIndices: indices,
      aValues: values,
      aShape: shape,
      dense,
      adjointA: adjointA,
      adjointB: adjointB)
  }

  @usableFromInline
  func _vjpMatmul(
    withDense dense: Tensor<Scalar>,
    adjointA: Bool = false,
    adjointB: Bool = false
  ) -> (Tensor<Scalar>, (Tensor<Scalar>) -> Tensor<Scalar>) {
    (matmul(withDense: dense, adjointA: adjointA, adjointB: adjointB), { v in
      let gradient = self.matmul(withDense: v, adjointA: !adjointA)
      return adjointB ? gradient.transposed() : gradient
    })
  }
}

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
    var uniqueNodeIndicesMap = [Int32: Int32]()
    for (i, node) in nodes.enumerated() {
      if let index = uniqueNodeIndicesMap[node] {
        nodeIndices.append(index)
      } else {
        let index = Int32(uniqueNodeIndices.count)
        nodeIndices.append(index)
        uniqueNodeIndices.append(node)
        uniqueNodeIndicesMap[node] = index
      }
      for neighbor in graph.neighbors[Int(node)] {
        if let index = uniqueNodeIndicesMap[neighbor] {
          neighborIndices[i].append(index)
        } else {
          let index = Int32(uniqueNodeIndices.count)
          neighborIndices[i].append(index)
          uniqueNodeIndices.append(neighbor)
          uniqueNodeIndicesMap[neighbor] = index
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
  internal func neighborProjectionMatrices(
    depth: Int
  ) -> (nodeIndices: Tensor<Int32>, matrices: [SparseTensor<Float>]) {
    var projectionMatrices = [SparseTensor<Float>]()
    projectionMatrices.reserveCapacity(depth)
    var previousIndices = uniqueNodeIndicesArray
    var indicesMap = [Int32: Int32](
      uniqueKeysWithValues: uniqueNodeIndicesArray.enumerated().map { ($1, Int32($0)) })
    for _ in 0..<depth {
      var currentIndices = previousIndices
      var indices = [Int32]()
      var values = [Float]()
      for (pi, previousIndex) in previousIndices.enumerated() {
        let neighbors = graph.neighbors[Int(previousIndex)]
        let scale = Float(1) / Float(1 + neighbors.count)
        indices.append(Int32(pi))
        indices.append(Int32(pi))
        values.append(scale)
        for neighbor in neighbors {
          if let index = indicesMap[neighbor] {
            indices.append(index)
            indices.append(Int32(pi))
            values.append(scale)
          } else {
            let index = Int32(currentIndices.count)
            indices.append(index)
            indices.append(Int32(pi))
            values.append(scale)
            currentIndices.append(neighbor)
            indicesMap[neighbor] = index
          }
        }
      }
      projectionMatrices.append(SparseTensor(
        indices: Tensor<Int32>(shape: [indices.count / 2, 2], scalars: indices),
        values: Tensor<Float>(values),
        shape: Tensor<Int64>([Int64(currentIndices.count), Int64(previousIndices.count)])))
      previousIndices = currentIndices
    }
    return (nodeIndices: Tensor<Int32>(previousIndices), matrices: projectionMatrices.reversed())
  }
}

public struct GraphPredictions: Differentiable {
  // @noDerivative public var neighborIndices: Tensor<Int32>
  public var labelProbabilities: Tensor<Float>
  public var neighborLabelProbabilities: Tensor<Float>
  public var qualities: Tensor<Float>
  public var qualitiesMask: Tensor<Float>

  @inlinable
  @differentiable
  public init(
    // neighborIndices: Tensor<Int32>,
    labelProbabilities: Tensor<Float>,
    neighborLabelProbabilities: Tensor<Float>,
    qualities: Tensor<Float>,
    qualitiesMask: Tensor<Float>
  ) {
    // self.neighborIndices = neighborIndices
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
    // let neighborLabelProbabilities = withoutDerivative(at: indexMap) {
    //   allProbabilities.gathering(atIndices: $0.neighborIndices)
    // }
    let nodesLatentQ = allLatentQ.gathering(atIndices: indexMap.nodeIndices).expandingShape(at: 1)
    let neighborsLatentQ = allLatentQ.gathering(atIndices: indexMap.neighborIndices)
    let qualities = logSoftmax(
      (nodesLatentQ + neighborsLatentQ).logSumExp(squeezingAxes: -1),
      alongAxis: -2)

    return GraphPredictions(
      // neighborIndices: indexMap.neighborIndices,
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

  public var hiddenLayers: [Sequential<Dropout<Float>, DenseNoBias<Float>>]
  public var predictionLayer: Dense<Float>
  public var nodeLatentLayer: Sequential<Dense<Float>, Reshape<Float>>

  public init(graph: Graph, hiddenUnitCounts: [Int], confusionLatentSize: Int) {
    self.graph = graph
    self.hiddenUnitCounts = hiddenUnitCounts
    self.confusionLatentSize = confusionLatentSize

    var inputSize = graph.featureCount
    self.hiddenLayers = [Sequential<Dropout<Float>, DenseNoBias<Float>>]()
    for hiddenUnitCount in hiddenUnitCounts {
      self.hiddenLayers.append(Sequential {
        Dropout<Float>(probability: 0.5)
        DenseNoBias<Float>(
          inputSize: inputSize,
          outputSize: hiddenUnitCount,
          activation: { leakyRelu($0) })
      })
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
      allFeatures = hiddenLayers[i](projectionMatrices.matrices[i].matmul(
        withDense: allFeatures,
        adjointA: true))
    }
    let allProbabilities = logSoftmax(predictionLayer(allFeatures))
    let allLatentQ = nodeLatentLayer(allFeatures)

    // Split up into the nodes and their neighbors.
    let labelProbabilities = allProbabilities.gathering(atIndices: indexMap.nodeIndices)
    let neighborLabelProbabilities = allProbabilities.gathering(atIndices: indexMap.neighborIndices)
    // let neighborLabelProbabilities = withoutDerivative(at: indexMap) {
    //   allProbabilities.gathering(atIndices: $0.neighborIndices)
    // }
    let nodesLatentQ = allLatentQ.gathering(atIndices: indexMap.nodeIndices).expandingShape(at: 1)
    let neighborsLatentQ = allLatentQ.gathering(atIndices: indexMap.neighborIndices)
    let qualities = logSoftmax(
      (nodesLatentQ * neighborsLatentQ).sum(squeezingAxes: -1),
      alongAxis: -2)

    return GraphPredictions(
      // neighborIndices: indexMap.neighborIndices,
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
      allFeatures = hiddenLayers[i](projectionMatrices.matrices[i].matmul(
        withDense: allFeatures,
        adjointA: true))
    }
    let allProbabilities = logSoftmax(predictionLayer(allFeatures))

    // Split up the nodes from their neighbors.
    return allProbabilities.gathering(atIndices: indexMap.nodeIndices)
  }

  public mutating func reset() {
    var inputSize = graph.featureCount
    self.hiddenLayers = [Sequential<Dropout<Float>, DenseNoBias<Float>>]()
    for hiddenUnitCount in hiddenUnitCounts {
      self.hiddenLayers.append(Sequential {
        Dropout<Float>(probability: 0.5)
        DenseNoBias<Float>(
          inputSize: inputSize,
          outputSize: hiddenUnitCount,
          activation: { leakyRelu($0) })
      })
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

public struct DecoupledGCNPredictor: GraphPredictor {
  @noDerivative public let graph: Graph
  @noDerivative public let lHiddenUnitCounts: [Int]
  @noDerivative public let qHiddenUnitCounts: [Int]

  public var lHiddenLayers: [Sequential<Dropout<Float>, DenseNoBias<Float>>]
  public var qHiddenLayers: [Sequential<Dropout<Float>, DenseNoBias<Float>>]
  public var lOutputLayer: Dense<Float>
  public var qOutputLayer: Sequential<Dense<Float>, Reshape<Float>>

  public init(graph: Graph, lHiddenUnitCounts: [Int], qHiddenUnitCounts: [Int]) {
    self.graph = graph
    self.lHiddenUnitCounts = lHiddenUnitCounts + [graph.classCount]
    self.qHiddenUnitCounts = qHiddenUnitCounts

    var lInputSize = graph.featureCount
    self.lHiddenLayers = [Sequential<Dropout<Float>, DenseNoBias<Float>>]()
    for hiddenUnitCount in (lHiddenUnitCounts + [graph.classCount]) {
      self.lHiddenLayers.append(Sequential {
        Dropout<Float>(probability: 0.5)
        DenseNoBias<Float>(
          inputSize: lInputSize,
          outputSize: hiddenUnitCount,
          activation: { leakyRelu($0) })
      })
      lInputSize = hiddenUnitCount
    }

    var qInputSize = graph.featureCount
    self.qHiddenLayers = [Sequential<Dropout<Float>, DenseNoBias<Float>>]()
    for hiddenUnitCount in qHiddenUnitCounts {
      self.qHiddenLayers.append(Sequential {
        Dropout<Float>(probability: 0.5)
        DenseNoBias<Float>(
          inputSize: qInputSize,
          outputSize: hiddenUnitCount,
          activation: { leakyRelu($0) })
      })
      qInputSize = hiddenUnitCount
    }

    let C = Int32(graph.classCount)
    self.lOutputLayer = Dense<Float>(inputSize: lInputSize, outputSize: Int(C))
    self.qOutputLayer = Sequential {
      Dense<Float>(inputSize: qInputSize, outputSize: Int(C * C))
      Reshape<Float>(shape: Tensor<Int32>([-1, C, C]))
    }
  }

  @differentiable
  public func callAsFunction(_ nodes: [Int32]) -> GraphPredictions {
    // We need features for all provided nodes and their neighbors.
    let indexMap = withoutDerivative(at: nodes) { NodeIndexMap(nodes: $0, graph: graph) }
    let lProjectionMatrices = indexMap.neighborProjectionMatrices(depth: lHiddenUnitCounts.count)

    // Compute the label probabilities.
    var lFeatures = graph.features.gathering(atIndices: lProjectionMatrices.nodeIndices)
    for i in 0..<lHiddenUnitCounts.count {
      lFeatures = lHiddenLayers[i](lProjectionMatrices.matrices[i].matmul(
        withDense: lFeatures,
        adjointA: true))
    }
    let lOutput = logSoftmax(lOutputLayer(lFeatures))
    let labelProbabilities = lOutput.gathering(atIndices: indexMap.nodeIndices)
    let neighborLabelProbabilities = lOutput.gathering(atIndices: indexMap.neighborIndices)
    // let neighborLabelProbabilities = withoutDerivative(at: indexMap) {
    //   lOutput.gathering(atIndices: $0.neighborIndices)
    // }

    // Compute the qualities.
    var qFeatures = graph.features.gathering(atIndices: indexMap.uniqueNodeIndices)
    qFeatures = qHiddenLayers.differentiableReduce(qFeatures) { $1($0) }
    let qOutput = qOutputLayer(qFeatures).gathering(atIndices: indexMap.neighborIndices)
    let qualities = logSoftmax(qOutput, alongAxis: -2)

    return GraphPredictions(
      // neighborIndices: indexMap.neighborIndices,
      labelProbabilities: labelProbabilities,
      neighborLabelProbabilities: neighborLabelProbabilities,
      qualities: qualities,
      qualitiesMask: indexMap.neighborsMask)
  }

  @differentiable
  public func labelProbabilities(_ nodes: [Int32]) -> Tensor<Float> {
    // We need features for all provided nodes and their neighbors.
    let lIndexMap = withoutDerivative(at: nodes) { NodeIndexMap(nodes: $0, graph: graph) }
    let lProjectionMatrices = lIndexMap.neighborProjectionMatrices(depth: lHiddenUnitCounts.count)

    // Compute the label probabilities.
    var lFeatures = graph.features.gathering(atIndices: lProjectionMatrices.nodeIndices)
    for i in 0..<lHiddenUnitCounts.count {
      lFeatures = lHiddenLayers[i](lProjectionMatrices.matrices[i].matmul(
        withDense: lFeatures,
        adjointA: true))
    }
    let lOutput = logSoftmax(lOutputLayer(lFeatures))
    return lOutput.gathering(atIndices: lIndexMap.nodeIndices)
  }

  public mutating func reset() {
    var lInputSize = graph.featureCount
    self.lHiddenLayers = [Sequential<Dropout<Float>, DenseNoBias<Float>>]()
    for hiddenUnitCount in lHiddenUnitCounts {
      self.lHiddenLayers.append(Sequential {
        Dropout<Float>(probability: 0.5)
        DenseNoBias<Float>(
          inputSize: lInputSize,
          outputSize: hiddenUnitCount,
          activation: { leakyRelu($0) })
      })
      lInputSize = hiddenUnitCount
    }

    var qInputSize = graph.featureCount
    self.qHiddenLayers = [Sequential<Dropout<Float>, DenseNoBias<Float>>]()
    for hiddenUnitCount in qHiddenUnitCounts {
      self.qHiddenLayers.append(Sequential {
        Dropout<Float>(probability: 0.5)
        DenseNoBias<Float>(
          inputSize: qInputSize,
          outputSize: hiddenUnitCount,
          activation: { leakyRelu($0) })
      })
      qInputSize = hiddenUnitCount
    }

    let C = Int32(graph.classCount)
    self.lOutputLayer = Dense<Float>(inputSize: lInputSize, outputSize: Int(C))
    self.qOutputLayer = Sequential {
      Dense<Float>(inputSize: qInputSize, outputSize: Int(C * C))
      Reshape<Float>(shape: Tensor<Int32>([-1, C, C]))
    }
  }
}
