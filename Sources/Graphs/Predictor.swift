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
  @differentiable//(wrt: dense, vjp: _vjpMatmul(withDense:adjointA:adjointB:))
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
  @derivative(of: matmul)
  func _vjpMatmul(
    withDense dense: Tensor<Scalar>,
    adjointA: Bool = false,
    adjointB: Bool = false
  ) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> Tensor<Scalar>) {
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
    // TODO: !!! Figure out a way to improve this.
    let maxNeighborCount = graph.maxNodeDegree // neighborIndices.map { $0.count }.max()!
    self.nodeIndicesArray = nodeIndices
    self.neighborIndicesArray = neighborIndices
    self.uniqueNodeIndicesArray = uniqueNodeIndices
    self.nodeIndices = Tensor<Int32>(nodeIndices)
    self.neighborIndices = Tensor<Int32>(
      stacking: neighborIndices.map {
        Tensor<Int32>($0 + [Int32](repeating: 0, count: maxNeighborCount - $0.count))
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
        let scale = Float(1) / Float(neighbors.count)
        indices.append(Int32(pi))
        indices.append(Int32(pi))
        values.append(1)
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

public struct Predictions: Differentiable {
  @noDerivative public var nodes: Tensor<Int32>
  @noDerivative public var neighborIndices: Tensor<Int32>
  @noDerivative public var neighborMask: Tensor<Float>
  public var labelLogits: Tensor<Float>
  public var qualityLogits: Tensor<Float>

  /// This is marked with `@noDerivative` because its gradients are never needed when training.
  @noDerivative public var qualityLogitsTransposed: Tensor<Float>

  @inlinable
  @differentiable(wrt: (labelLogits, qualityLogits))
  public init(
    nodes: Tensor<Int32>,
    neighborIndices: Tensor<Int32>,
    neighborMask: Tensor<Float>,
    labelLogits: Tensor<Float>,
    qualityLogits: Tensor<Float>,
    qualityLogitsTransposed: Tensor<Float>
  ) {
    self.nodes = nodes
    self.neighborIndices = neighborIndices
    self.neighborMask = neighborMask
    self.labelLogits = labelLogits
    self.qualityLogits = qualityLogits
    self.qualityLogitsTransposed = qualityLogitsTransposed
  }
}

extension Predictions {
  @inlinable
  @differentiable
  public init(
    graph: Graph,
    nodes: [Int32],
    labelLogits: Tensor<Float>,
    qualityLogits: Tensor<Float>,
    qualityLogitsTransposed: Tensor<Float>
  ) {
    let neighborIndices = Tensor<Int32>(
      stacking: withoutDerivative(at: nodes) {
        $0.map { graph.neighbors[Int($0)] }.map {
          Tensor<Int32>($0 + [Int32](
            repeating: 0,
            count: graph.maxNodeDegree - $0.count))
        }
      },
      alongAxis: 0)

    let neighborMask = Tensor<Float>(
      stacking: withoutDerivative(at: nodes) {
        $0.map { graph.neighbors[Int($0)] }.map {
          Tensor<Float>(
            [Float](repeating: 1 / Float($0.count), count: $0.count) +
              [Float](repeating: 0, count: graph.maxNodeDegree - $0.count))
        }
      },
      alongAxis: 0)

    self.nodes = Tensor<Int32>(nodes)
    self.neighborIndices = neighborIndices
    self.neighborMask = neighborMask
    self.labelLogits = labelLogits
    self.qualityLogits = qualityLogits
    self.qualityLogitsTransposed = qualityLogitsTransposed
  }
}

public struct InMemoryPredictions {
  public let labelLogits: LabelLogits
  public let qualityLogits: [QualityLogits]
  public let qualityLogitsTransposed: [QualityLogits]
}

extension InMemoryPredictions {
  public init(fromPredictions predictions: Predictions, using graph: Graph) {
    let neighborCounts = Tensor<Int32>(predictions.neighborMask .> 0)
      .sum(squeezingAxes: -1)
      .unstacked(alongAxis: 0)
      .map { Int($0.scalarized()) }
    let h = predictions.labelLogits.scalars
    let q = zip(predictions.qualityLogits.unstacked(alongAxis: 0), neighborCounts).map {
      $0[0..<$1]
    }
    let qT = zip(predictions.qualityLogitsTransposed.unstacked(alongAxis: 0), neighborCounts).map {
      $0[0..<$1]
    }
    self.labelLogits = LabelLogits(
      logits: h,
      nodeCount: graph.nodeCount,
      labelCount: graph.classCount)
    self.qualityLogits = q.map {
      QualityLogits(
        logits: $0.scalars,
        nodeCount: $0.shape[0],
        labelCount: graph.classCount)
    }
    self.qualityLogitsTransposed = qT.map {
      QualityLogits(
        logits: $0.scalars,
        nodeCount: $0.shape[0],
        labelCount: graph.classCount)
    }
  }
}

public struct LabelLogits {
  public let logits: [Float]
  public let nodeCount: Int
  public let labelCount: Int

  public let maxLabelLogits: [Float]

  public init(logits: [Float], nodeCount: Int, labelCount: Int) {
    self.logits = logits
    self.nodeCount = nodeCount
    self.labelCount = labelCount
    self.maxLabelLogits = {
      var maxLabelLogits = [Float]()
      maxLabelLogits.reserveCapacity(nodeCount)
      for node in 0..<nodeCount {
        var maxLogit = logits[node * labelCount]
        for label in 1..<labelCount {
          maxLogit = max(maxLogit, logits[node * labelCount + label])
        }
        maxLabelLogits.append(maxLogit)
      }
      return maxLabelLogits
    }()
  }

  @inlinable
  public func labelLogit(node: Int, label: Int) -> Float {
    logits[node * labelCount + label]
  }

  @inlinable
  public func labelLogits(forNode node: Int) -> [Float] {
    [Float](logits[(node * labelCount)..<((node + 1) * labelCount)])
  }
}

public struct QualityLogits {
  public let logits: [Float]
  public let nodeCount: Int
  public let labelCount: Int
  public let maxQualityLogits: [Float]
  public let maxQualityLogitsPerLabel: [Float]
  public let maxQualityLogitsPerNeighborLabel: [Float]

  public init(logits: [Float], nodeCount: Int, labelCount: Int) {
    self.logits = logits
    self.nodeCount = nodeCount
    self.labelCount = labelCount

    self.maxQualityLogits = {
      var maxQualityLogits = [Float]()
      maxQualityLogits.reserveCapacity(nodeCount)
      for node in 0..<nodeCount {
        var maxLogit = logits[node * labelCount * labelCount]
        for l in 0..<labelCount {
          for k in 0..<labelCount {
            if k != 0 || l > 0 {
              maxLogit = max(maxLogit, logits[node * labelCount * labelCount + l * labelCount + k])
            }
          }
        }
        maxQualityLogits.append(maxLogit)
      }
      return maxQualityLogits
    }()

    self.maxQualityLogitsPerLabel = {
      var maxQualityLogitsPerLabel = [Float]()
      maxQualityLogitsPerLabel.reserveCapacity(nodeCount * labelCount)
      for node in 0..<nodeCount {
        for l in 0..<labelCount {
          var maxLogit = logits[node * labelCount * labelCount + l * labelCount]
          for k in 1..<labelCount {
            maxLogit = max(maxLogit, logits[node * labelCount * labelCount + l * labelCount + k])
          }
          maxQualityLogitsPerLabel.append(maxLogit)
        }
      }
      return maxQualityLogitsPerLabel
    }()

    self.maxQualityLogitsPerNeighborLabel = {
      var maxQualityLogitsPerNeighborLabel = [Float]()
      maxQualityLogitsPerNeighborLabel.reserveCapacity(nodeCount * labelCount)
      for node in 0..<nodeCount {
        for k in 0..<labelCount {
          var maxLogit = logits[node * labelCount * labelCount + k]
          for l in 1..<labelCount {
            maxLogit = max(maxLogit, logits[node * labelCount * labelCount + l * labelCount + k])
          }
          maxQualityLogitsPerNeighborLabel.append(maxLogit)
        }
      }
      return maxQualityLogitsPerNeighborLabel
    }()
  }

  @inlinable
  public func qualityLogit(forNeighbor neighbor: Int, nodeLabel: Int, neighborLabel: Int) -> Float {
    logits[neighbor * labelCount * labelCount + nodeLabel * labelCount + neighborLabel]
  }

  @inlinable
  public func maxQualityLogit(forNeighbor neighbor: Int, nodeLabel: Int) -> Float {
    maxQualityLogitsPerLabel[neighbor * labelCount + nodeLabel]
  }

  @inlinable
  public func maxQualityLogit(forNeighbor neighbor: Int, neighborLabel: Int) -> Float {
    maxQualityLogitsPerNeighborLabel[neighbor * labelCount + neighborLabel]
  }
}

public protocol GraphPredictor: Differentiable, KeyPathIterable {
  @differentiable(wrt: self)
  func predictions(forNodes nodes: Tensor<Int32>, using graph: Graph) -> Predictions

  @differentiable(wrt: self)
  func labelLogits(forNodes nodes: Tensor<Int32>, using graph: Graph) -> Tensor<Float>

  mutating func reset()
}

extension GraphPredictor {
  /// - Note: This is necessary due to a compiler automatic differentiation bug.
  @differentiable(wrt: self)
  public func predictionsHelper(forNodes nodes: Tensor<Int32>, using graph: Graph) -> Predictions {
    self.predictions(forNodes: nodes, using: graph)
  }

  /// - Note: This is necessary due to a compiler automatic differentiation bug.
  @differentiable(wrt: self)
  public func labelLogitsHelper(
    forNodes nodes: Tensor<Int32>,
    using graph: Graph
  ) -> Tensor<Float> {
    self.labelLogits(forNodes: nodes, using: graph)
  }
}

public struct MLPPredictor: GraphPredictor {
  @noDerivative public let featureCount: Int
  @noDerivative public let classCount: Int
  @noDerivative public let hiddenUnitCounts: [Int]
  @noDerivative public let confusionLatentSize: Int
  @noDerivative public let dropout: Float

  public var hiddenLayers: [Sequential<Dropout<Float>, Dense<Float>>]
  @noDerivative public let hiddenDropout: Dropout<Float>
  public var predictionLayer: Dense<Float>
  public var nodeLatentLayer: Dense<Float>
  public var neighborLatentLayer: Dense<Float>

  public init(
    featureCount: Int,
    classCount: Int,
    hiddenUnitCounts: [Int],
    confusionLatentSize: Int,
    dropout: Float
  ) {
    self.featureCount = featureCount
    self.classCount = classCount
    self.hiddenUnitCounts = hiddenUnitCounts
    self.confusionLatentSize = confusionLatentSize
    self.dropout = dropout

    var inputSize = featureCount
    self.hiddenLayers = [Sequential<Dropout<Float>, Dense<Float>>]()
    for hiddenUnitCount in hiddenUnitCounts {
      self.hiddenLayers.append(Sequential {
        Dropout<Float>(probability: Double(dropout))
        Dense<Float>(
          inputSize: inputSize,
          outputSize: hiddenUnitCount,
          activation: gelu)
      })
      inputSize = hiddenUnitCount
    }
    self.hiddenDropout = Dropout<Float>(probability: Double(dropout))
    self.predictionLayer = Dense<Float>(inputSize: inputSize, outputSize: classCount)
    self.nodeLatentLayer = Dense<Float>(inputSize: inputSize, outputSize: classCount * classCount * confusionLatentSize)
    self.neighborLatentLayer = Dense<Float>(inputSize: inputSize, outputSize: classCount * classCount * confusionLatentSize)
  }

  @differentiable(wrt: self)
  public func predictions(forNodes nodes: Tensor<Int32>, using graph: Graph) -> Predictions {
    // We need features for all provided nodes and their neighbors.
    let nodeScalars = nodes.scalars
    let indexMap = NodeIndexMap(nodes: nodeScalars, graph: graph)

    // Compute features, label probabilities, and qualities for all requested nodes.
    let allFeatures = graph.features.gathering(atIndices: indexMap.uniqueNodeIndices)
    let allLatent = hiddenDropout(hiddenLayers.differentiableReduce(allFeatures) { $1($0) })
    let allLabelLogits = logSoftmax(predictionLayer(allLatent))
    let labelLogits = allLabelLogits.gathering(atIndices: indexMap.nodeIndices)

    // Split up into the nodes and their neighbors.
    let C = Int32(classCount)
    let L = Int32(confusionLatentSize)
    let allNodeLatentQ = nodeLatentLayer(allLatent)
    let allNeighborLatentQ = neighborLatentLayer(allLatent)
    let nodesLatentQ = allNodeLatentQ.gathering(atIndices: indexMap.nodeIndices)
      .reshaped(toShape: Tensor<Int32>([-1, 1, C, C, L]))
    let neighborsLatentQ = allNeighborLatentQ.gathering(atIndices: indexMap.neighborIndices)
      .reshaped(toShape: Tensor<Int32>([-1, Int32(indexMap.neighborIndices.shape[1]), C, C, L]))
    let qualityLogits = logSoftmax(
      (nodesLatentQ + neighborsLatentQ).logSumExp(squeezingAxes: -1),
      alongAxis: -2)

    let nodesLatentQTranspose = allNodeLatentQ.gathering(atIndices: indexMap.neighborIndices)
      .reshaped(toShape: Tensor<Int32>([-1, Int32(indexMap.neighborIndices.shape[1]), C, C, L]))
    let neighborsLatentQTranspose = allNeighborLatentQ.gathering(atIndices: indexMap.nodeIndices)
      .reshaped(toShape: Tensor<Int32>([-1, 1, C, C, L]))
    let qualityLogitsTransposed = logSoftmax(
      withoutDerivative(at: nodesLatentQTranspose + neighborsLatentQTranspose) {
        $0.logSumExp(squeezingAxes: -1)
      },
      alongAxis: -1)

    return Predictions(
      graph: graph,
      nodes: nodeScalars,
      labelLogits: labelLogits,
      qualityLogits: qualityLogits,
      qualityLogitsTransposed: qualityLogitsTransposed)
  }

  @differentiable(wrt: self)
  public func labelLogits(forNodes nodes: Tensor<Int32>, using graph: Graph) -> Tensor<Float> {
    let nodeFeatures = graph.features.gathering(atIndices: nodes)
    let nodeLatent = hiddenLayers.differentiableReduce(nodeFeatures) { $1($0) }
    return logSoftmax(predictionLayer(nodeLatent))
  }

  public mutating func reset() {
    var inputSize = featureCount
    self.hiddenLayers = [Sequential<Dropout<Float>, Dense<Float>>]()
    for hiddenUnitCount in hiddenUnitCounts {
      self.hiddenLayers.append(Sequential {
        Dropout<Float>(probability: Double(dropout))
        Dense<Float>(
          inputSize: inputSize,
          outputSize: hiddenUnitCount,
          activation: gelu)
      })
      inputSize = hiddenUnitCount
    }
    self.predictionLayer = Dense<Float>(inputSize: inputSize, outputSize: classCount)
    self.nodeLatentLayer = Dense<Float>(inputSize: inputSize, outputSize: classCount * classCount * confusionLatentSize)
    self.neighborLatentLayer = Dense<Float>(inputSize: inputSize, outputSize: classCount * classCount * confusionLatentSize)
  }
}

//public struct DecoupledMLPPredictor: GraphPredictor {
//  @noDerivative public let graph: Graph
//  @noDerivative public let lHiddenUnitCounts: [Int]
//  @noDerivative public let qHiddenUnitCounts: [Int]
//  @noDerivative public let confusionLatentSize: Int
//  @noDerivative public let dropout: Float
//
//  public var lHiddenLayers: [Sequential<Dropout<Float>, Dense<Float>>]
//  public var qHiddenLayers: [Sequential<Dropout<Float>, Dense<Float>>]
//  public var predictionLayer: Dense<Float>
//  public var qNodeLayer: Dense<Float>
//  public var qNeighborLayer: Dense<Float>
//
//  public init(
//    graph: Graph,
//    lHiddenUnitCounts: [Int],
//    qHiddenUnitCounts: [Int],
//    confusionLatentSize: Int,
//    dropout: Float
//  ) {
//    self.graph = graph
//    self.lHiddenUnitCounts = lHiddenUnitCounts
//    self.qHiddenUnitCounts = qHiddenUnitCounts
//    self.confusionLatentSize = confusionLatentSize
//    self.dropout = dropout
//
//    var lInputSize = graph.featureCount
//    self.lHiddenLayers = [Sequential<Dropout<Float>, Dense<Float>>]()
//    for lHiddenUnitCount in lHiddenUnitCounts {
//      self.lHiddenLayers.append(Sequential {
//        Dropout<Float>(probability: Double(dropout))
//        Dense<Float>(inputSize: lInputSize, outputSize: lHiddenUnitCount, activation: gelu)
//      })
//      lInputSize = lHiddenUnitCount
//    }
//    var qInputSize = graph.featureCount
//    self.qHiddenLayers = [Sequential<Dropout<Float>, Dense<Float>>]()
//    for hiddenUnitCount in qHiddenUnitCounts {
//      self.qHiddenLayers.append(Sequential {
//        Dropout<Float>(probability: Double(dropout))
//        Dense<Float>(inputSize: qInputSize, outputSize: hiddenUnitCount, activation: gelu)
//      })
//      qInputSize = hiddenUnitCount
//    }
//    self.predictionLayer = Dense<Float>(inputSize: lInputSize, outputSize: graph.classCount)
//    self.qNodeLayer = Dense<Float>(inputSize: qInputSize, outputSize: graph.classCount * graph.classCount * confusionLatentSize)
//    self.qNeighborLayer = Dense<Float>(inputSize: qInputSize, outputSize: graph.classCount * graph.classCount * confusionLatentSize)
//  }
//
//  @differentiable(wrt: self)
//  public func callAsFunction(_ nodes: Tensor<Int32>) -> Predictions {
//    // We need features for all provided nodes and their neighbors.
//    let nodeScalars = nodes.scalars
//    let indexMap = NodeIndexMap(nodes: nodeScalars, graph: graph)
//
//    // Compute features, label probabilities, and qualities for all requested nodes.
//    let allFeatures = graph.features.gathering(atIndices: indexMap.uniqueNodeIndices)
//    let allLatent = lHiddenLayers.differentiableReduce(allFeatures) { $1($0) }
//    let allLogits = logSoftmax(predictionLayer(allLatent))
//    let labelLogits = allLogits.gathering(atIndices: indexMap.nodeIndices)
//
//    // Split up into the nodes and their neighbors.
//    let C = Int32(graph.classCount)
//    let L = Int32(confusionLatentSize)
//    let allLatentQ = qHiddenLayers.differentiableReduce(allFeatures) { $1($0) }
//    let nodesLatent = allLatentQ.gathering(atIndices: indexMap.nodeIndices)
//    let neighborsLatent = allLatentQ.gathering(atIndices: indexMap.neighborIndices)
//    let nodesLatentQ = qNodeLayer(nodesLatent)
//      .reshaped(toShape: Tensor<Int32>([-1, 1, C, C, L]))
//    let neighborsLatentQ = qNeighborLayer(neighborsLatent)
//      .reshaped(toShape: Tensor<Int32>([-1, Int32(indexMap.neighborIndices.shape[1]), C, C, L]))
//    let qualityLogits = logSoftmax(
//      (nodesLatentQ + neighborsLatentQ).logSumExp(squeezingAxes: -1),
//      alongAxis: -2)
//
//    let nodesLatentQTranspose = qNodeLayer(neighborsLatent)
//      .reshaped(toShape: Tensor<Int32>([-1, Int32(indexMap.neighborIndices.shape[1]), C, C, L]))
//    let neighborsLatentQTranspose = qNeighborLayer(nodesLatent)
//      .reshaped(toShape: Tensor<Int32>([-1, 1, C, C, L]))
//    let qualityLogitsTransposed = logSoftmax(
//      withoutDerivative(at: nodesLatentQTranspose + neighborsLatentQTranspose) {
//        $0.logSumExp(squeezingAxes: -1)
//      },
//      alongAxis: -1)
//
//    return Predictions(
//      graph: graph,
//      nodes: nodeScalars,
//      labelLogits: labelLogits,
//      qualityLogits: qualityLogits,
//      qualityLogitsTransposed: qualityLogitsTransposed)
//  }
//
//  @differentiable(wrt: self)
//  public func labelLogits(_ nodes: Tensor<Int32>) -> Tensor<Float> {
//    let nodeFeatures = graph.features.gathering(atIndices: nodes)
//    let nodeLatent = lHiddenLayers.differentiableReduce(nodeFeatures) { $1($0) }
//    return logSoftmax(predictionLayer(nodeLatent))
//  }
//
//  public mutating func reset() {
//    var lInputSize = graph.featureCount
//    self.lHiddenLayers = [Sequential<Dropout<Float>, Dense<Float>>]()
//    for lHiddenUnitCount in lHiddenUnitCounts {
//      self.lHiddenLayers.append(Sequential {
//        Dropout<Float>(probability: Double(dropout))
//        Dense<Float>(inputSize: lInputSize, outputSize: lHiddenUnitCount, activation: gelu)
//      })
//      lInputSize = lHiddenUnitCount
//    }
//    var qInputSize = graph.featureCount
//    self.qHiddenLayers = [Sequential<Dropout<Float>, Dense<Float>>]()
//    for hiddenUnitCount in qHiddenUnitCounts {
//      self.qHiddenLayers.append(Sequential {
//        Dropout<Float>(probability: Double(dropout))
//        Dense<Float>(inputSize: qInputSize, outputSize: hiddenUnitCount, activation: gelu)
//      })
//      qInputSize = hiddenUnitCount
//    }
//    self.predictionLayer = Dense<Float>(inputSize: lInputSize, outputSize: graph.classCount)
//    self.qNodeLayer = Dense<Float>(inputSize: qInputSize, outputSize: graph.classCount * graph.classCount * confusionLatentSize)
//    self.qNeighborLayer = Dense<Float>(inputSize: qInputSize, outputSize: graph.classCount * graph.classCount * confusionLatentSize)
//  }
//}

//public struct GCNPredictor: GraphPredictor {
//  @noDerivative public let graph: Graph
//  @noDerivative public let hiddenUnitCounts: [Int]
//  @noDerivative public let dropout: Float
//
//  public var hiddenLayers: [Sequential<DenseNoBias<Float>, Dropout<Float>>]
//  @noDerivative public let hiddenDropout: Dropout<Float>
//  public var predictionLayer: Dense<Float>
//  public var nodeLatentLayer: Dense<Float>
//  public var neighborLatentLayer: Dense<Float>
//
//  public init(graph: Graph, hiddenUnitCounts: [Int], dropout: Float) {
//    self.graph = graph
//    self.hiddenUnitCounts = hiddenUnitCounts
//    self.dropout = dropout
//
//    var inputSize = graph.featureCount
//    self.hiddenLayers = [Sequential<DenseNoBias<Float>, Dropout<Float>>]()
//    for hiddenUnitCount in hiddenUnitCounts {
//      self.hiddenLayers.append(Sequential {
//        DenseNoBias<Float>(inputSize: inputSize, outputSize: hiddenUnitCount)
//        Dropout<Float>(probability: Double(dropout))
//      })
//      inputSize = hiddenUnitCount
//    }
//    self.hiddenDropout = Dropout<Float>(probability: Double(dropout))
//    self.predictionLayer = Dense<Float>(inputSize: inputSize, outputSize: graph.classCount)
//    self.nodeLatentLayer = Dense<Float>(inputSize: inputSize, outputSize: graph.classCount * graph.classCount)
//    self.neighborLatentLayer = Dense<Float>(inputSize: inputSize, outputSize: graph.classCount * graph.classCount)
//  }
//
//  @differentiable(wrt: self)
//  public func callAsFunction(_ nodes: [Int32]) -> Predictions {
//    // We need features for all provided nodes and their neighbors.
//    let indexMap = withoutDerivative(at: nodes) { NodeIndexMap(nodes: $0, graph: graph) }
//    let projectionMatrices = indexMap.neighborProjectionMatrices(depth: hiddenUnitCounts.count)
//
//    // Compute features, label probabilities, and qualities for all requested nodes.
//    var allFeatures = graph.features.gathering(atIndices: projectionMatrices.nodeIndices)
//    for i in 0..<hiddenUnitCounts.count {
//      allFeatures = gelu(projectionMatrices.matrices[i].matmul(
//        withDense: hiddenDropout(hiddenLayers[i](allFeatures)),
//        adjointA: true))
//    }
//    let allProbabilities = logSoftmax(predictionLayer(allFeatures))
//
//    // Split up into the nodes and their neighbors.
//    let labelProbabilities = allProbabilities.gathering(atIndices: indexMap.nodeIndices)
//    let C = Int32(graph.classCount)
//    let nodesLatent = nodeLatentLayer(allFeatures)
//    let neighborsLatent = neighborLatentLayer(allFeatures)
//    let nodesLatentQ = nodesLatent.gathering(atIndices: indexMap.nodeIndices).reshaped(
//        toShape: Tensor<Int32>([-1, 1, C, C]))
//    let neighborsLatentQ = neighborsLatent.gathering(atIndices: indexMap.neighborIndices).reshaped(
//        toShape: Tensor<Int32>([-1, Int32(indexMap.neighborIndices.shape[1]), C, C]))
//    let qualities = logSoftmax(nodesLatentQ + neighborsLatentQ, alongAxis: -2)
//
//    let nodesLatentQTranspose = nodesLatent.gathering(atIndices: indexMap.neighborIndices).reshaped(
//      toShape: Tensor<Int32>([-1, Int32(indexMap.neighborIndices.shape[1]), C, C]))
//    let neighborsLatentQTranspose = nodesLatent.gathering(atIndices: indexMap.nodeIndices).reshaped(
//        toShape: Tensor<Int32>([-1, 1, C, C]))
//    let qualitiesTranspose = logSoftmax(
//      nodesLatentQTranspose + neighborsLatentQTranspose,
//      alongAxis: -2)
//
//    return Predictions(
//      neighborIndices: indexMap.neighborIndices,
//      labelProbabilities: labelProbabilities,
//      qualities: qualities,
//      qualitiesTranspose: qualitiesTranspose,
//      qualitiesMask: indexMap.neighborMask)
//  }
//
//  @differentiable(wrt: self)
//  public func labelProbabilities(_ nodes: [Int32]) -> Tensor<Float> {
//    // We need features for all provided nodes and their neighbors.
//    let indexMap = withoutDerivative(at: nodes) { NodeIndexMap(nodes: $0, graph: graph) }
//    let projectionMatrices = indexMap.neighborProjectionMatrices(depth: hiddenUnitCounts.count)
//
//    // Compute features, label probabilities, and qualities for all requested nodes.
//    var allFeatures = graph.features.gathering(atIndices: projectionMatrices.nodeIndices)
//    for i in 0..<hiddenUnitCounts.count {
//      allFeatures = gelu(projectionMatrices.matrices[i].matmul(
//        withDense: hiddenDropout(hiddenLayers[i](allFeatures)),
//        adjointA: true))
//    }
//    let allProbabilities = logSoftmax(predictionLayer(allFeatures))
//
//    // Split up the nodes from their neighbors.
//    return allProbabilities.gathering(atIndices: indexMap.nodeIndices)
//  }
//
//  public mutating func reset() {
//    var inputSize = graph.featureCount
//    self.hiddenLayers = [Sequential<DenseNoBias<Float>, Dropout<Float>>]()
//    for hiddenUnitCount in hiddenUnitCounts {
//      self.hiddenLayers.append(Sequential {
//        DenseNoBias<Float>(inputSize: inputSize, outputSize: hiddenUnitCount)
//        Dropout<Float>(probability: Double(dropout))
//      })
//      inputSize = hiddenUnitCount
//    }
//    self.predictionLayer = Dense<Float>(inputSize: inputSize, outputSize: graph.classCount)
//    self.nodeLatentLayer = Dense<Float>(inputSize: inputSize, outputSize: graph.classCount * graph.classCount)
//    self.neighborLatentLayer = Dense<Float>(inputSize: inputSize, outputSize: graph.classCount * graph.classCount)
//  }
//}
//
//public struct DecoupledGCNPredictor: GraphPredictor {
//  @noDerivative public let graph: Graph
//  @noDerivative public let lHiddenUnitCounts: [Int]
//  @noDerivative public let qHiddenUnitCounts: [Int]
//  @noDerivative public let confusionLatentSize: Int
//
//  public var lHiddenLayers: [Sequential<Dropout<Float>, DenseNoBias<Float>>]
//  public var qHiddenLayers: [Sequential<Dropout<Float>, DenseNoBias<Float>>]
//  public var lOutputLayer: DenseNoBias<Float>
//  public var qNodeLayer: Dense<Float>
//  public var qNeighborLayer: Dense<Float>
//
//  public init(
//    graph: Graph,
//    lHiddenUnitCounts: [Int],
//    qHiddenUnitCounts: [Int],
//    confusionLatentSize: Int
//  ) {
//    self.graph = graph
//    self.lHiddenUnitCounts = lHiddenUnitCounts
//    self.qHiddenUnitCounts = qHiddenUnitCounts
//    self.confusionLatentSize = confusionLatentSize
//
//    var lInputSize = graph.featureCount
//    self.lHiddenLayers = [Sequential<Dropout<Float>, DenseNoBias<Float>>]()
//    for hiddenUnitCount in lHiddenUnitCounts {
//      self.lHiddenLayers.append(Sequential {
//        Dropout<Float>(probability: 0.5)
//        DenseNoBias<Float>(inputSize: lInputSize, outputSize: hiddenUnitCount)
//      })
//      lInputSize = hiddenUnitCount
//    }
//
//    var qInputSize = graph.featureCount
//    self.qHiddenLayers = [Sequential<Dropout<Float>, DenseNoBias<Float>>]()
//    for hiddenUnitCount in qHiddenUnitCounts {
//      self.qHiddenLayers.append(Sequential {
//        Dropout<Float>(probability: 0.5)
//        DenseNoBias<Float>(inputSize: qInputSize, outputSize: hiddenUnitCount)
//      })
//      qInputSize = hiddenUnitCount
//    }
//
//    let C = Int32(graph.classCount)
//    let L = Int32(confusionLatentSize)
//    self.lOutputLayer = DenseNoBias<Float>(inputSize: lInputSize, outputSize: Int(C))
//    self.qNodeLayer = Dense<Float>(inputSize: qInputSize, outputSize: Int(C * C * L))
//    self.qNeighborLayer = Dense<Float>(inputSize: qInputSize, outputSize: Int(C * C * L))
//  }
//
//  @differentiable(wrt: self)
//  public func callAsFunction(_ nodes: [Int32]) -> Predictions {
//    // We need features for all provided nodes and their neighbors.
//    let indexMap = withoutDerivative(at: nodes) { NodeIndexMap(nodes: $0, graph: graph) }
//    let lProjectionMatrices = indexMap.neighborProjectionMatrices(
//      depth: lHiddenUnitCounts.count + 1)
//
//    // Compute the label probabilities.
//    var lFeatures = graph.features.gathering(atIndices: lProjectionMatrices.nodeIndices)
//    for i in 0..<lHiddenUnitCounts.count {
//      lFeatures = gelu(lProjectionMatrices.matrices[i].matmul(
//        withDense: lHiddenLayers[i](lFeatures),
//        adjointA: true))
//    }
//    let lOutput = logSoftmax(lProjectionMatrices.matrices[lHiddenUnitCounts.count].matmul(
//      withDense: lOutputLayer(lFeatures),
//      adjointA: true))
//    let labelProbabilities = lOutput.gathering(atIndices: indexMap.nodeIndices)
//
//    // Compute the qualities.
//    // TODO: Why not use `indexMap.neighborIndices` directly?
//    let C = Int32(graph.classCount)
//    let L = Int32(confusionLatentSize)
//    let qProjectionMatrices = indexMap.neighborProjectionMatrices(
//      depth: qHiddenUnitCounts.count + 1)
//    var qFeatures = graph.features.gathering(atIndices: qProjectionMatrices.nodeIndices)
//    for i in 0..<qHiddenUnitCounts.count {
//      qFeatures = gelu(qProjectionMatrices.matrices[i].matmul(
//        withDense: qHiddenLayers[i](qFeatures),
//        adjointA: true))
//    }
//    let nodesLatentQ = qProjectionMatrices.matrices[qHiddenUnitCounts.count].matmul(
//      withDense: qNodeLayer(qFeatures),
//      adjointA: true
//    ).gathering(atIndices: indexMap.nodeIndices).reshaped(
//      toShape: Tensor<Int32>([-1, 1, C, C, L]))
//    let neighborsLatentQ = qProjectionMatrices.matrices[qHiddenUnitCounts.count].matmul(
//      withDense: qNeighborLayer(qFeatures),
//      adjointA: true
//    ).gathering(atIndices: indexMap.neighborIndices).reshaped(
//      toShape: Tensor<Int32>([-1, Int32(indexMap.neighborIndices.shape[1]), C, C, L]))
//    let qualities = logSoftmax(
//      (nodesLatentQ + neighborsLatentQ).logSumExp(squeezingAxes: -1),
//      alongAxis: -2)
//
//    return Predictions(
//      neighborIndices: indexMap.neighborIndices,
//      labelProbabilities: labelProbabilities,
//      qualities: qualities,
//      qualitiesTranspose: qualities, // TODO: !!!
//      qualitiesMask: indexMap.neighborMask)
//  }
//
//  @differentiable(wrt: self)
//  public func labelProbabilities(_ nodes: [Int32]) -> Tensor<Float> {
//    // We need features for all provided nodes and their neighbors.
//    let lIndexMap = withoutDerivative(at: nodes) { NodeIndexMap(nodes: $0, graph: graph) }
//    let lProjectionMatrices = lIndexMap.neighborProjectionMatrices(
//      depth: lHiddenUnitCounts.count + 1)
//
//    // Compute the label probabilities.
//    var lFeatures = graph.features.gathering(atIndices: lProjectionMatrices.nodeIndices)
//    for i in 0..<lHiddenUnitCounts.count {
//      lFeatures = gelu(lProjectionMatrices.matrices[i].matmul(
//        withDense: lHiddenLayers[i](lFeatures),
//        adjointA: true))
//    }
//    let lOutput = logSoftmax(lProjectionMatrices.matrices[lHiddenUnitCounts.count].matmul(
//      withDense: lOutputLayer(lFeatures),
//      adjointA: true))
//    return lOutput.gathering(atIndices: lIndexMap.nodeIndices)
//  }
//
//  public mutating func reset() {
//    var lInputSize = graph.featureCount
//    self.lHiddenLayers = [Sequential<Dropout<Float>, DenseNoBias<Float>>]()
//    for hiddenUnitCount in lHiddenUnitCounts {
//      self.lHiddenLayers.append(Sequential {
//        Dropout<Float>(probability: 0.5)
//        DenseNoBias<Float>(inputSize: lInputSize, outputSize: hiddenUnitCount)
//      })
//      lInputSize = hiddenUnitCount
//    }
//
//    var qInputSize = graph.featureCount
//    self.qHiddenLayers = [Sequential<Dropout<Float>, DenseNoBias<Float>>]()
//    for hiddenUnitCount in qHiddenUnitCounts {
//      self.qHiddenLayers.append(Sequential {
//        Dropout<Float>(probability: 0.5)
//        DenseNoBias<Float>(inputSize: qInputSize, outputSize: hiddenUnitCount)
//      })
//      qInputSize = hiddenUnitCount
//    }
//
//    let C = Int32(graph.classCount)
//    let L = Int32(confusionLatentSize)
//    self.lOutputLayer = DenseNoBias<Float>(inputSize: lInputSize, outputSize: Int(C))
//    self.qNodeLayer = Dense<Float>(inputSize: qInputSize, outputSize: Int(C * C * L))
//    self.qNeighborLayer = Dense<Float>(inputSize: qInputSize, outputSize: Int(C * C * L))
//  }
//}
//
//public struct DecoupledGCNPredictorV2: GraphPredictor {
//  @noDerivative public let graph: Graph
//  @noDerivative public let lHiddenUnitCounts: [Int]
//  @noDerivative public let qHiddenUnitCounts: [Int]
//  @noDerivative public let dropout: Float
//
//  public var lHiddenLayers: [Sequential<Dropout<Float>, DenseNoBias<Float>>]
//  public var qHiddenLayers: [Sequential<Dropout<Float>, Dense<Float>>]
//  public var lOutputLayer: DenseNoBias<Float>
//  public var qNodeLayer: Dense<Float>
//  public var qNeighborLayer: Dense<Float>
//
//  public init(graph: Graph, lHiddenUnitCounts: [Int], qHiddenUnitCounts: [Int], dropout: Float) {
//    self.graph = graph
//    self.lHiddenUnitCounts = lHiddenUnitCounts
//    self.qHiddenUnitCounts = qHiddenUnitCounts
//    self.dropout = dropout
//
//    var lInputSize = graph.featureCount
//    self.lHiddenLayers = [Sequential<Dropout<Float>, DenseNoBias<Float>>]()
//    for hiddenUnitCount in lHiddenUnitCounts {
//      self.lHiddenLayers.append(Sequential {
//        Dropout<Float>(probability: Double(dropout))
//        DenseNoBias<Float>(inputSize: lInputSize, outputSize: hiddenUnitCount)
//      })
//      lInputSize = hiddenUnitCount
//    }
//
//    var qInputSize = graph.featureCount
//    self.qHiddenLayers = [Sequential<Dropout<Float>, Dense<Float>>]()
//    for hiddenUnitCount in qHiddenUnitCounts {
//      self.qHiddenLayers.append(Sequential {
//        Dropout<Float>(probability: Double(dropout))
//        Dense<Float>(inputSize: qInputSize, outputSize: hiddenUnitCount, activation: gelu)
//      })
//      qInputSize = hiddenUnitCount
//    }
//    let C = graph.classCount
//    self.lOutputLayer = DenseNoBias<Float>(inputSize: lInputSize, outputSize: C)
//    self.qNodeLayer = Dense<Float>(inputSize: qInputSize, outputSize: C * C)
//    self.qNeighborLayer = Dense<Float>(inputSize: qInputSize, outputSize: C * C)
//  }
//
//  @differentiable(wrt: self)
//  public func callAsFunction(_ nodes: [Int32]) -> Predictions {
//    // We need features for all provided nodes and their neighbors.
//    let indexMap = withoutDerivative(at: nodes) { NodeIndexMap(nodes: $0, graph: graph) }
//    let lProjectionMatrices = indexMap.neighborProjectionMatrices(
//      depth: lHiddenUnitCounts.count + 1)
//
//    // Compute the label probabilities.
//    var lFeatures = graph.features.gathering(atIndices: lProjectionMatrices.nodeIndices)
//    for i in 0..<lHiddenUnitCounts.count {
//      lFeatures = gelu(lProjectionMatrices.matrices[i].matmul(
//        withDense: lHiddenLayers[i](lFeatures),
//        adjointA: true))
//    }
//    let lOutput = logSoftmax(lProjectionMatrices.matrices[lHiddenUnitCounts.count].matmul(
//      withDense: lOutputLayer(lFeatures),
//      adjointA: true))
//    let labelProbabilities = lOutput.gathering(atIndices: indexMap.nodeIndices)
//
//    // Compute the qualities.
//    let C = Int32(graph.classCount)
//    let allFeatures = graph.features.gathering(atIndices: indexMap.uniqueNodeIndices)
//    let allLatentQ = qHiddenLayers.differentiableReduce(allFeatures) { $1($0) }
//    let nodesLatent = allLatentQ.gathering(atIndices: indexMap.nodeIndices)
//    let neighborsLatent = allLatentQ.gathering(atIndices: indexMap.neighborIndices)
//    let nodesLatentQ = qNodeLayer(nodesLatent).reshaped(toShape: Tensor<Int32>([-1, 1, C, C]))
//    let neighborsLatentQ = qNeighborLayer(neighborsLatent)
//      .reshaped(toShape: Tensor<Int32>([-1, Int32(indexMap.neighborIndices.shape[1]), C, C]))
//    let qualities = logSoftmax(nodesLatentQ + neighborsLatentQ, alongAxis: -2)
//
//    let nodesLatentQTranspose = qNodeLayer(neighborsLatent)
//      .reshaped(toShape: Tensor<Int32>([-1, Int32(indexMap.neighborIndices.shape[1]), C, C]))
//    let neighborsLatentQTranspose = qNeighborLayer(nodesLatent)
//      .reshaped(toShape: Tensor<Int32>([-1, 1, C, C]))
//    let qualitiesTranspose = logSoftmax(
//      nodesLatentQTranspose + neighborsLatentQTranspose,
//      alongAxis: -2)
//
//    return Predictions(
//      neighborIndices: indexMap.neighborIndices,
//      labelProbabilities: labelProbabilities,
//      qualities: qualities,
//      qualitiesTranspose: qualitiesTranspose,
//      qualitiesMask: indexMap.neighborMask)
//  }
//
//  @differentiable(wrt: self)
//  public func labelProbabilities(_ nodes: [Int32]) -> Tensor<Float> {
//    // We need features for all provided nodes and their neighbors.
//    let lIndexMap = withoutDerivative(at: nodes) { NodeIndexMap(nodes: $0, graph: graph) }
//    let lProjectionMatrices = lIndexMap.neighborProjectionMatrices(
//      depth: lHiddenUnitCounts.count + 1)
//
//    // Compute the label probabilities.
//    var lFeatures = graph.features.gathering(atIndices: lProjectionMatrices.nodeIndices)
//    for i in 0..<lHiddenUnitCounts.count {
//      lFeatures = gelu(lProjectionMatrices.matrices[i].matmul(
//        withDense: lHiddenLayers[i](lFeatures),
//        adjointA: true))
//    }
//    let lOutput = logSoftmax(lProjectionMatrices.matrices[lHiddenUnitCounts.count].matmul(
//      withDense: lOutputLayer(lFeatures),
//      adjointA: true))
//    return lOutput.gathering(atIndices: lIndexMap.nodeIndices)
//  }
//
//  public mutating func reset() {
//    var lInputSize = graph.featureCount
//    self.lHiddenLayers = [Sequential<Dropout<Float>, DenseNoBias<Float>>]()
//    for hiddenUnitCount in lHiddenUnitCounts {
//      self.lHiddenLayers.append(Sequential {
//        Dropout<Float>(probability: Double(dropout))
//        DenseNoBias<Float>(inputSize: lInputSize, outputSize: hiddenUnitCount)
//      })
//      lInputSize = hiddenUnitCount
//    }
//
//    var qInputSize = graph.featureCount
//    self.qHiddenLayers = [Sequential<Dropout<Float>, Dense<Float>>]()
//    for hiddenUnitCount in qHiddenUnitCounts {
//      self.qHiddenLayers.append(Sequential {
//        Dropout<Float>(probability: Double(dropout))
//        Dense<Float>(inputSize: qInputSize, outputSize: hiddenUnitCount, activation: gelu)
//      })
//      qInputSize = hiddenUnitCount
//    }
//    let C = graph.classCount
//    self.lOutputLayer = DenseNoBias<Float>(inputSize: lInputSize, outputSize: C)
//    self.qNodeLayer = Dense<Float>(inputSize: qInputSize, outputSize: C * C)
//    self.qNeighborLayer = Dense<Float>(inputSize: qInputSize, outputSize: C * C)
//  }
//}
//
//public struct DecoupledGCNPredictorV3: GraphPredictor {
//  @noDerivative public let graph: Graph
//  @noDerivative public let lHiddenUnitCounts: [Int]
//  @noDerivative public let qHiddenUnitCounts: [Int]
//  @noDerivative public let dropout: Float
//
//  public var lHiddenLayers: [Sequential<Dropout<Float>, DenseNoBias<Float>>]
//  public var qHiddenLayers: [Sequential<Dropout<Float>, Dense<Float>>]
//  public var lOutputLayer: DenseNoBias<Float>
//  public var qOutputLayer: Dense<Float>
//
//  public init(graph: Graph, lHiddenUnitCounts: [Int], qHiddenUnitCounts: [Int], dropout: Float) {
//    self.graph = graph
//    self.lHiddenUnitCounts = lHiddenUnitCounts
//    self.qHiddenUnitCounts = qHiddenUnitCounts
//    self.dropout = dropout
//
//    var lInputSize = graph.featureCount
//    self.lHiddenLayers = [Sequential<Dropout<Float>, DenseNoBias<Float>>]()
//    for hiddenUnitCount in lHiddenUnitCounts {
//      self.lHiddenLayers.append(Sequential {
//        Dropout<Float>(probability: Double(dropout))
//        DenseNoBias<Float>(inputSize: lInputSize, outputSize: hiddenUnitCount)
//      })
//      lInputSize = hiddenUnitCount
//    }
//
//    var qInputSize = graph.featureCount
//    self.qHiddenLayers = [Sequential<Dropout<Float>, Dense<Float>>]()
//    for hiddenUnitCount in qHiddenUnitCounts {
//      self.qHiddenLayers.append(Sequential {
//        Dropout<Float>(probability: Double(dropout))
//        Dense<Float>(inputSize: qInputSize, outputSize: hiddenUnitCount, activation: gelu)
//      })
//      qInputSize = hiddenUnitCount
//    }
//    let C = graph.classCount
//    self.lOutputLayer = DenseNoBias<Float>(inputSize: lInputSize, outputSize: C)
//    self.qOutputLayer = Dense<Float>(inputSize: qInputSize, outputSize: C * C)
//  }
//
//  @differentiable(wrt: self)
//  public func callAsFunction(_ nodes: [Int32]) -> Predictions {
//    // We need features for all provided nodes and their neighbors.
//    let indexMap = withoutDerivative(at: nodes) { NodeIndexMap(nodes: $0, graph: graph) }
//    let lProjectionMatrices = indexMap.neighborProjectionMatrices(
//      depth: lHiddenUnitCounts.count + 1)
//
//    // Compute the label probabilities.
//    var lFeatures = graph.features.gathering(atIndices: lProjectionMatrices.nodeIndices)
//    for i in 0..<lHiddenUnitCounts.count {
//      lFeatures = gelu(lProjectionMatrices.matrices[i].matmul(
//        withDense: lHiddenLayers[i](lFeatures),
//        adjointA: true))
//    }
//    let lOutput = logSoftmax(lProjectionMatrices.matrices[lHiddenUnitCounts.count].matmul(
//      withDense: lOutputLayer(lFeatures),
//      adjointA: true))
//    let labelProbabilities = lOutput.gathering(atIndices: indexMap.nodeIndices)
//
//    // Compute the qualities.
//    let allFeatures = graph.features.gathering(atIndices: indexMap.uniqueNodeIndices)
//    let allLatentQ = qHiddenLayers.differentiableReduce(allFeatures) { $1($0) }
//    let L = Int32(qHiddenUnitCounts.last!)
//    let C = Int32(graph.classCount)
//    let nodesLatentQ = allLatentQ.gathering(atIndices: indexMap.nodeIndices)
//      .reshaped(toShape: Tensor<Int32>([-1, 1, L]))
//    let neighborsLatentQ = allLatentQ.gathering(atIndices: indexMap.neighborIndices)
//      .reshaped(toShape: Tensor<Int32>([-1, Int32(indexMap.neighborIndices.shape[1]), L]))
//    let qualities = logSoftmax(
//      qOutputLayer(nodesLatentQ + neighborsLatentQ)
//        .reshaped(toShape: Tensor<Int32>([-1, Int32(indexMap.neighborIndices.shape[1]), C, C])),
//      alongAxis: -2)
//    // let agreements = sigmoid(qOutputLayer((nodesLatentQ - neighborsLatentQ).squared()))
//    // let qualities = log(agreementsToQualities(agreements, classCount: graph.classCount))
//
//    return Predictions(
//      neighborIndices: indexMap.neighborIndices,
//      labelProbabilities: labelProbabilities,
//      qualities: qualities,
//      qualitiesTranspose: qualities, // TODO: !!!
//      qualitiesMask: indexMap.neighborMask)
//  }
//
//  @differentiable(wrt: self)
//  public func labelProbabilities(_ nodes: [Int32]) -> Tensor<Float> {
//    // We need features for all provided nodes and their neighbors.
//    let lIndexMap = withoutDerivative(at: nodes) { NodeIndexMap(nodes: $0, graph: graph) }
//    let lProjectionMatrices = lIndexMap.neighborProjectionMatrices(
//      depth: lHiddenUnitCounts.count + 1)
//
//    // Compute the label probabilities.
//    var lFeatures = graph.features.gathering(atIndices: lProjectionMatrices.nodeIndices)
//    for i in 0..<lHiddenUnitCounts.count {
//      lFeatures = gelu(lProjectionMatrices.matrices[i].matmul(
//        withDense: lHiddenLayers[i](lFeatures),
//        adjointA: true))
//    }
//    let lOutput = logSoftmax(lProjectionMatrices.matrices[lHiddenUnitCounts.count].matmul(
//      withDense: lOutputLayer(lFeatures),
//      adjointA: true))
//    return lOutput.gathering(atIndices: lIndexMap.nodeIndices)
//  }
//
//  public mutating func reset() {
//    var lInputSize = graph.featureCount
//    self.lHiddenLayers = [Sequential<Dropout<Float>, DenseNoBias<Float>>]()
//    for hiddenUnitCount in lHiddenUnitCounts {
//      self.lHiddenLayers.append(Sequential {
//        Dropout<Float>(probability: Double(dropout))
//        DenseNoBias<Float>(inputSize: lInputSize, outputSize: hiddenUnitCount)
//      })
//      lInputSize = hiddenUnitCount
//    }
//
//    var qInputSize = graph.featureCount
//    self.qHiddenLayers = [Sequential<Dropout<Float>, Dense<Float>>]()
//    for hiddenUnitCount in qHiddenUnitCounts {
//      self.qHiddenLayers.append(Sequential {
//        Dropout<Float>(probability: Double(dropout))
//        Dense<Float>(inputSize: qInputSize, outputSize: hiddenUnitCount, activation: gelu)
//      })
//      qInputSize = hiddenUnitCount
//    }
//    let C = graph.classCount
//    self.lOutputLayer = DenseNoBias<Float>(inputSize: lInputSize, outputSize: C)
//    self.qOutputLayer = Dense<Float>(inputSize: qInputSize, outputSize: C * C)
//  }
//}
//
//// @differentiable
//// fileprivate func sinkhorn(_ input: Tensor<Float>) -> Tensor<Float> {
////   // let n = input.shape[1]
////   // let inputShape = [Int](input.shape.dimensions.prefix(upTo: input.rank - 2))
////   var result = input
////   for _ in 0..<5 {
////     result = logSoftmax(result, alongAxis: -2)//.reshaped(to: TensorShape(inputShape + [n, 1]))
////     result = logSoftmax(result, alongAxis: -1)//.reshaped(to: TensorShape(inputShape + [1, n]))
////   }
////   return result
//// }
//
//// @differentiable
//// fileprivate func agreementsToQualities(
////   _ agreements: Tensor<Float>,
////   classCount: Int
//// ) -> Tensor<Float> {
////   let agreementsDiagonal = agreements
////     .tiled(multiples: Tensor<Int32>([1, 1, Int32(classCount)]))
////     .diagonal()
////   let disagreementsDiagonal = agreements
////     .tiled(multiples: Tensor<Int32>([1, 1, Int32(classCount)]))
////     .diagonal()
////   let disagreements = withoutDerivative(at: agreementsDiagonal) { Tensor<Float>(onesLike: $0) } *
////     (1 - agreements.expandingShape(at: -1)) / Float(classCount - 1)
////   return agreementsDiagonal + disagreements - disagreementsDiagonal / Float(classCount - 1)
//// }
