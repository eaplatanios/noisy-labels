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

public struct LabeledData: TensorGroup {
  public let nodes: Tensor<Int32>  // [BatchSize]
  public let labels: Tensor<Int32> // [BatchSize]

  public init(nodeIndices: Tensor<Int32>, nodeLabels: Tensor<Int32>) {
    self.nodes = nodeIndices
    self.labels = nodeLabels
  }

  public init<C: RandomAccessCollection>(_handles: C) where C.Element: _AnyTensorHandle {
    precondition(_handles.count == 2)
    let niIndex = _handles.startIndex
    let nlIndex = _handles.index(niIndex, offsetBy: 1)
    self.nodes = Tensor<Int32>(handle: TensorHandle<Int32>(handle: _handles[niIndex]))
    self.labels = Tensor<Int32>(handle: TensorHandle<Int32>(handle: _handles[nlIndex]))
  }

  public var _tensorHandles: [_AnyTensorHandle] {
    nodes._tensorHandles + labels._tensorHandles
  }
}

public struct Graph {
  public let nodeCount: Int
  public let featureCount: Int
  public let classCount: Int
  public let features: Tensor<Float>
  public let neighbors: [[Int32]]
  public let labels: [Int32: Int]
  public let trainNodes: [Int32]
  public let validationNodes: [Int32]
  public let testNodes: [Int32]
  public let otherNodes: [Int32]
  public let groupedNodes: [[Int32]]

  public init(
    nodeCount: Int,
    featureCount: Int,
    classCount: Int,
    features: Tensor<Float>,
    neighbors: [[Int32]],
    labels: [Int32: Int],
    trainNodes: [Int32],
    validationNodes: [Int32],
    testNodes: [Int32],
    otherNodes: [Int32]
  ) {
    self.nodeCount = nodeCount
    self.featureCount = featureCount
    self.classCount = classCount
    self.features = features
    self.neighbors = neighbors
    self.labels = labels
    self.trainNodes = trainNodes
    self.validationNodes = validationNodes
    self.testNodes = testNodes
    self.otherNodes = otherNodes

    // Compute the grouped nodes.
    var groupedNodes = [[Int32]]()
    var allNodes = Set<Int32>(trainNodes)
    var groupNodes = Set<Int32>(trainNodes)
    groupedNodes.append([Int32](groupNodes))
    while groupNodes.count < nodeCount && groupNodes.count > 0 {
      groupNodes = Set(groupNodes.flatMap { neighbors[Int($0)] })
        .filter { !allNodes.contains($0) }
      let newLevel: [Int32] = [Int32](groupNodes).map {
        ($0, Set(neighbors[Int($0)]).filter { allNodes.contains($0) }.count)
      }.sorted { $0.1 < $1.1 }.map { $0.0 }
      if !newLevel.isEmpty {
        groupedNodes.append(newLevel)
        allNodes = allNodes.union(newLevel)
      }
    }
    if allNodes.count < nodeCount {
      groupedNodes.append((validationNodes + testNodes + otherNodes).filter {
        !allNodes.contains($0)
      })
    }
    self.groupedNodes = groupedNodes
  }

  public var maxNodeDegree: Int { neighbors.map { $0.count }.max()! }

  public var labeledData: LabeledData {
    LabeledData(
      nodeIndices: Tensor<Int32>(trainNodes),
      nodeLabels: Tensor<Int32>(trainNodes.map { Int32(labels[$0]!) }))
  }

  public var nodes: [Int32] { [Int32](0..<Int32(nodeCount)) }
  public var unlabeledNodes: [Int32] { validationNodes + testNodes + otherNodes }

  public var nodesTensor: Tensor<Int32> { Tensor<Int32>(nodes) }
  public var unlabeledNodesTensor: Tensor<Int32> { Tensor<Int32>(unlabeledNodes) }
}

extension Graph {
  public init(loadFromDirectory directory: URL) throws {
    // Load the node features file.
    logger.info("Graph / Loading Features")
    let features = try parse(
      tsvFileAt: directory.appendingPathComponent("features.txt")
    ).map { $0[1].split(separator: " ").map { Float($0)! } }

    // Load the edges file.
    var neighbors = [[Int32]](repeating: [], count: features.count)
    for lineParts in try parse(tsvFileAt: directory.appendingPathComponent("edges.txt")) {
      if lineParts.count < 2 { continue }
      let node1 = Int(lineParts[0])!
      let node2 = Int(lineParts[1])!
      neighbors[node1].append(Int32(node2))
      neighbors[node2].append(Int32(node1))
    }

    // Load the node labels file.
    logger.info("Graph / Loading Labels")
    var trainNodes = [Int32]()
    var validationNodes = [Int32]()
    var testNodes = [Int32]()
    var otherNodes = [Int32]()
    var labels = [Int32: Int]()
    var classCount = 0
    for lineParts in try parse(tsvFileAt: directory.appendingPathComponent("labels.txt")) {
      if lineParts.count < 3 { continue }
      let node = Int32(lineParts[0])!
      let label = Int(lineParts[1])!
      labels[node] = label
      classCount = max(classCount, label + 1)
      switch lineParts[2] {
      case "train": trainNodes.append(node)
      case "val": validationNodes.append(node)
      case "test": testNodes.append(node)
      case "unlabeled": otherNodes.append(node)
      default: ()
      }
    }

    // trainNodes = trainNodes + validationNodes[0..<250]
    // validationNodes = [Int32](validationNodes[250...])

    logger.info("Graph / Initializing")

    // let featureVectors = features.map { f in Tensor<Float>(shape: [f.count], scalars: f) }
    // let featureMoments = Tensor<Float>(
    //   stacking: [Tensor<Float>](featureVectors.enumerated().filter {
    //     (trainNodes + validationNodes).contains(Int32($0.offset))
    //   }.map { $0.element }),
    //   alongAxis: 0).moments(alongAxes: 0)
    // let featuresMean = featureMoments.mean
    // let featuresStd = sqrt(featureMoments.variance).replacing(
    //   with: Tensor<Float>(onesLike: featureMoments.variance),
    //   where: featureMoments.variance .== 0)
    // self.features = (Tensor<Float>(stacking: featureVectors, alongAxis: 0) - featuresMean) / featuresStd

    let featureVectors = features.map { f in Tensor<Float>(shape: [f.count], scalars: f) }

    self.init(
      nodeCount: features.count,
      featureCount: features[0].count,
      classCount: classCount,
      features: Tensor<Float>(stacking: featureVectors, alongAxis: 0),
//      features: Tensor<Float>(
//        stacking: featureVectors.map { f -> Tensor<Float> in
//          let sum = sqrt(f.squared().sum())
//          return f / sum.replacing(with: Tensor<Float>(onesLike: sum), where: sum .== 0)
//        },
//        alongAxis: 0),
      neighbors: neighbors,
      labels: labels,
      trainNodes: trainNodes,
      validationNodes: validationNodes,
      testNodes: testNodes,
      otherNodes: otherNodes)
  }
}

extension Graph {
  public func split<T: RandomNumberGenerator>(
    trainProportion: Float,
    validationProportion: Float,
    using generator: inout T
  ) -> Graph {
    assert(0 < trainProportion + validationProportion && trainProportion + validationProportion < 1)
    let nodes = trainNodes + validationNodes + testNodes + otherNodes
    let shuffledNodes = nodes.shuffled(using: &generator)
    let trainCount = Int(trainProportion * Float(shuffledNodes.count))
    let validationCount = Int(validationProportion * Float(shuffledNodes.count))
    let trainNodes = [Int32](shuffledNodes[0..<trainCount])
    let validationNodes = [Int32](shuffledNodes[trainCount..<(trainCount + validationCount)])
    let testNodes = [Int32](shuffledNodes[(trainCount + validationCount)...])
    return Graph(
      nodeCount: nodeCount,
      featureCount: featureCount,
      classCount: classCount,
      features: features,
      neighbors: neighbors,
      labels: labels,
      trainNodes: trainNodes,
      validationNodes: validationNodes,
      testNodes: testNodes,
      otherNodes: [])
  }

  fileprivate struct Edge: Hashable {
    let source: Int32
    let target: Int32

    public init(_ source: Int32, _ target: Int32) {
      if source < target {
        self.source = source
        self.target = target
      } else {
        self.source = target
        self.target = source
      }
    }
  }

  public var badEdgeProportion: Float {
    var edges = Set<Edge>()
    var badEdgeCount = 0
    for (node, neighbors) in neighbors.enumerated() {
      let node = Int32(node)
      for neighbor in neighbors {
        let edge = Edge(node, neighbor)
        if !edges.contains(edge) && labels[edge.source] != labels[edge.target] {
          badEdgeCount += 1
        }
        edges.insert(edge)
      }
    }
    return Float(badEdgeCount) / Float(edges.count)
   }

  public func corrupted<T: RandomNumberGenerator>(
    targetBadEdgeProportion: Float,
    using generator: inout T
  ) -> Graph {
    var edges = Set<Edge>()
    var badEdgeCount = 0
    for (node, neighbors) in neighbors.enumerated() {
      let node = Int32(node)
      for neighbor in neighbors {
        let edge = Edge(node, neighbor)
        if !edges.contains(edge) && labels[edge.source] != labels[edge.target] {
          badEdgeCount += 1
        }
        edges.insert(edge)
      }
    }
    while Float(badEdgeCount) / Float(edges.count) < targetBadEdgeProportion {
      let source = Int32.random(in: 0..<Int32(nodeCount), using: &generator)
      let target = Int32.random(in: 0..<Int32(nodeCount), using: &generator)
      let edge = Edge(source, target)
      if !edges.contains(edge) && labels[edge.source] != labels[edge.target] {
        edges.insert(edge)
        badEdgeCount += 1
      }
    }
    var neighbors = [[Int32]](repeating: [], count: nodeCount)
    for edge in edges {
      neighbors[Int(edge.source)].append(edge.target)
      neighbors[Int(edge.target)].append(edge.source)
    }
    return Graph(
      nodeCount: nodeCount,
      featureCount: featureCount,
      classCount: classCount,
      features: features,
      neighbors: neighbors,
      labels: labels,
      trainNodes: trainNodes,
      validationNodes: validationNodes,
      testNodes: testNodes,
      otherNodes: otherNodes)
  }
}

public struct SubGraph {
  @usableFromInline internal let graph: Graph
  @usableFromInline internal let mapFromOriginalIndex: [Int: Int]?
  @usableFromInline internal let mapToOriginalIndex: [Int]?

  @inlinable public var nodeCount: Int { graph.nodeCount }
  @inlinable public var featureCount: Int { graph.featureCount }
  @inlinable public var classCount: Int { graph.classCount }
  @inlinable public var features: Tensor<Float> { graph.features }
  @inlinable public var neighbors: [[Int32]] { graph.neighbors }
  @inlinable public var labels: [Int32: Int] { graph.labels }
  @inlinable public var trainNodes: [Int32] { graph.trainNodes }
  @inlinable public var validationNodes: [Int32] { graph.validationNodes }
  @inlinable public var testNodes: [Int32] { graph.testNodes }
  @inlinable public var otherNodes: [Int32] { graph.otherNodes }
  @inlinable public var maxNodeDegree: Int { graph.maxNodeDegree }
  @inlinable public var labeledData: LabeledData { graph.labeledData }
  @inlinable public var nodes: [Int32] { graph.nodes }
  @inlinable public var unlabeledNodes: [Int32] { graph.unlabeledNodes }
  @inlinable public var nodesTensor: Tensor<Int32> { graph.nodesTensor }
  @inlinable public var unlabeledNodesTensor: Tensor<Int32> { graph.unlabeledNodesTensor }

  @usableFromInline
  internal init(graph: Graph, mapFromOriginalIndex: [Int: Int]?) {
    self.graph = graph
    self.mapFromOriginalIndex = mapFromOriginalIndex
    if let mapFromOriginalIndex = mapFromOriginalIndex {
      var mapToOriginalIndex = [Int](repeating: 0, count: graph.nodeCount)
      for (originalIndex, newIndex) in mapFromOriginalIndex {
        mapToOriginalIndex[newIndex] = originalIndex
      }
      self.mapToOriginalIndex = mapToOriginalIndex
    } else {
      self.mapToOriginalIndex = nil
    }
  }

  @inlinable
  public func originalIndex(ofNode node: Int32) -> Int32 {
    if let mapToOriginalIndex = mapToOriginalIndex { return Int32(mapToOriginalIndex[Int(node)]) }
    return node
  }

  @inlinable
  public func transformOriginalSample(_ sample: [Int32]) -> [Int32] {
    if let mapFromOriginalIndex = mapFromOriginalIndex {
      var newSample = [Int32](repeating: 0, count: nodeCount)
      for (originalIndex, newIndex) in mapFromOriginalIndex {
        newSample[newIndex] = sample[originalIndex]
      }
      return newSample
    }
    return sample
  }
}

extension Graph {
  public func subGraph(upToDepth depth: Int) -> SubGraph {
    if depth >= groupedNodes.count { return SubGraph(graph: self, mapFromOriginalIndex: nil) }
    let nodesToKeep = groupedNodes.prefix(depth).flatMap { $0 }
    let nodesToKeepTensor = Tensor<Int32>(nodesToKeep)
    var mapFromOriginalIndex = [Int: Int]()
    mapFromOriginalIndex.reserveCapacity(nodesToKeep.count)
    for (index, node) in nodesToKeep.enumerated() {
      mapFromOriginalIndex[Int(node)] = index
    }
    let graph = Graph(
      nodeCount: nodesToKeep.count,
      featureCount: featureCount,
      classCount: classCount,
      features: features.gathering(atIndices: nodesToKeepTensor),
      neighbors: nodesToKeep.map {
        self.neighbors[Int($0)].compactMap { mapFromOriginalIndex[Int($0)] }.map(Int32.init)
      },
      labels: [Int32: Int](
        uniqueKeysWithValues: [Int32](self.labels.keys)
          .filter { mapFromOriginalIndex.keys.contains(Int($0)) }
          .map { Int32(mapFromOriginalIndex[Int($0)]!) }
          .map { ($0, self.labels[$0]!) }),
      trainNodes: trainNodes.compactMap { mapFromOriginalIndex[Int($0)] }.map(Int32.init),
      validationNodes: validationNodes.compactMap { mapFromOriginalIndex[Int($0)] }.map(Int32.init),
      testNodes: testNodes.compactMap { mapFromOriginalIndex[Int($0)] }.map(Int32.init),
      otherNodes: otherNodes.compactMap { mapFromOriginalIndex[Int($0)] }.map(Int32.init))
    return SubGraph(graph: graph, mapFromOriginalIndex: mapFromOriginalIndex)
  }
}

fileprivate func parse(tsvFileAt fileURL: URL) throws -> [[String]] {
  try Data(contentsOf: fileURL).withUnsafeBytes {
    $0.split(separator: UInt8(ascii: "\n")).map {
      $0.split(separator: UInt8(ascii: "\t"))
        .map { String(decoding: UnsafeRawBufferPointer(rebasing: $0), as: UTF8.self) }
    }
  }
}
