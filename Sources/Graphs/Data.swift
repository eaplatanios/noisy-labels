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

  public var maxNodeDegree: Int { neighbors.map { $0.count }.max()! }

  public var labeledData: LabeledData {
    let nodeLabels = trainNodes.map { self.labels[$0]! }
    return LabeledData(
      nodeIndices: Tensor<Int32>(trainNodes),
      nodeLabels: Tensor<Int32>(nodeLabels.map(Int32.init)))
  }

  public var nodes: [Int32] { [Int32](0..<Int32(nodeCount)) }
  public var unlabeledNodes: [Int32] { validationNodes + testNodes + otherNodes }

  public var nodesTensor: Tensor<Int32> { Tensor<Int32>(nodes) }
  public var unlabeledNodesTensor: Tensor<Int32> { Tensor<Int32>(unlabeledNodes) }

  public var unlabeledNodeIndices: Tensor<Int32> {
    Tensor<Int32>(validationNodes + testNodes + otherNodes)
  }

  public var maxBatchNeighborCount: Int { maxNeighborCount }

  public var maxNeighborCount: Int { neighbors.map { $0.count }.max()! }

  public var leveledData: [[Int32]] {
    var leveledData = [[Int32]]()
    var allNodes = Set<Int32>(trainNodes)
    var nodes = Set<Int32>(trainNodes)
    leveledData.append([Int32](nodes))
    while nodes.count < nodeCount && nodes.count > 0 {
      nodes = Set(nodes.flatMap { Set(neighbors[Int($0)]) }).filter { !allNodes.contains($0) }
      let newLevel: [Int32] = [Int32](nodes).map {
        ($0, Set(neighbors[Int($0)]).filter { allNodes.contains($0) }.count)
      }.sorted { $0.1 < $1.1 }.map { $0.0 }
      if !newLevel.isEmpty {
        leveledData.append(newLevel)
        allNodes = allNodes.union(newLevel)
      }
    }
    if allNodes.count < nodeCount {
      leveledData.append((validationNodes + testNodes + otherNodes).filter {
        !allNodes.contains($0)
      })
    }
    return leveledData
  }

  public func data(atDepth depth: Int) -> Tensor<Int32> {
    let nodes = leveledData.prefix(depth).flatMap { $0 }
    print("Training on \(nodes.count) / \(nodeCount) nodes.")
    return Tensor<Int32>(nodes)
  }
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
    var unlabeledNodes = [Int32]()
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
        case "unlabeled": unlabeledNodes.append(node)
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
    // self.features = Tensor<Float>(stacking: featureVectors, alongAxis: 0)
    self.features = Tensor<Float>(
      stacking: featureVectors.map { f -> Tensor<Float> in
        let sum = sqrt(f.squared().sum())
        return f / sum.replacing(with: Tensor<Float>(onesLike: sum), where: sum .== 0)
      },
      alongAxis: 0)

    self.nodeCount = features.count
    self.featureCount = features[0].count
    self.classCount = classCount
    // let ff: [Tensor<Float>] = features.map { f in Tensor<Float>(shape: [f.count], scalars: f) }
    // self.features = Tensor<Float>(
    //   stacking: ff,
    //   // stacking: ff.map { f -> Tensor<Float> in
    //   //   let sum = f.sum()
    //   //   return f / sum.replacing(with: Tensor<Float>(onesLike: sum), where: sum .== 0)
    //   // },
    //   alongAxis: 0)
    self.neighbors = neighbors
    self.labels = labels
    self.trainNodes = trainNodes
    self.validationNodes = validationNodes
    self.testNodes = testNodes
    self.otherNodes = unlabeledNodes
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

fileprivate func parse(tsvFileAt fileURL: URL) throws -> [[String]] {
  try Data(contentsOf: fileURL).withUnsafeBytes {
    $0.split(separator: UInt8(ascii: "\n")).map {
      $0.split(separator: UInt8(ascii: "\t"))
        .map { String(decoding: UnsafeRawBufferPointer(rebasing: $0), as: UTF8.self) }
    }
  }
}
