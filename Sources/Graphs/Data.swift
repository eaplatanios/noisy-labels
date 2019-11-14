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
  public let nodeIndices: Tensor<Int32>  // [BatchSize]
  public let nodeLabels: Tensor<Int32>   // [BatchSize]

  public init(nodeIndices: Tensor<Int32>, nodeLabels: Tensor<Int32>) {
    self.nodeIndices = nodeIndices
    self.nodeLabels = nodeLabels
  }

  public init<C: RandomAccessCollection>(_handles: C) where C.Element: _AnyTensorHandle {
    precondition(_handles.count == 2)
    let niIndex = _handles.startIndex
    let nlIndex = _handles.index(niIndex, offsetBy: 1)
    self.nodeIndices = Tensor<Int32>(handle: TensorHandle<Int32>(handle: _handles[niIndex]))
    self.nodeLabels = Tensor<Int32>(handle: TensorHandle<Int32>(handle: _handles[nlIndex]))
  }

  public var _tensorHandles: [_AnyTensorHandle] {
    nodeIndices._tensorHandles + nodeLabels._tensorHandles
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

  public var labeledData: LabeledData {
    let nodeLabels = trainNodes.map { self.labels[$0]! }
    return LabeledData(
      nodeIndices: Tensor<Int32>(trainNodes),
      nodeLabels: Tensor<Int32>(nodeLabels.map(Int32.init)))
  }

  public var unlabeledData: Tensor<Int32> {
    Tensor<Int32>(validationNodes + testNodes)
  }

  public var allUnlabeledData: Tensor<Int32> {
    Tensor<Int32>(trainNodes + validationNodes + testNodes)
  }
  
  public var unlabeledNodeIndices: Tensor<Int32> {
    Tensor<Int32>(validationNodes + testNodes)
  }

  public var maxBatchNeighborCount: Int { maxNeighborCount }

  public var maxNeighborCount: Int { neighbors.map { $0.count }.max()! }
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
        default: ()
      }
    }

    logger.info("Graph / Initializing")
    self.nodeCount = features.count
    self.featureCount = features[0].count
    self.classCount = classCount
    self.features = Tensor<Float>(
      stacking: features.map(Tensor<Float>.init),
      alongAxis: 0)
    self.neighbors = neighbors
    self.labels = labels
    self.trainNodes = trainNodes
    self.validationNodes = validationNodes
    self.testNodes = testNodes
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
