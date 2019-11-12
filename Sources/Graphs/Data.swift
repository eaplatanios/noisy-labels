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

public struct UnlabeledData: TensorGroup {
  public let nodeIndices: Tensor<Int32>      // [BatchSize]
  public let nodeFeatures: Tensor<Float>     // [BatchSize, FeatureCount]
  public let neighborIndices: Tensor<Int32>  // [BatchSize, MaxNeighborCount]
  public let neighborFeatures: Tensor<Float> // [BatchSize, MaxNeighborCount, FeatureCount]
  public let neighborMask: Tensor<Float>     // [BatchSize, MaxNeighborCount]

  public init(
    nodeIndices: Tensor<Int32>,
    nodeFeatures: Tensor<Float>,
    neighborIndices: Tensor<Int32>,
    neighborFeatures: Tensor<Float>,
    neighborMask: Tensor<Float>
  ) {
    self.nodeIndices = nodeIndices
    self.nodeFeatures = nodeFeatures
    self.neighborIndices = neighborIndices
    self.neighborFeatures = neighborFeatures
    self.neighborMask = neighborMask
  }

  public init<C: RandomAccessCollection>(_handles: C) where C.Element: _AnyTensorHandle {
    precondition(_handles.count == 5)
    let niIndex = _handles.startIndex
    let nfIndex = _handles.index(niIndex, offsetBy: 1)
    let nniIndex = _handles.index(nfIndex, offsetBy: 1)
    let nnfIndex = _handles.index(nniIndex, offsetBy: 1)
    let nnmIndex = _handles.index(nnfIndex, offsetBy: 1)
    self.nodeIndices = Tensor<Int32>(handle: TensorHandle<Int32>(handle: _handles[niIndex]))
    self.nodeFeatures = Tensor<Float>(handle: TensorHandle<Float>(handle: _handles[nfIndex]))
    self.neighborIndices = Tensor<Int32>(handle: TensorHandle<Int32>(handle: _handles[nniIndex]))
    self.neighborFeatures = Tensor<Float>(handle: TensorHandle<Float>(handle: _handles[nnfIndex]))
    self.neighborMask = Tensor<Float>(handle: TensorHandle<Float>(handle: _handles[nnmIndex]))
  }

  public var _tensorHandles: [_AnyTensorHandle] {
    var handles = nodeIndices._tensorHandles + nodeFeatures._tensorHandles
    handles += neighborIndices._tensorHandles + neighborFeatures._tensorHandles
    return handles + neighborMask._tensorHandles
  }
}

public struct Data {
  public let nodeCount: Int
  public let featureCount: Int
  public let classCount: Int
  public let nodeFeatures: [[Float]]
  public let nodeNeighbors: [[Int]]
  public let nodeLabels: [Int: Int]
  public let trainNodes: Set<Int>
  public let validationNodes: Set<Int>
  public let testNodes: Set<Int>

  public var labeledData: LabeledData {
    convertToLabeledData(nodeIndices: Array(trainNodes))
  }

  public var unlabeledData: UnlabeledData {
    convertToUnlabeledData(nodeIndices: Array(validationNodes.union(testNodes)))
  }

  public var allUnlabeledData: UnlabeledData {
    convertToUnlabeledData(nodeIndices: Array(trainNodes.union(validationNodes).union(testNodes)))
  }

  public var maxNeighborCount: Int { nodeNeighbors.map { $0.count }.max()! }
}

extension Data {
  public init(loadFromDirectory directory: URL) throws {
    // Load the node features file.
    logger.info("Data / Loading Features")
    let nodeFeatures = try parse(
      tsvFileAt: directory.appendingPathComponent("features.txt")
    ).map { $0[1].split(separator: " ").map { Float($0)! } }

    // Load the edges file.
    var nodeNeighbors = [[Int]](repeating: [], count: nodeFeatures.count)
    for lineParts in try parse(tsvFileAt: directory.appendingPathComponent("edges.txt")) {
      if lineParts.count < 2 { continue }
      let node1 = Int(lineParts[0])!
      let node2 = Int(lineParts[1])!
      nodeNeighbors[node1].append(node2)
      nodeNeighbors[node2].append(node1)
    }

    // Load the node labels file.
    logger.info("Data / Loading Labels")
    var trainNodes = Set<Int>()
    var validationNodes = Set<Int>()
    var testNodes = Set<Int>()
    var nodeLabels = [Int: Int]()
    var classCount = 0
    for lineParts in try parse(tsvFileAt: directory.appendingPathComponent("labels.txt")) {
      if lineParts.count < 3 { continue }
      let node = Int(lineParts[0])!
      let label = Int(lineParts[1])!
      nodeLabels[node] = label
      classCount = max(classCount, label + 1)
      switch lineParts[2] {
        case "train": trainNodes.update(with: node)
        case "val": validationNodes.update(with: node)
        case "test": testNodes.update(with: node)
        default: ()
      }
    }

    logger.info("Data / Initializing")
    self.nodeCount = nodeFeatures.count
    self.featureCount = nodeFeatures[0].count
    self.classCount = classCount
    self.nodeFeatures = nodeFeatures
    self.nodeNeighbors = nodeNeighbors
    self.nodeLabels = nodeLabels
    self.trainNodes = trainNodes
    self.validationNodes = validationNodes
    self.testNodes = testNodes
  }
}

extension Data {
  fileprivate func convertToLabeledData(nodeIndices: [Int]) -> LabeledData {
    let nodeLabels = nodeIndices.map { self.nodeLabels[$0]! }
    return LabeledData(
      nodeIndices: Tensor<Int32>(nodeIndices.map(Int32.init)),
      nodeLabels: Tensor<Int32>(nodeLabels.map(Int32.init)))
  }

  fileprivate func convertToUnlabeledData(nodeIndices: [Int]) -> UnlabeledData {
    let nodeFeatures = nodeIndices.map { self.nodeFeatures[$0] }
    let neighborIndices = nodeIndices.map { nodeNeighbors[$0] }
    let neighborFeatures = neighborIndices.map { $0.map { self.nodeFeatures[$0] } }
    let neighborMask = neighborIndices.map { $0.map { _ in Float(1) } }
    return UnlabeledData(
      nodeIndices: Tensor<Int32>(nodeIndices.map(Int32.init)),
      nodeFeatures: Tensor<Float>(
        stacking: nodeFeatures.map(Tensor<Float>.init),
        alongAxis: 0),
      neighborIndices: Tensor<Int32>(
        stacking: neighborIndices.map {
          Tensor<Int32>($0.map(Int32.init)).padded(
            forSizes: [(before: 0, after: maxNeighborCount - $0.count)])
        },
        alongAxis: 0),
      neighborFeatures: Tensor<Float>(
        stacking: neighborFeatures.map {
          Tensor<Float>(
            stacking: $0.map(Tensor<Float>.init),
            alongAxis: 0
          ).padded(forSizes: [
            (before: 0, after: maxNeighborCount - $0.count),
            (before: 0, after: 0)])
        },
        alongAxis: 0),
      neighborMask: Tensor<Float>(
        stacking: neighborMask.map {
          Tensor<Float>($0).padded(forSizes: [(before: 0, after: maxNeighborCount - $0.count)])
        },
        alongAxis: 0))
  }
}

fileprivate func parse(tsvFileAt fileURL: URL) throws -> [[String]] {
  try Foundation.Data(contentsOf: fileURL).withUnsafeBytes {
    $0.split(separator: UInt8(ascii: "\n")).map {
      $0.split(separator: UInt8(ascii: "\t"))
        .map { String(decoding: UnsafeRawBufferPointer(rebasing: $0), as: UTF8.self) }
    }
  }
}
