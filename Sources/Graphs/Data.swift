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
  public let neighborIndices: Tensor<Int32>  // [BatchSize, MaxNeighborCount]
  public let neighborMask: Tensor<Float>     // [BatchSize, MaxNeighborCount]

  public init(
    nodeIndices: Tensor<Int32>,
    neighborIndices: Tensor<Int32>,
    neighborMask: Tensor<Float>
  ) {
    self.nodeIndices = nodeIndices
    self.neighborIndices = neighborIndices
    self.neighborMask = neighborMask
  }

  public init<C: RandomAccessCollection>(_handles: C) where C.Element: _AnyTensorHandle {
    precondition(_handles.count == 3)
    let niIndex = _handles.startIndex
    let nniIndex = _handles.index(niIndex, offsetBy: 1)
    let nnmIndex = _handles.index(nniIndex, offsetBy: 1)
    self.nodeIndices = Tensor<Int32>(handle: TensorHandle<Int32>(handle: _handles[niIndex]))
    self.neighborIndices = Tensor<Int32>(handle: TensorHandle<Int32>(handle: _handles[nniIndex]))
    self.neighborMask = Tensor<Float>(handle: TensorHandle<Float>(handle: _handles[nnmIndex]))
  }

  public var _tensorHandles: [_AnyTensorHandle] {
    nodeIndices._tensorHandles + neighborIndices._tensorHandles + neighborMask._tensorHandles
  }
}

public struct Data {
  public let nodeCount: Int
  public let featureCount: Int
  public let classCount: Int
  public let nodeFeatures: [[Float]]
  public let nodeNeighbors: [[Int]]
  public let nodeLabels: [Int: Int]
  public let trainNodes: [Int]
  public let validationNodes: [Int]
  public let testNodes: [Int]

  public let maxBatchNeighborCount: Int = 10

  public var labeledData: LabeledData {
    convertToLabeledData(nodeIndices: trainNodes)
  }

  public var unlabeledData: UnlabeledData {
    convertToUnlabeledData(
      nodeIndices: validationNodes + testNodes,
      maxNeighborCount: maxBatchNeighborCount)
  }

  public var allUnlabeledData: UnlabeledData {
    convertToUnlabeledData(
      nodeIndices: trainNodes + validationNodes + testNodes,
      maxNeighborCount: maxBatchNeighborCount)
  }
  
  public var unlabeledNodeIndices: Tensor<Int32> {
    Tensor<Int32>((validationNodes + testNodes).map(Int32.init))
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
    var trainNodes = [Int]()
    var validationNodes = [Int]()
    var testNodes = [Int]()
    var nodeLabels = [Int: Int]()
    var classCount = 0
    for lineParts in try parse(tsvFileAt: directory.appendingPathComponent("labels.txt")) {
      if lineParts.count < 3 { continue }
      let node = Int(lineParts[0])!
      let label = Int(lineParts[1])!
      nodeLabels[node] = label
      classCount = max(classCount, label + 1)
      switch lineParts[2] {
        case "train": trainNodes.append(node)
        case "val": validationNodes.append(node)
        case "test": testNodes.append(node)
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

  fileprivate func convertToUnlabeledData(
    nodeIndices: [Int],
    maxNeighborCount: Int? = nil
  ) -> UnlabeledData {
    let maxNeighborCount = maxNeighborCount ?? self.maxNeighborCount
    var batchedNodeIndices = [Tensor<Int32>]()
    var batchedNeighborIndices = [Tensor<Int32>]()
    var batchedNeighborMasks = [Tensor<Float>]()
    for nodeIndex in nodeIndices {
      let neighborIndices = nodeNeighbors[nodeIndex]
      var neighborCount = 0
      while neighborCount < neighborIndices.count {
        let t = min(neighborCount + maxNeighborCount, neighborIndices.count)
        let neighborIndicesBatch = neighborIndices[neighborCount..<t]
        let neighborMaskBatch = neighborIndicesBatch.map { _ in Float(1) }
        batchedNodeIndices.append(Tensor<Int32>(Int32(nodeIndex)))
        batchedNeighborIndices.append(
          Tensor<Int32>(neighborIndicesBatch.map(Int32.init))
            .padded(forSizes: [(before: 0, after: maxNeighborCount - neighborIndicesBatch.count)]))
        batchedNeighborMasks.append(
          Tensor<Float>(neighborMaskBatch)
          .padded(forSizes: [(before: 0, after: maxNeighborCount - neighborMaskBatch.count)]))
        neighborCount += maxNeighborCount
      }
    }
    return UnlabeledData(
      nodeIndices: Tensor<Int32>(stacking: batchedNodeIndices, alongAxis: 0),
      neighborIndices: Tensor<Int32>(stacking: batchedNeighborIndices, alongAxis: 0),
      neighborMask: Tensor<Float>(stacking: batchedNeighborMasks, alongAxis: 0))
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
