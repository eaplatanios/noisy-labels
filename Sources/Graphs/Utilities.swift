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

import Logging
import TensorFlow

internal let logger = Logger(label: "Graphs")

/// Returns the log-softmax of the specified tensor element-wise.
@inlinable
@differentiable
public func logSoftmax<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>,
  alongAxis axis: Int
) -> Tensor<T> {
  x - x.logSumExp(alongAxes: Tensor<Int32>(Int32(axis)))
}

public extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @differentiable(wrt: self)
  func logSumExp(alongAxes axes: Tensor<Int32>) -> Tensor {
    let rawMax = max(alongAxes: axes)
    let offset = withoutDerivative(at: rawMax) { rawMax in 
      rawMax.replacing(
        with: Tensor<Scalar>(zerosLike: rawMax),
        where: rawMax.isInfinite)
    }
    let result = TensorFlow.log(TensorFlow.exp(self - offset).sum(alongAxes: axes))
    return result + offset
  }

  @inlinable
  @differentiable(wrt: self)
  func logSumExp(alongAxes axes: [Int]) -> Tensor {
    // TODO(TF-433): Remove workaround for differentiating `map`.
    let axes = withoutDerivative(at: axes) { $0.map(Int32.init) }
    return logSumExp(alongAxes: Tensor<Int32>(axes))
  }

  @inlinable
  @differentiable(wrt: self)
  func logSumExp(alongAxes axes: Int...) -> Tensor {
    logSumExp(alongAxes: axes)
  }
}

// extension Dataset {
//   @inlinable
//   internal func batched(
//     _ batchSize: Int,
//     paddedShapes: [Tensor<Int64>],
//     paddingValues: Element
//   ) -> Dataset {
//     Dataset(_handle: _Raw.paddedBatchDataset(
//       inputDataset: _handle,
//       batchSize: Tensor(Int64(batchSize)),
//       paddedShapes: paddedShapes,
//       paddingValues: paddingValues,
//       outputShapes: Element._unknownShapeList))
//   }
// }
//
// extension Model {
//   internal var batchPaddedShapes: [Tensor<Int64>] {
//     [[], [-1], [-1], [-1, -1], [-1]]
//   }

//   internal var batchPaddingValues: UnlabeledData {
//     UnlabeledData(
//       nodeIndices: Tensor<Int32>(0),
//       nodeFeatures: Tensor<Float>(0),
//       neighborIndices: Tensor<Int32>(0),
//       neighborFeatures: Tensor<Float>(0),
//       neighborMask: Tensor<Float>(0))
//   }
// }
