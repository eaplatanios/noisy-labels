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

internal extension Tensor {
  @inlinable
  @_semantics("autodiff.nonvarying")
  func withoutDerivative() -> Tensor { return self }
}

public extension Tensor where Scalar: TensorFlowFloatingPoint {
  /// Returns a boolean tensor indicating which elements of `x` are finite.
  @inlinable var isFinite: Tensor<Bool> { Raw.isFinite(self) }

  /// Returns a boolean tensor indicating which elements of `x` are infinite.
  @inlinable var isInfinite: Tensor<Bool> { Raw.isInf(self) }

  /// Returns a boolean tensor indicating which elements of `x` are NaN-valued.
  @inlinable var isNaN: Tensor<Bool> { Raw.isNan(self) }
}

/// Returns the logarithm of `1 + x` element-wise.
@inlinable
@differentiable(vjp: _vjpLog1p)
public func log1p<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Raw.log1p(x)
}

@inlinable
func _vjpLog1p<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
  (log1p(x), { v in Raw.xdivy(v, 1 + x) })
}

/// Computes `log(1 - exp(x))` using a numerically stable approach.
///
/// The approach is shown in Equation 7 of:
/// https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf.
@inlinable
@differentiable
public func log1mexp<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  // let isTooSmall = -x .< T(log(2.0))
  // // This `replacing` will ultimately be a no-op because we will not select this code-path 
  // // whenever we use the surrogate `-Tensor(onesLike: x)`.
  // let ones = Tensor(onesLike: xSafe)
  // let xSafe = x.replacing(with: -ones, where: isTooSmall)
  // return log1p(-exp(xSafe)).replacing(with: log(-expm1(x)), where: isTooSmall)
  return log(1 - exp(x))
}

/// Returns the log-sigmoid of the specified tensor element-wise. Specifically,
/// `y = log(1 / (1 + exp(-x)))`. For numerical stability, we use `y = -softplus(-x)`.
@inlinable
@differentiable
public func logSigmoid<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  -softplus(-x)
}

/// Returns the softplus of the specified tensor element-wise.
/// Specifically, computes `log(exp(features) + 1)`.
@inlinable
@differentiable(vjp: _vjpSoftplus)
public func softplus<T: TensorFlowFloatingPoint>(_ features: Tensor<T>) -> Tensor<T> {
  Raw.softplus(features: features)
}

@inlinable
internal func _vjpSoftplus<T: TensorFlowFloatingPoint>(
  _ features: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
  (softplus(features), { v in Raw.softplusGrad(gradients: v, features: features)})
}

public extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @differentiable(wrt: self)
  func logSumExp(squeezingAxes axes: Tensor<Int32>) -> Tensor {
    let rawMax = max(alongAxes: axes)
    let offset = rawMax.replacing(
      with: Tensor<Scalar>(zerosLike: rawMax),
      where: rawMax.isFinite
    ).withoutDerivative()
    let result = TensorFlow.log(TensorFlow.exp(self - offset).sum(squeezingAxes: axes))
    return result + offset.reshaped(toShape: result.shapeTensor.withoutDerivative())
  }

  @inlinable
  @differentiable(wrt: self)
  func logSumExp(squeezingAxes axes: [Int]) -> Tensor {
    // TODO(TF-433): Remove workaround for differentiating `map`.
    let axes = {axes.map(Int32.init)}()
    return logSumExp(squeezingAxes: Tensor<Int32>(axes))
  }

  @inlinable
  @differentiable(wrt: self)
  func logSumExp(squeezingAxes axes: Int...) -> Tensor {
    return logSumExp(squeezingAxes: axes)
  }

  @inlinable
  @differentiable(wrt: self)
  func logSumExp() -> Tensor {
    return flattened().logSumExp(squeezingAxes: 0)
  }

  @inlinable
  @differentiable(wrt: self)
  func logSumExp(alongAxes axes: Tensor<Int32>) -> Tensor {
    let offset = max(alongAxes: axes)
    // TODO:
    // let offset = rawMax.replacing(
    //   with: Tensor<Scalar>(zerosLike: rawMax), where: isFinite(rawMax))
    let result = TensorFlow.log(TensorFlow.exp(self - offset).sum(alongAxes: axes))
    return result + offset
  }

  @inlinable
  @differentiable(wrt: self)
  func logSumExp(alongAxes axes: [Int]) -> Tensor {
    // TODO(TF-433): Remove workaround for differentiating `map`.
    let axes = {axes.map(Int32.init)}()
    return logSumExp(alongAxes: Tensor<Int32>(axes))
  }

  @inlinable
  @differentiable(wrt: self)
  func logSumExp(alongAxes axes: Int...) -> Tensor {
    return logSumExp(alongAxes: axes)
  }
}

//===------------------------------------------------------------------------------------------===//
// Basic
//===------------------------------------------------------------------------------------------===//

public extension Tensor {
  // /// Gathers slices of this tensor at `indices` along the `axis` dimension, while ignoring the
  // /// first `batchDims` dimensions that correspond to batch dimensions.
  // ///
  // /// Performs similar functionality to `gathering`, except that the resulting tensor shape is now:
  // /// `self.shape[..<axis] + indices.shape[batchDims...] + self.shape[(axis + 1)...]`.
  // ///
  // /// - Parameters:
  // ///   - indices: Contains the indices to gather.
  // ///   - axis: Dimension along which to gather. Negative values wrap around.
  // ///   - batchDims: Number of leading batch dimensions to ignore.
  // ///
  // /// - Precondition: `axis` must be in the range `[-rank, rank)`, while also being greater than
  // ///     or equal to `batchDims`.
  // /// - Precondition: `batchDims` must be less than `indices.rank`.
  // ///
  // /// - Returns: The gathered tensor.
  // @inlinable
  // @differentiable(wrt: self where Scalar : TensorFlowFloatingPoint)
  // func batchGathering(
  //   atIndices indices: Tensor<Int32>,
  //   alongAxis axis: Int,
  //   numBatchDims batchDims: Int
  // ) -> Tensor {
  //   precondition(batchDims >= 0 && batchDims < indices.rank,
  //     "'numBatchDims' must be non-negative and less than 'indices.rank'.")
  //   precondition(batchDims < rank, "'numBatchDims' must be less than the tensor's rank.")

  //   // Handle the axis argument by transposing the axis dimension so that it is the first
  //   // non-batch dimension, recursively calling `batchGathering` with `axis = 0`, and then
  //   // transposing the result to put the pre-axis dimensions before the indices dimensions.
  //   if axis != batchDims {
  //     // Adjust axis to be positive.
  //     let posAxis = axis < 0 ? axis + rank : axis

  //     precondition(posAxis >= 0 && posAxis < rank, "'axis' is out of range.")
  //     precondition(batchDims <= posAxis, "'batchDims' must be less than or equal to 'axis'.")

  //     // Move self[axis] up to self[batchDims].
  //     let permutation = Tensor<Int32>(concatenating: [
  //       Tensor<Int32>(rangeFrom: 0, to: Int32(batchDims), stride: 1),
  //       Tensor<Int32>(Int32(axis)).rankLifted(),
  //       Tensor<Int32>(rangeFrom: Int32(batchDims), to: Int32(posAxis), stride: 1),
  //       Tensor<Int32>(rangeFrom: Int32(axis) + 1, to: Int32(rank), stride: 1)])
  //     let tensor = transposed(withPermutations: permutation)
  //     let result = tensor.batchGathering(
  //       atIndices: indices, alongAxis: batchDims, numBatchDims: batchDims)

  //     // Move the result dimensions corresponding to self[batchDims ..< axis] to just before
  //     // the dimensions corresponding to indices[batchDims ...].
  //     let start = indices.rank + posAxis - batchDims
  //     let resultPermutation = Tensor<Int32>(concatenating: [
  //       Tensor<Int32>(rangeFrom: 0, to: Int32(batchDims), stride: 1),
  //       Tensor<Int32>(rangeFrom: Int32(indices.rank), to: Int32(start), stride: 1),
  //       Tensor<Int32>(rangeFrom: Int32(batchDims), to: Int32(indices.rank), stride: 1),
  //       Tensor<Int32>(rangeFrom: Int32(start), to: Int32(result.rank), stride: 1)])
  //     return result.transposed(withPermutations: resultPermutation)
  //   }

  //   var batchIndices = indices
  //   var accumulated = Tensor<Int32>(ones: [])
  //   for d in (1...batchDims).reversed() {
  //     accumulated *= shapeTensor[d]
  //     let dValue = shapeTensor[d - 1]
  //     let dIndices = Tensor<Int32>(
  //       rangeFrom: Tensor<Int32>(zeros: []),
  //       to: dValue,
  //       stride: Tensor<Int32>(ones: [])
  //     ) * accumulated
  //     let dShape = Tensor<Int32>(concatenating: [
  //       Tensor<Int32>([Int32](repeating: 1, count: d - 1)),
  //       dValue.rankLifted(),
  //       Tensor<Int32>([Int32](repeating: 1, count: indices.rank - 1))])
  //     batchIndices += dIndices.reshaped(toShape: dShape)
  //   }

  //   let flatIndices = batchIndices.flattened()
  //   let outerShape = shapeTensor[Int(batchDims + 1)...]
  //   let innerShape = shapeTensor[..<Int(batchDims + 1)].product(squeezingAxes: [0])
  //   let flatTensor = reshaped(toShape: innerShape.rankLifted().concatenated(with: outerShape))
  //   let flatResult = flatTensor.gathering(atIndices: flatIndices)
  //   return flatResult.reshaped(toShape: indices.shapeTensor.concatenated(with: outerShape))
  // }

  /// Returns a tensor by gathering slices of the input at `indices` along the `axis` dimension
  ///
  /// For 0-D (scalar) `indices`:
  /// ```
  /// result[p_0,          ..., p_{axis-1},
  ///        p_{axis + 1}, ..., p_{N-1}] =
  /// self[p_0,          ..., p_{axis-1},
  ///      indices,
  ///      p_{axis + 1}, ..., p_{N-1}]
  /// ```
  ///
  /// For 1-D (vector) `indices`:
  /// ```
  /// result[p_0,          ..., p_{axis-1},
  ///        i,
  ///        p_{axis + 1}, ..., p_{N-1}] =
  /// self[p_0,          ..., p_{axis-1},
  ///      indices[i],
  ///      p_{axis + 1}, ..., p_{N-1}]
  /// ```
  ///
  /// In the general case, produces a resulting tensor where:
  /// ```
  /// result[p_0,             ..., p_{axis-1},
  ///        i_{batch\_dims}, ..., i_{M-1},
  ///        p_{axis + 1},    ..., p_{N-1}] =
  /// self[p_0,             ..., p_{axis-1},
  ///      indices[i_0,     ..., i_{M-1}],
  ///      p_{axis + 1},    ..., p_{N-1}]
  /// ```
  /// where `N = self.rank` and `M = indices.rank`.
  ///
  /// The shape of the resulting tensor is:
  /// `self.shape[..<axis] + indices.shape + self.shape[(axis + 1)...]`.
  ///
  /// - Note: On CPU, if an out-of-range index is found, an error is thrown. On GPU, if an
  /// out-of-range index is found, a 0 is stored in the corresponding output values.
  ///
  /// - Parameters:
  ///   - indices: Contains the indices to gather at.
  ///   - axis: Dimension along which to gather. Negative values wrap around.
  ///
  /// - Precondition: `axis` must be in the range `[-rank, rank)`.
  ///
  /// - Returns: The gathered tensor.
  @inlinable
  @differentiable(wrt: self, vjp: _vjpGathering where Scalar : TensorFlowFloatingPoint)
  func gathering(atIndices indices: Tensor<Int32>, alongAxis axis: Int = 0) -> Tensor {
    return Raw.gatherV2(params: self, indices: indices, axis: Tensor<Int32>(Int32(axis)))
  }

  /// Returns slices of this tensor at `indices`, while ignoring the first `batchDims` dimensions
  /// that correspond to batch dimensions. The gather is performed along the first non-batch
  /// dimension.
  ///
  /// Performs similar functionality to `gathering`, except that the resulting tensor shape is 
  /// now:
  /// ```
  /// self.shape[..<batchDims] + 
  ///   indices.shape[batchDims...] + 
  ///   self.shape[(batchDims + indices.rank + 1)...]
  /// ```
  ///
  /// - Parameters:
  ///   - indices: Contains the indices to gather.
  ///   - batchDims: Number of leading batch dimensions to ignore.
  ///
  /// - Precondition: `batchDims` must be less than `indices.rank`.
  ///
  /// - Returns: The gathered tensor.
  @inlinable
  @differentiable(wrt: self where Scalar : TensorFlowFloatingPoint)
  func batchGathering(atIndices indices: Tensor<Int32>) -> Tensor {
    var batchIndices = indices
    var accumulated = Tensor<Int32>(ones: [])
    accumulated *= Swift.withoutDerivative(at: shapeTensor) { $0[1] }
    let dValue = Swift.withoutDerivative(at: shapeTensor) { $0[0] }
    let dIndices = Tensor<Int32>(
      rangeFrom: Tensor<Int32>(zeros: []),
      to: dValue,
      stride: Tensor<Int32>(ones: [])
    ) * accumulated
    let dShape = Tensor<Int32>(concatenating: [
      dValue.rankLifted(),
      Tensor<Int32>([Int32](repeating: 1, count: indices.rank - 1))])
    batchIndices += dIndices.reshaped(toShape: dShape)
    let flatIndices = batchIndices.flattened()
    let outerShape = Swift.withoutDerivative(at: shapeTensor) { $0[2...] }
    let innerShape = Swift.withoutDerivative(at: shapeTensor) { $0[..<2] }.product(squeezingAxes: [0])
    let flatTensor = reshaped(toShape: innerShape.rankLifted().concatenated(with: outerShape))
    let flatResult = flatTensor.gathering(atIndices: flatIndices)
    return flatResult.reshaped(toShape: indices.shapeTensor.concatenated(with: outerShape))
  }

  /// Returns a tensor by gathering the values after applying the provided boolean mask to the input.
  ///
  /// For example:
  /// ```
  /// // 1-D example
  /// // tensor is [0, 1, 2, 3]
  /// // mask is [true, false, true, false]
  /// tensor.gathering(where: mask) // is [0, 2]
  ///
  /// // 2-D example
  /// // tensor is [[1, 2], [3, 4], [5, 6]]
  /// // mask is [true, false, true]
  /// tensor.gathering(where: mask) // is [[1, 2], [5, 6]]
  /// ```
  ///
  /// In general, `0 < mask.rank = K <= tensor.rank`, and the `mask`'s shape must match the first
  /// K dimensions of the `tensor`'s shape. We then have:
  /// `tensor.gathering(where: mask)[i, j1, ..., jd] = tensor[i1, ..., iK, j1, ..., jd]`, where
  /// `[i1, ..., iK]` is the `i`th `true` entry of `mask` (row-major order).
  ///
  /// The `axis` could be used with `mask` to indicate the axis to mask from. In that case,
  /// `axis + mask.rank <= tensor.rank` and the `mask``'s shape must match the first
  /// `axis + mask.rank` dimensions of the `tensor`'s shape.
  ///
  /// - Parameters:
  ///   - mask: K-D boolean tensor, where `K <= self.rank`.
  ///   - axis: 0-D integer tensor representing the axis in `self` to mask from, where
  ///     `K + axis <= self.rank`.
  ///
  /// - Precondition: The `mask` cannot be a scalar: `mask.rank != 0`.
  ///
  /// - Returns: `(self.rank - K + 1)`-dimensional tensor populated by entries in this tensor
  ///   corresponding to `true` values in `mask`.
  @inlinable
  // @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  func gathering(where mask: Tensor<Bool>, alongAxis axis: Int = 0) -> Tensor {
    precondition(mask.rank != 0, "The boolean mask cannot be a scalar.")
    // TODO: Remove once control flow AD is supported.
    let rank = self.rank
    let posAxis = { axis < 0 ? axis + rank : axis }()
    let leadingSize = shapeTensor[posAxis ..< posAxis + mask.rank].product().rankLifted()
    let reshapedTensor = reshaped(
      toShape: Tensor<Int32>(concatenating: [
        shapeTensor[..<posAxis],
        leadingSize,
        shapeTensor[(posAxis + mask.rank)...]]))
    let indices = Tensor<Int32>(mask.flattened().nonZeroIndices().squeezingShape(at: 1))
    return reshapedTensor.gathering(atIndices: indices, alongAxis: posAxis)
  }
}
