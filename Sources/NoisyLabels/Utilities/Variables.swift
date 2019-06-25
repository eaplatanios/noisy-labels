import TensorFlow

@inlinable @inline(__always)
internal func createVariable(
  _ name: String,
  dataType: TensorDataType,
  shape: TensorShape
) -> ResourceHandle {
  return Raw.varHandleOp(
    container: "",
    sharedName: name,
    dtype: dataType,
    shape: shape)
}

@inlinable @inline(__always)
internal func assignVariable<Scalar: TensorFlowScalar>(
  _ variable: ResourceHandle,
  _ value: Tensor<Scalar>
) {
  Raw.assignVariableOp(resource: variable, value: value)
}

@inlinable @inline(__always)
internal func readVariable<Scalar: TensorFlowScalar>(
  _ variable: ResourceHandle
) -> Tensor<Scalar> {
  return Raw.readVariableOp(resource: variable)
}

@inlinable @inline(__always)
internal func gatherVariable<Scalar: TensorFlowScalar, I: TensorFlowInteger>(
  _ variable: ResourceHandle,
  _ indices: Tensor<I>
) -> Tensor<Scalar> {
  return Raw.resourceGatherNd(resource: variable, indices: indices)
}

@inlinable @inline(__always)
internal func scatterAddVariable<Scalar: TensorFlowNumeric, I: TensorFlowInteger>(
  _ variable: ResourceHandle,
  _ indices: Tensor<I>,
  _ updates: Tensor<Scalar>
) {
  Raw.resourceScatterNdAdd(ref: variable, indices: indices, updates: updates)
}
