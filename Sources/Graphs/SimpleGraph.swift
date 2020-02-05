func makeSimpleGraph(_ firstComponentSize: Int, _ secondComponentSize: Int) -> Graph {
  return Graph(
    nodeCount: firstComponentSize + secondComponentSize,
    featureCount: 1,
    classCount: 2,
    features: Tensor<Float>(
      stacking: (0..<firstComponentSize).map { _ in Tensor<Float>([1, 0]) } + (0..<secondComponentSize).map { _ in Tensor<Float>([0, 1]) }, alongAxis: 0)),
    neighbors:
      (0..<firstComponentSize).map { [($0 - 1 + firstComponentSize) % firstComponentSize, ($0 + 1) % firstComponentSize] } +
      (0..<secondComponentSize).map { [($0 - 1 + secondComponentSize) % secondComponentSize + firstComponentSize, ($0 + 1) % secondComponentSize + firstComponentSize] },
    labels:
      [Int32](repeating: 0, count: firstComponentSize) + (0..<secondComponentSize).map { Int32(%0 % 2) },
    trainNodes: [Int32](0..<Int32(firstComponentSize + secondComponentSize)),
    validationNodes: [Int32](0..<Int32(firstComponentSize + secondComponentSize)),
    testNodes: [Int32](0..<Int32(firstComponentSize + secondComponentSize)),
    unlabeledNodes: [Int32](0..<Int32(firstComponentSize + secondComponentSize))
}
