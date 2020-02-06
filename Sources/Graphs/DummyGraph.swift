import TensorFlow

func makeSimpleGraph(_ firstComponentSize: Int, _ secondComponentSize: Int) -> Graph {
  let nodeCount = firstComponentSize + secondComponentSize
  let features = Tensor<Float>(
    stacking: (0..<firstComponentSize).map { _ in Tensor<Float>([1, 0]) } +
      (0..<secondComponentSize).map { _ in Tensor<Float>([0, 1]) },
    alongAxis: 0)
  let firstNeighbors = [Int](0..<firstComponentSize).map { index -> [Int32] in
    let first = Int32((index - 1 + firstComponentSize) % firstComponentSize)
    let second = Int32((index + 1) % firstComponentSize)
    return [first, second]
  }
  let secondNeighbors = [Int](0..<secondComponentSize).map { index -> [Int32] in
    let first = Int32((index - 1 + secondComponentSize) % secondComponentSize + firstComponentSize)
    let second = Int32((index + 1) % secondComponentSize + firstComponentSize)
    return [first, second]
  }
  let neighbors = [[Int32]](firstNeighbors + secondNeighbors)
  let labels = [Int](repeating: 0, count: firstComponentSize) +
    [Int]((0..<secondComponentSize).map { $0 % 2 })
  let validationNodes = [Int32]([Int32(firstComponentSize + 0), Int32(firstComponentSize + 10)])
  let testNodes = [Int32]([Int32(firstComponentSize + 5), Int32(firstComponentSize + 15)])
  let trainNodes = [Int32](0..<Int32(firstComponentSize + secondComponentSize)).filter {
    !validationNodes.contains($0) && !testNodes.contains($0)
  }
  let otherNodes = [Int32]()
  return Graph(
    nodeCount: nodeCount,
    featureCount: 2,
    classCount: 2,
    features: features,
    neighbors: neighbors,
    labels: [Int32: Int](
      uniqueKeysWithValues: labels.enumerated().map { (Int32($0.0), $0.1) }),
    trainNodes: trainNodes,
    validationNodes: validationNodes,
    testNodes: testNodes,
    otherNodes: otherNodes)

}
