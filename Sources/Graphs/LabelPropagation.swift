
import TensorFlow

public func propagateLabels(inGraph graph: Graph) -> [Int32] {
  var labelProbabilities = Tensor<Float>(zeros: [graph.nodeCount, graph.classCount])
  var labeledNodes = Set<Int32>()
  var nodesToLabel = Set<Int32>()

  // Start with the labeled nodes.
  for node in graph.trainNodes {
    let label = graph.labels[node]!
    labelProbabilities = _Raw.tensorScatterAdd(
      labelProbabilities,
      indices: Tensor<Int32>([node]).expandingShape(at: -1),
      updates: Tensor<Float>(
        oneHotAtIndices: Tensor<Int32>(Int32(label)),
        depth: graph.classCount).expandingShape(at: 0))
    labeledNodes.update(with: node)
    nodesToLabel.remove(node)
    graph.neighbors[Int(node)].forEach {
      if !labeledNodes.contains($0) {
        nodesToLabel.update(with: $0)
      }
    }
  }

  // Proceed with label propagation for the unlabeled nodes.
  while !nodesToLabel.isEmpty {
    for node in nodesToLabel {
      let labeledNeighbors = graph.neighbors[Int(node)].filter(labeledNodes.contains)
      var probabilities = Tensor<Float>(zeros: [graph.classCount])
      for neighbor in labeledNeighbors {
        probabilities += labelProbabilities[Int(neighbor)]
      }
      probabilities /= probabilities.sum()
      labelProbabilities = _Raw.tensorScatterAdd(
        labelProbabilities,
        indices: Tensor<Int32>([node]).expandingShape(at: -1),
        updates: probabilities.expandingShape(at: 0))
    }
    for node in nodesToLabel {
      labeledNodes.update(with: node)
      nodesToLabel.remove(node)
      graph.neighbors[Int(node)].forEach {
        if !labeledNodes.contains($0) {
          nodesToLabel.update(with: $0)
        }
      }
    }
  }

  for node in (0..<Int32(graph.nodeCount)).filter({ !labeledNodes.contains($0) }) {
    labelProbabilities = _Raw.tensorScatterAdd(
      labelProbabilities,
      indices: Tensor<Int32>([node]).expandingShape(at: -1),
      updates: Tensor<Float>(
        repeating: 1 / Float(graph.classCount),
        shape: [1, graph.classCount]))
  }

  return labelProbabilities.argmax(squeezingAxis: -1).scalars
}
