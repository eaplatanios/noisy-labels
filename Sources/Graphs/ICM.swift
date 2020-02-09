
func iteratedConditionalModes(
  labelLogits: LabelLogits,
  qualityLogits: [QualityLogits],
  graph: Graph,
  maxStepCount: Int = 100
) -> [Int32] {
  // logger.info("Starting ICM approximation of the labels mode.")
  // var labels = labelPropagation(graph: graph).map { $0.indexOfMax! }
  var labels = initialization(
    labelLogits: labelLogits,
    qualityLogits: qualityLogits,
    graph: graph)
  let leveledData = graph.groupedNodes.suffix(from: 1)
  for _ in 0..<maxStepCount {
    var changed = 0
    for level in leveledData {
      for node in level.shuffled() {
        let neighbors = graph.neighbors[Int(node)]
        var scores = labelLogits.labelLogits(forNode: Int(node))
        for (index, neighbor) in neighbors.enumerated() {
          for k in 0..<graph.classCount {
            scores[k] += qualityLogits[Int(node)].qualityLogit(
              forNeighbor: index,
              nodeLabel: k,
              neighborLabel: Int(labels[Int(neighbor)])) / Float(neighbors.count)
          }
        }
        let label = Int32(scores.indexOfMax!)
        if labels[Int(node)] != label { changed += 1 }
        labels[Int(node)] = label
      }
    }
    // logger.info("ICM Step \(step): \(changed) labels changed.")
    // if changed == 0 { break }
    if changed < graph.nodeCount / 300 { break }
  }
  return labels
}

func initialization(
  labelLogits: LabelLogits,
  qualityLogits: [QualityLogits],
  graph: Graph
) -> [Int32] {
  var labels = [Int32](repeating: -1, count: graph.nodeCount)

  // Start with the labeled nodes.
  for node in graph.trainNodes {
    labels[Int(node)] = Int32(graph.labels[node]!)
  }

  // Proceed in a breadth-first order from the labeled nodes.
  for level in graph.groupedNodes.suffix(from: 1) {
    for node in level {
      let neighbors = graph.neighbors[Int(node)]
      var scores = labelLogits.labelLogits(forNode: Int(node))
      for (index, neighbor) in neighbors.enumerated() {
        let label = labels[Int(neighbor)]
        if label != -1 {
          for k in 0..<graph.classCount {
            scores[k] += qualityLogits[Int(node)].qualityLogit(
              forNeighbor: index,
              nodeLabel: k,
              neighborLabel: Int(label)) / Float(neighbors.count)
          }
        } else {
          for k in 0..<graph.classCount {
            scores[k] += qualityLogits[Int(node)].maxQualityLogit(
              forNeighbor: index,
              nodeLabel: k) / Float(neighbors.count)
          }
        }
      }
      labels[Int(node)] = Int32(scores.indexOfMax!)
    }
  }

  return labels
}

func labelPropagation(graph: Graph) -> [[Float]] {
  var labelScores = [[Float]](
    repeating: [Float](repeating: 0, count: graph.classCount),
    count: graph.nodeCount)
  var labeledNodes = Set<Int32>()
  var nodesToLabel = Set<Int32>()

  // Start with the labeled nodes.
  for node in graph.trainNodes {
    let label = graph.labels[node]!
    labelScores[Int(node)][label] += 1
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
      var scoresSum: Float = 0
      for neighbor in labeledNeighbors {
        for label in 0..<graph.classCount {
          let score = labelScores[Int(neighbor)][label]
          labelScores[Int(node)][label] += score
          scoresSum += score
        }
      }
      for label in 0..<graph.classCount {
        labelScores[Int(node)][label] /= scoresSum
      }
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
    for label in 0..<graph.classCount {
      labelScores[Int(node)][label] = 1 / Float(graph.classCount)
    }
  }

  return labelScores
}
