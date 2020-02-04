public struct SearchState {
  public let parentState: SearchState?
  public let splitNodeIndex: Int
  public let splitValue: Int
  public let upperBound: Float
}

func getNodeLabel(labeledNodes: [Int32: Int], sortedNodes: [Int32], currentState: SearchState, node: Int) -> Int? {
  if let label = labeledNodes[node] {
    return label
  }

  var state: SearchState? = currentState
  while let currState = state, currState.splitNodeIndex != -1 {
    if sortedNodes[currState.splitNodeIndex] == node {
      return currState.splitValue
    }
    state = currState.parentState
  }

  return nil
}

func bestNodeAssignments(h: Tensor<Float>, q: [Tensor<Float>], G: Graph) -> [Int32] {
  var priorityQueue = Heap<SearchState>(sort: { $0.upperBound > $1.upperBound })
  let nodeDegrees = G.neighbors.map { $0.count }
  let labeledNodes = [Int32: Int](
    uniqueKeysWithValues: G.trainNodes.map { G.labels[$0]! })
  let sortedNodes = (0..<G.nodeCount)
    .filter { labeledNodes.keys.contains($0) }
    .sorted { nodeDegrees[$0] > nodeDegrees[$1] }

  var upperBound: Float = 0
  for node in sortedNodes {
    upperBound += h[node].max().scalarized()
    for neighbor in G.neighbors[node] {
      if let label = labeledNodes[neighbor] {
        upperBound += q[node][Int(neighbor), 0..., label].max().scalarized()
      } else {
        upperBound += q[node][Int(neighbor)].max().scalarized()
      }
    }
  }

  priorityQueue.insert(SearchState(
    parentState: nil,
    splitNode: -1,
    splitValue: -1,
    upperBound: upperBound))
  while let currentState = priorityQueue.remove() {
    let nextNodeIndex = currentState.splitNodeIndex + 1
    if nextNodeIndex == sortedNodes.count {
      // we found the optimal node assignments
      var results = [Int32](repeating: 0, count: G.nodeCount)
      var state: SearchState? = currentState
      while let currState = state, currState.splitNodeIndex != -1 {
        let node = sortedNodes[currState.splitNodeIndex]
        results[node] = currState.splitValue
        state = currState.parentState
      }
      for (node, label) in labeledNodes {
        results[Int(node)] = label
      }
      return results
    }

    let node = sortedNodes[nextNodeIndex]
    var boundContribution: Float = h[node].max().scalarized()
    var neighborLabels = [Int](repeating: -1, count: G.neighbors[node].count)
    for (index, neighbor) in G.neighbors[node].enumerated() {
      if let label = getNodeLabel(labeledNodes, sortedNodes, currentState, neighbor) {
        neighborLabels[index] = label
        boundContribution += q[node][Int(neighbor), 0..., label].max().scalarized()
      } else {
        boundContribution += q[node][Int(neighbor)]
      }
    }

    for k in 0..<G.classCount {
      var newContribution: Float = h[node, k].scalarized()
      for (index, neighbor) in G.neighbors[node].enumerated() {
        if let label = neighborLabels[index], label != -1 {
          newContribution += q[node][Int(neighbor), k, label].scalarized()
        } else {
          newContribution += q[node][Int(neighbor), k, 0...].max().scalarized()
        }
      }

      priorityQueue.insert(SearchState(
        parentState: currentState,
        splitNode: nextNodeIndex,
        splitValue: k,
        upperBound: currentState.upperBound - boundContribution + maxContribution))
    }
  }
}
