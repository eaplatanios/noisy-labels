import TensorFlow


public class SearchState {
  public let parentState: SearchState?
  public let splitNodeIndex: Int
  public let splitValue: Int
  public let upperBound: Float

  public init(
    parentState: SearchState?,
    splitNodeIndex: Int,
    splitValue: Int,
    upperBound: Float
  ) {
    self.parentState = parentState
    self.splitNodeIndex = splitNodeIndex
    self.splitValue = splitValue
    self.upperBound = upperBound
  }
}

func getNodeLabel(
  _ labeledNodes: [Int32: Int],
  _ sortedNodes: [Int],
  _ currentState: SearchState,
  _ node: Int
) -> Int? {
  if let label = labeledNodes[Int32(node)] {
    return label
  }

  var state: SearchState? = currentState
  while let currState = state, currState.splitNodeIndex != -1 {
    if sortedNodes[currState.splitNodeIndex] == Int(node) {
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
    uniqueKeysWithValues: G.trainNodes.map { ($0, G.labels[$0]!) })
  let sortedNodes = (0..<G.nodeCount)
    .filter { labeledNodes.keys.contains(Int32($0)) }
    .sorted { nodeDegrees[$0] > nodeDegrees[$1] }

  var upperBound: Float = 0
  for node in sortedNodes {
    upperBound += h[node].max().scalarized()
    for (index, neighbor) in G.neighbors[node].enumerated() {
      if let label = labeledNodes[neighbor] {
        upperBound += q[node][index, 0..., label].max().scalarized()
      } else {
        upperBound += q[node][index].max().scalarized()
      }
    }
  }

  priorityQueue.insert(SearchState(
    parentState: nil,
    splitNodeIndex: -1,
    splitValue: -1,
    upperBound: upperBound))
  var previousBound = upperBound
  while let currentState = priorityQueue.remove() {
    let nextNodeIndex = currentState.splitNodeIndex + 1
    let currentBound = currentState.upperBound
    print("Queue Size: \(currentBound) | Priority Queue Size: \(priorityQueue.count) | Split Node Index: \(currentState.splitNodeIndex)")
    if currentBound > previousBound + 1e-9 {
      print("haha Christoph...urgh")
      exit(0)
    }
    previousBound = currentBound
    if nextNodeIndex == sortedNodes.count {
      // we found the optimal node assignments
      var results = [Int32](repeating: 0, count: G.nodeCount)
      var state: SearchState? = currentState
      while let currState = state, currState.splitNodeIndex != -1 {
        let node = sortedNodes[currState.splitNodeIndex]
        results[node] = Int32(currState.splitValue)
        state = currState.parentState
      }
      for (node, label) in labeledNodes {
        results[Int(node)] = Int32(label)
      }
      return results
    }

    let node = sortedNodes[nextNodeIndex]
    var boundContribution: Float = h[node].max().scalarized()
    var neighborLabels = [Int](repeating: -1, count: G.neighbors[node].count)
    for (index, neighbor) in G.neighbors[node].enumerated() {
      if let label = getNodeLabel(labeledNodes, sortedNodes, currentState, Int(neighbor)) {
        neighborLabels[index] = label
        boundContribution += q[node][index, 0..., label].max().scalarized()
      } else {
        boundContribution += q[node][index].max().scalarized()
      }
    }

    for k in 0..<G.classCount {
      var newContribution: Float = h[node, k].scalarized()
      for index in G.neighbors[node].indices {
        let label = neighborLabels[index]
        if label != -1 {
          newContribution += q[node][index, k, label].scalarized()
        } else {
          newContribution += q[node][index, k, 0...].max().scalarized()
        }
      }

      priorityQueue.insert(SearchState(
        parentState: currentState,
        splitNodeIndex: nextNodeIndex,
        splitValue: k,
        upperBound: currentState.upperBound - boundContribution + newContribution))
    }
  }

  return []
}
