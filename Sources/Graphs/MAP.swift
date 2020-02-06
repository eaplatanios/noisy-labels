//import TensorFlow
//
//public class SearchState {
//  public let parentState: SearchState?
//  public let splitNodeIndex: Int
//  public let splitValue: Int
//  public let upperBound: Float
//
//  public init(
//    parentState: SearchState?,
//    splitNodeIndex: Int,
//    splitValue: Int,
//    upperBound: Float
//  ) {
//    self.parentState = parentState
//    self.splitNodeIndex = splitNodeIndex
//    self.splitValue = splitValue
//    self.upperBound = upperBound
//  }
//}
//
//func getNodeLabel(
//  _ labeledNodes: [Int32: Int],
//  _ sortedNodes: [Int],
//  _ currentState: SearchState,
//  _ node: Int
//) -> Int? {
//  if let label = labeledNodes[Int32(node)] {
//    return label
//  }
//
//  var state: SearchState? = currentState
//  while let currState = state, currState.splitNodeIndex != -1 {
//    if sortedNodes[currState.splitNodeIndex] == Int(node) {
//      return currState.splitValue
//    }
//    state = currState.parentState
//  }
//
//  return nil
//}
//
//func bestNodeAssignments(h: LabelLogits, q: [QualityLogits], G: Graph) -> [Int32] {
//  var priorityQueue = PriorityQueue<SearchState>(order: { $0.upperBound < $1.upperBound })
//  let nodeDegrees = G.neighbors.map { $0.count }
//  let labeledNodes = [Int32: Int](
//    uniqueKeysWithValues: G.trainNodes.map { ($0, G.labels[$0]!) })
//  let sortedNodes = (0..<G.nodeCount)
//    .filter { !labeledNodes.keys.contains(Int32($0)) }
//    .sorted { nodeDegrees[$0] > nodeDegrees[$1] }
//
//  var upperBound: Float = 0
//  for node in sortedNodes {
//    upperBound += h.maxLabelLogits[node]
//    for (index, neighbor) in G.neighbors[node].enumerated() {
//      if let label = labeledNodes[neighbor] {
//        upperBound += q[node].maxQualityLogit(forNode: index, alongL: label)
//      } else {
//        upperBound += q[node].maxQualityLogits[index]
//      }
//    }
//  }
//
//  priorityQueue.push(SearchState(
//    parentState: nil,
//    splitNodeIndex: -1,
//    splitValue: -1,
//    upperBound: upperBound))
//  var previousBound = upperBound
//  while let currentState = priorityQueue.pop() {
//    let nextNodeIndex = currentState.splitNodeIndex + 1
//    let currentBound = currentState.upperBound
//    print("Current Bound: \(currentBound) | Priority Queue Size: \(priorityQueue.count) | Split Node Index: \(currentState.splitNodeIndex)")
//    if currentBound + 1e-9 > previousBound {
//      print("haha Christoph...urgh")
//      exit(0)
//    }
//    previousBound = currentBound
//    if nextNodeIndex == sortedNodes.count {
//      // we found the optimal node assignments
//      var results = [Int32](repeating: 0, count: G.nodeCount)
//      var state: SearchState? = currentState
//      while let currState = state, currState.splitNodeIndex != -1 {
//        let node = sortedNodes[currState.splitNodeIndex]
//        results[node] = Int32(currState.splitValue)
//        state = currState.parentState
//      }
//      for (node, label) in labeledNodes {
//        results[Int(node)] = Int32(label)
//      }
//      return results
//    }
//
//    let node = sortedNodes[nextNodeIndex]
//    var boundContribution: Float = h.maxLabelLogits[node]
//    var neighborLabels = [Int](repeating: -1, count: G.neighbors[node].count)
//    for (index, neighbor) in G.neighbors[node].enumerated() {
//      if let label = getNodeLabel(labeledNodes, sortedNodes, currentState, Int(neighbor)) {
//        neighborLabels[index] = label
//        boundContribution += q[node].maxQualityLogit(forNode: index, alongL: label)
//      } else {
//        boundContribution += q[node].maxQualityLogits[index]
//      }
//    }
//
//    for k in 0..<G.classCount {
//      var newContribution: Float = h.labelLogit(node: node, label: k)
//      for index in G.neighbors[node].indices {
//        let label = neighborLabels[index]
//        if label != -1 {
//          newContribution += q[node].qualityLogits(node: index, l: k, k: label)
//        } else {
//          newContribution += q[node].maxQualityLogit(forNode: index, alongK: k)
//        }
//      }
//
//      if newContribution > boundContribution {
//        print("Bounds: \(newContribution) > \(boundContribution)")
//        print("Neighbor Labels: \(neighborLabels)")
//        exit(0)
//      }
//
//      priorityQueue.push(SearchState(
//        parentState: currentState,
//        splitNodeIndex: nextNodeIndex,
//        splitValue: k,
//        upperBound: currentState.upperBound - boundContribution + newContribution))
//    }
//  }
//
//  return []
//}
