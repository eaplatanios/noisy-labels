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

func bestNodeAssignments(h: LabelLogits, q: [QualityLogits], G: Graph) -> [Int32] {
  var priorityQueue = PriorityQueue<SearchState>(order: { $0.upperBound < $1.upperBound })
  let nodeDegrees = G.neighbors.map { $0.count }
  let labeledNodes = [Int32: Int](
    uniqueKeysWithValues: G.trainNodes.map { ($0, G.labels[$0]!) })
  let sortedNodes = (0..<G.nodeCount)
    .filter { !labeledNodes.keys.contains(Int32($0)) }
    .sorted { nodeDegrees[$0] < nodeDegrees[$1] }

  var upperBound: Float = 0
  for node in sortedNodes {
    upperBound += h.maxLabelLogits[node]
    for (index, neighbor) in G.neighbors[node].enumerated() {
      if let label = labeledNodes[neighbor] {
        upperBound += q[node].maxQualityLogit(forNode: index, alongL: label)
      } else {
        upperBound += q[node].maxQualityLogits[index]
      }
    }
  }

  priorityQueue.push(SearchState(
    parentState: nil,
    splitNodeIndex: -1,
    splitValue: -1,
    upperBound: upperBound))
  var previousBound = upperBound
  while let currentState = priorityQueue.pop() {
    let nextNodeIndex = currentState.splitNodeIndex + 1
    let currentBound = currentState.upperBound
    print("Current Bound: \(currentBound) | Priority Queue Size: \(priorityQueue.count) | Split Node Index: \(currentState.splitNodeIndex)")
    if currentBound + 1e-9 > previousBound {
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
    var boundContribution: Float = h.maxLabelLogits[node]
    var neighborLabels = [Int](repeating: -1, count: G.neighbors[node].count)
    for (index, neighbor) in G.neighbors[node].enumerated() {
      if let label = getNodeLabel(labeledNodes, sortedNodes, currentState, Int(neighbor)) {
        neighborLabels[index] = label
        boundContribution += q[node].maxQualityLogit(forNode: index, alongL: label)
      } else {
        boundContribution += q[node].maxQualityLogits[index]
      }
    }

    for k in 0..<G.classCount {
      var newContribution: Float = h.labelLogit(node: node, label: k)
      for index in G.neighbors[node].indices {
        let label = neighborLabels[index]
        if label != -1 {
          newContribution += q[node].qualityLogits(node: index, l: k, k: label)
        } else {
          newContribution += q[node].maxQualityLogit(forNode: index, alongK: k)
        }
      }

      if newContribution > boundContribution {
        print("Bounds: \(newContribution) > \(boundContribution)")
        print("Neighbor Labels: \(neighborLabels)")
        exit(0)
      }

      priorityQueue.push(SearchState(
        parentState: currentState,
        splitNodeIndex: nextNodeIndex,
        splitValue: k,
        upperBound: currentState.upperBound - boundContribution + newContribution))
    }
  }

  return []
}

public struct LabelLogits {
  public let logits: [Float]
  public let nodeCount: Int
  public let labelCount: Int

  public let maxLabelLogits: [Float]

  public init(logits: [Float], nodeCount: Int, labelCount: Int) {
    self.logits = logits
    self.nodeCount = nodeCount
    self.labelCount = labelCount
    self.maxLabelLogits = {
      var maxLabelLogits = [Float]()
      maxLabelLogits.reserveCapacity(nodeCount)
      for node in 0..<nodeCount {
        var maxLogit = logits[node * labelCount]
        for label in 1..<labelCount {
          maxLogit = max(maxLogit, logits[node * labelCount + label])
        }
        maxLabelLogits.append(maxLogit)
      }
      return maxLabelLogits
    }()
  }

  @inlinable
  public func labelLogit(node: Int, label: Int) -> Float {
    logits[node * labelCount + label]
  }
}

public struct QualityLogits {
  public let logits: [Float]
  public let nodeCount: Int
  public let labelCount: Int
  public let maxQualityLogits: [Float]
  public let maxQualityLogitsAlongK: [Float]
  public let maxQualityLogitsAlongL: [Float]

  public init(logits: [Float], nodeCount: Int, labelCount: Int) {
    self.logits = logits
    self.nodeCount = nodeCount
    self.labelCount = labelCount

    self.maxQualityLogits = {
      var maxQualityLogits = [Float]()
      maxQualityLogits.reserveCapacity(nodeCount)
      for node in 0..<nodeCount {
        var maxLogit = logits[node * labelCount * labelCount]
        for l in 0..<labelCount {
          for k in 0..<labelCount {
            if k != 0 || l > 0 {
              maxLogit = max(maxLogit, logits[node * labelCount * labelCount + l * labelCount + k])
            }
          }
        }
        maxQualityLogits.append(maxLogit)
      }
      return maxQualityLogits
    }()

    self.maxQualityLogitsAlongK = {
      var maxQualityLogitsAlongK = [Float]()
      maxQualityLogitsAlongK.reserveCapacity(nodeCount * labelCount)
      for node in 0..<nodeCount {
        for l in 0..<labelCount {
          var maxLogit = logits[node * labelCount * labelCount + l * labelCount]
          for k in 1..<labelCount {
            maxLogit = max(maxLogit, logits[node * labelCount * labelCount + l * labelCount + k])
          }
          maxQualityLogitsAlongK.append(maxLogit)
        }
      }
      return maxQualityLogitsAlongK
    }()

    self.maxQualityLogitsAlongL = {
      var maxQualityLogitsAlongL = [Float]()
      maxQualityLogitsAlongL.reserveCapacity(nodeCount * labelCount)
      for node in 0..<nodeCount {
        for k in 0..<labelCount {
          var maxLogit = logits[node * labelCount * labelCount + k]
          for l in 1..<labelCount {
            maxLogit = max(maxLogit, logits[node * labelCount * labelCount + l * labelCount + k])
          }
          maxQualityLogitsAlongL.append(maxLogit)
        }
      }
      return maxQualityLogitsAlongL
    }()
  }

  @inlinable
  public func qualityLogits(node: Int, l: Int, k: Int) -> Float {
    logits[node * labelCount * labelCount + l * labelCount + k]
  }

  @inlinable
  public func maxQualityLogit(forNode node: Int, alongK label: Int) -> Float {
    maxQualityLogitsAlongK[node * labelCount + label]
  }

  @inlinable
  public func maxQualityLogit(forNode node: Int, alongL label: Int) -> Float {
    maxQualityLogitsAlongL[node * labelCount + label]
  }
}
