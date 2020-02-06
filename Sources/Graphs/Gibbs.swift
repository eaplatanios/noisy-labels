
import Foundation
import Progress

func gibbsMarginalMAP(
  labelLogits: LabelLogits,
  qualityLogits: [QualityLogits],
  graph: Graph,
  sampleCount: Int = 10,
  burnInSampleCount: Int = 100,
  thinningSampleCount: Int = 10
) -> [Int32] {
  var samples = [[Int32]]()
  var sample = [Int32](repeating: -1, count: graph.nodeCount)

  // Start with the labeled nodes.
  for node in graph.trainNodes {
    sample[Int(node)] = Int32(graph.labels[node]!)
  }

  let leveledData = graph.leveledData.suffix(from: 1)

  var progressBar = ProgressBar(
    count: burnInSampleCount + (sampleCount - 1) * thinningSampleCount,
    configuration: [
      ProgressString(string: "Gibbs Sampling Progress"),
      ProgressIndex(),
      ProgressBarLine(),
      ProgressTimeEstimates()])
  var currentSampleCount = 0
  while samples.count < sampleCount {
    for level in leveledData {
      for node in level.shuffled() {
        var probabilities = labelLogits.labelLogits(forNode: Int(node)).map(exp)
        for (neighborIndex, neighbor) in graph.neighbors[Int(node)].enumerated() {
          let neighborNodeIndex = Int(graph.neighbors[Int(neighbor)].firstIndex(of: node)!)
          for k in 0..<graph.classCount {
            for l in 0..<graph.classCount {
              probabilities[k] += exp(qualityLogits[Int(node)].qualityLogits(node: neighborIndex, l: k, k: l))
              probabilities[k] += exp(qualityLogits[Int(neighbor)].qualityLogits(node: neighborNodeIndex, l: l, k: k))
            }
          }
        }
        let random = Float.random(in: 0..<probabilities.reduce(0, +))
        var accumulator: Float = 0
        for (k, p) in probabilities.enumerated() {
          accumulator += p
          if random < accumulator {
            sample[Int(node)] = Int32(k)
            break
          }
        }
      }
    }
    currentSampleCount += 1
    progressBar.setValue(currentSampleCount)
    if currentSampleCount > burnInSampleCount &&
       currentSampleCount.isMultiple(of: thinningSampleCount) {
      samples.append(sample)
    }
  }

  var marginalMAP = sample
  let unlabeledNodes = graph.validationNodes + graph.testNodes + graph.unlabeledNodes
  for node in unlabeledNodes {
    var scores = [Float](repeating: 0, count: graph.classCount)
    for sample in samples {
      scores[Int(sample[Int(node)])] += 1
    }
    marginalMAP[Int(node)] = Int32(scores.indexOfMax!)
  }

  return marginalMAP
}
