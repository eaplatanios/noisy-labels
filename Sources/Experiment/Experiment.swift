// Copyright 2019, Emmanouil Antonios Platanios. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

import Foundation
import NoisyLabels
import Progress
import TensorFlow

fileprivate func sample<T>(from a: [T], count k: Int) -> [T] {
  var a = a
  for i in 0..<k {
    let r = i + Int(arc4random_uniform(UInt32(a.count - i)))
    if i != r {
      a.swapAt(i, r)
    }
  }
  return Array(a[0..<k])
}

fileprivate extension Array where Element == Float {
  var mean: Float {
    get {
      return self.reduce(0, { $0 + $1 }) / Float(self.count)
    }
  }

  var standardDeviation: Element {
    get {
      let mean = self.reduce(0, { $0 + $1 }) / Float(self.count)
      let variance = self.map { ($0 - mean) * ($0 - mean) }
      return TensorFlow.sqrt(variance.mean)
    }
  }
}

public struct Experiment {
  public let dataDir: URL
  public let dataset: Dataset
  public let usingFeatures: Bool
  public let learners: [String: (NoisyLabels.Data<Int, String, Int>) -> Learner]
  public let concurrentTaskCount: Int = 1

  let data: NoisyLabels.Data<Int, String, Int>

  public init(
    dataDir: URL,
    dataset: Dataset,
    usingFeatures: Bool,
    learners: [String: (NoisyLabels.Data<Int, String, Int>) -> Learner]
  ) throws {
    self.dataDir = dataDir
    self.dataset = dataset
    self.usingFeatures = usingFeatures
    self.learners = learners
    self.data = try dataset.loader(dataDir: dataDir).load(withFeatures: usingFeatures)
  }

  public func run() -> [Result] {
    var predictorSamplesCount = 0
    for (predictorCount, repetitionCount) in zip(
      dataset.predictorCounts(),
      dataset.repetitionCounts()
    ) {
      predictorSamplesCount += data.predictors.count <= predictorCount ? 1 : repetitionCount
    }

    var progressBar = ProgressBar(
      count: learners.count * predictorSamplesCount,
      configuration: [
        ProgressString(string: "Experiment Run:"),
        ProgressIndex(),
        ProgressBarLine(),
        ProgressTimeEstimates()])
    let progressBarQueue = DispatchQueue(label: "Noisy Labels Experiment Progress Bar")
    
    var results = [Result]()
    for (learnerName, learner) in learners {
      for (predictorCount, repetitionCount) in zip(
        dataset.predictorCounts(),
        dataset.repetitionCounts()
      ) {
        // TODO: resetSeed()
        let predictorSamples = data.predictors.count <= predictorCount ?
          [data.predictors] :
          (0..<repetitionCount).map { _ in sample(from: data.predictors, count: predictorCount) }
        var currentResults = [EvaluationResult?](repeating: nil, count: predictorSamples.count)
        DispatchQueue.concurrentPerform(iterations: predictorSamples.count) { repetition in
          progressBarQueue.sync { progressBar.next() }
          let filteredData = data.filtered(
            predictors: predictorSamples[repetition],
            keepInstances: true)
          var learner = learner(filteredData)
          learner.train(using: filteredData)
          currentResults[repetition] = EvaluationResult(
            merging: learner.evaluatePerLabel(using: filteredData))
        }

        // Aggregate the results.
        let dateFormatter = DateFormatter()
        dateFormatter.locale = Locale(identifier: "en_US_POSIX")
        dateFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ssZZZZZ"
        dateFormatter.timeZone = TimeZone(secondsFromGMT: 0)
        let timeStamp = dateFormatter.string(from: Date())
        let accuracies = currentResults.map { $0!.accuracy }
        results.append(Result(
          timeStamp: timeStamp,
          learner: learnerName,
          predictorCount: predictorCount,
          metric: "accuracy",
          valueMean: accuracies.mean,
          valueStandardDeviation: accuracies.standardDeviation))
        let aucs = currentResults.map { $0!.auc }
        results.append(Result(
          timeStamp: timeStamp,
          learner: learnerName,
          predictorCount: predictorCount,
          metric: "auc",
          valueMean: aucs.mean,
          valueStandardDeviation: aucs.standardDeviation))
      }
    }

    logger.info("Finished all experiments.")
    return results
  }
}

public extension Experiment {
  struct Result: Codable {
    public let timeStamp: String
    public let learner: String
    public let predictorCount: Int
    public let metric: String
    public let valueMean: Float
    public let valueStandardDeviation: Float
  }
}
