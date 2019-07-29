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
  var mean: Float { reduce(0, { $0 + $1 }) / Float(count) }

  var standardDeviation: Element {
    let mean = reduce(0, { $0 + $1 }) / Float(count)
    let variance = map { ($0 - mean) * ($0 - mean) }
    return TensorFlow.sqrt(variance.mean)
  }
}

public struct Experiment {
  public let dataDir: URL
  public let dataset: Dataset
  public let usingFeatures: Bool
  public let learners: [String: (NoisyLabels.Data<Int, String, Int>) -> Learner]
  public let concurrentTaskCount: Int = 1

  internal let data: NoisyLabels.Data<Int, String, Int>
  internal let dispatchQueue: DispatchQueue = DispatchQueue(
    label: "Noisy Labels Experiments",
    attributes: .concurrent)
  internal let dispatchGroup: DispatchGroup = DispatchGroup()

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
    self.data = try dataset.loader(dataDir).load(withFeatures: usingFeatures)
  }

  public func run(
    callback: ExperimentCallback? = nil,
    runs: [Experiment.Run]? = nil
  ) -> [Result] {
    let runs = runs ?? dataset.runs
    let totalRunCount = runs.map { run in
      switch run {
      case let .simple(predictorCount, repetitionCount):
        return data.predictors.count <= predictorCount ? 1 : repetitionCount
      case let .redundancy(max, repetitionCount):
        return data.predictors.count <= max ? 1 : repetitionCount
      }
    }.reduce(0, +)

    var progressBar = ProgressBar(
      count: learners.count * totalRunCount,
      configuration: [
        ProgressString(string: "Experiment Run:"),
        ProgressIndex(),
        ProgressBarLine(),
        ProgressTimeEstimates()])
    let progressBarQueue = DispatchQueue(label: "Noisy Labels Experiment Progress Bar")

    for (learnerName, learner) in learners {
      for run in runs {
        switch run {
        case let .simple(predictorCount, repetitionCount):
          // TODO: resetSeed()
          let predictorSamples = data.predictors.count <= predictorCount ?
            [data.predictors] :
            (0..<repetitionCount).map { _ in sample(from: data.predictors, count: predictorCount) }
          for repetition in 0..<predictorSamples.count {
            dispatchQueue.async(group: dispatchGroup) { [data] () in
              progressBarQueue.sync { progressBar.next() }
              let filteredData = data.filtered(
                predictors: predictorSamples[repetition],
                keepInstances: true)
              var learner = learner(filteredData)
              learner.train(using: filteredData)
              let result = EvaluationResult(merging: learner.evaluatePerLabel(using: filteredData))
              let dateFormatter = DateFormatter()
              dateFormatter.locale = Locale(identifier: "en_US_POSIX")
              dateFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ssZZZZZ"
              dateFormatter.timeZone = TimeZone(secondsFromGMT: 0)
              let timeStamp = dateFormatter.string(from: Date())
              for (metric, value) in [
                ("madErrorRank", result.madErrorRank),
                ("madError", result.madError),
                ("accuracy", result.accuracy),
                ("auc", result.auc)
              ] {
                callback?(Result(
                  timeStamp: timeStamp,
                  learner: learnerName,
                  parameterType: .predictorCount,
                  parameter: predictorCount,
                  metric: metric,
                  value: value))
              }
            }
          }
        case let .redundancy(max, repetitionCount):
          fatalError("Redundancy experiments have not been implemented yet.")
        }
      }
    }

    dispatchGroup.wait()
    logger.info("Finished all experiments.")
    return results
  }
}

public protocol ExperimentCallback {
  func callAsFunction(_ result: Experiment.Result)
}

public struct ResultsWriter: ExperimentCallback {
  public let fileURL: URL

  private let dispatchQueue: DispatchQueue = DispatchQueue(label: "Results Writer")

  public init(at fileURL: URL) {
    self.fileURL = fileURL
    if !FileManager.default.fileExists(atPath: fileURL.path) {
      let header = "timeStamp\tlearner\tparameterType\tparameter\tmetric\tvalue\n"
      FileManager.default.createFile(atPath: fileURL.path, contents: header.data(using: .utf8))
    }
  }

  public func callAsFunction(_ result: Experiment.Result) {
    let resultParts = [
      "\(result.timeStamp)",
      "\(result.learner)",
      "\(result.parameterType)",
      "\(result.parameter)",
      "\(result.metric)",
      "\(result.value)"]
    let result = resultParts.joined(separator: "\t") + "\n"
    dispatchQueue.sync {
      let fileHandle = try! FileHandle(forWritingTo: fileURL)
      fileHandle.seekToEndOfFile()
      fileHandle.write(result.data(using: .utf8)!)
      fileHandle.closeFile()
    }
  }
}

extension Experiment {
  public struct Result: Codable {
    public let timeStamp: String
    public let learner: String
    public let parameterType: ParameterType
    public let parameter: Int
    public let metric: String
    public let value: Float
  }

  public enum ParameterType: String, Codable {
    case predictorCount, redundancy
  }
}
