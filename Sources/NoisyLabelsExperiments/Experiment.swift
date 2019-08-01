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

public struct Experiment<Dataset: NoisyLabelsExperiments.Dataset>
where Dataset.Loader.Predictor: Equatable {
  public typealias Instance = Dataset.Loader.Instance
  public typealias Predictor = Dataset.Loader.Predictor
  public typealias Label = Dataset.Loader.Label

  public let dataDir: URL
  public let dataset: Dataset
  public let learners: [String: Learner]

  internal let data: NoisyLabels.Data<Instance, Predictor, Label>
  internal let concurrentQueue = DispatchQueue(label: "Noisy Labels", attributes: .concurrent)
  internal let serialQueue = DispatchQueue(label: "Noisy Labels Serial")
  internal let dispatchGroup = DispatchGroup()
  internal let progressBarDispatchQueue = DispatchQueue(label: "Progress Bar")

  public init(dataDir: URL, dataset: Dataset, learners: [String: Learner]) throws {
    self.dataDir = dataDir
    self.dataset = dataset
    self.learners = learners
    self.data = try dataset.loader(dataDir).load(
      withFeatures: learners.contains(where: { $0.value.requiresFeatures }))
  }

  public func run(
    callback: ((ExperimentResult) -> ())? = nil,
    runs: [ExperimentRun]? = nil,
    parallelismLimit: Int? = nil
  ) {
    let runs = runs ?? dataset.runs
    let totalRunCount = runs.map { run in
      switch run {
      case let .predictorSubsampling(_, repetitionCount): return repetitionCount
      case let .redundancy(_, repetitionCount): return repetitionCount
      }
    }.reduce(0, +)
    var progressBar = ProgressBar(
      count: learners.count * totalRunCount,
      configuration: [
        ProgressString(string: "Experiment:"),
        ProgressIndex(),
        ProgressBarLine(),
        ProgressTimeEstimates()])
    let dispatchSemaphore: DispatchSemaphore? = {
      if let limit = parallelismLimit {
        return DispatchSemaphore(value: limit)
      } else {
        return nil
      }
    }()
    for run in runs {
      switch run {
      case let .predictorSubsampling(predictorCount, repetitionCount):
        // TODO: resetSeed()
        let predictorSamples = data.predictors.count <= predictorCount ?
          [[Predictor]](repeating: data.predictors, count: repetitionCount) :
          (0..<repetitionCount).map { _ in sample(from: data.predictors, count: predictorCount) }
        for (learnerName, learner) in learners {
          let queue = learner.supportsMultiThreading ? concurrentQueue : serialQueue
          for repetition in 0..<predictorSamples.count {
            queue.async(group: dispatchGroup) { [data] () in
              dispatchSemaphore?.wait()
              defer { dispatchSemaphore?.signal() }
              self.progressBarDispatchQueue.sync { progressBar.next() }
              let filteredData = data.filtered(
                predictors: predictorSamples[repetition],
                keepInstances: true)
              var learner = learner.createFn(filteredData)
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
                callback?(ExperimentResult(
                  timeStamp: timeStamp,
                  learner: learnerName,
                  parameterType: .predictorCount,
                  parameter: predictorSamples.count,
                  metric: metric,
                  value: value))
              }
            }
          }
        }
      case let .redundancy(maxRedundancy, repetitionCount):
        for _ in 0..<repetitionCount {
          // TODO: resetSeed()
          let filteredData = data.withMaxRedundancy(maxRedundancy)
          for (learnerName, learner) in learners {
            let queue = learner.supportsMultiThreading ? concurrentQueue : serialQueue
            queue.async(group: dispatchGroup) { [data] () in
              dispatchSemaphore?.wait()
              defer { dispatchSemaphore?.signal() }
              self.progressBarDispatchQueue.sync { progressBar.next() }
              var learner = learner.createFn(filteredData)
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
                callback?(ExperimentResult(
                  timeStamp: timeStamp,
                  learner: learnerName,
                  parameterType: .redundancy,
                  parameter: maxRedundancy,
                  metric: metric,
                  value: value))
              }
            }
          }
        }
      }
    }
    dispatchGroup.wait()
    logger.info("Finished all experiments.")
  }
}

extension Experiment {
  public struct Learner {
    public let createFn: (NoisyLabels.Data<Instance, Predictor, Label>) -> NoisyLabels.Learner
    public let requiresFeatures: Bool
    public let supportsMultiThreading: Bool
  }
}

public func resultsWriter(at fileURL: URL) throws -> (ExperimentResult) -> () {
  if !FileManager.default.fileExists(atPath: fileURL.path) {
    try FileManager.default.createDirectory(
      at: fileURL.deletingLastPathComponent(), 
      withIntermediateDirectories: true)
    let header = "timeStamp\tlearner\tparameterType\tparameter\tmetric\tvalue\n"
    FileManager.default.createFile(atPath: fileURL.path, contents: header.data(using: .utf8))
  }
  let semaphore = DispatchSemaphore(value: 1)
  return { [fileURL, semaphore] result in
    let resultParts = [
      "\(result.timeStamp)",
      "\(result.learner)",
      "\(result.parameterType)",
      "\(result.parameter)",
      "\(result.metric)",
      "\(result.value)"]
    let result = resultParts.joined(separator: "\t") + "\n"
    semaphore.wait()
    defer { semaphore.signal() }
    let fileHandle = try! FileHandle(forWritingTo: fileURL)
    fileHandle.seekToEndOfFile()
    fileHandle.write(result.data(using: .utf8)!)
    fileHandle.closeFile()
  }
}

public enum ExperimentRun {
  case predictorSubsampling(predictorCount: Int, repetitionCount: Int)
  case redundancy(maxRedundancy: Int, repetitionCount: Int)
}

public struct ExperimentResult: Codable {
  public let timeStamp: String
  public let learner: String
  public let parameterType: ParameterType
  public let parameter: Int
  public let metric: String
  public let value: Float
}

extension ExperimentResult {
  public enum ParameterType: String, Codable {
    case predictorCount, redundancy
  }
}
