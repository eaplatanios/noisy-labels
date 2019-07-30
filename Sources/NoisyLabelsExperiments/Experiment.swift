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

internal extension Array where Element == Float {
  var mean: Float { reduce(0, { $0 + $1 }) / Float(count) }
  var standardDeviation: Element {
    let mean = reduce(0, { $0 + $1 }) / Float(count)
    let variance = map { ($0 - mean) * ($0 - mean) }
    return TensorFlow.sqrt(variance.mean)
  }
}

public struct Experiment<Dataset: NoisyLabelsExperiments.Dataset>
where Dataset.Loader.Predictor: Equatable {
  public typealias Instance = Dataset.Loader.Instance
  public typealias Predictor = Dataset.Loader.Predictor
  public typealias Label = Dataset.Loader.Label

  public let dataDir: URL
  public let dataset: Dataset
  public let learners: [String: Learner]
  public let concurrentTaskCount: Int = 1

  internal let data: NoisyLabels.Data<Instance, Predictor, Label>
  internal let runsDispatchQueue: DispatchQueue = DispatchQueue(label: "Experiment")
  internal let runsDispatchGroup: DispatchGroup = DispatchGroup()
  internal let callbackSemaphore = DispatchSemaphore(value: 1)

  public init(
    dataDir: URL,
    dataset: Dataset,
    learners: [String: Learner]
  ) throws {
    self.dataDir = dataDir
    self.dataset = dataset
    self.learners = learners
    self.data = try dataset.loader(dataDir).load(
      withFeatures: learners.contains(where: { $0.value.requiresFeatures }))
  }

  public func run(
    callback: ((ExperimentResult) -> ())? = nil,
    runs: [ExperimentRun]? = nil
  ) {
    let runs = runs ?? dataset.runs
    let totalRunCount = runs.map { run in
      switch run {
      case let .predictorSubsampling(predictorCount, repetitionCount):
        return data.predictors.count <= predictorCount ? 1 : repetitionCount
      case let .redundancy(max, repetitionCount):
        return data.predictors.count <= max ? 1 : repetitionCount
      }
    }.reduce(0, +)
    let progressBarDispatchQueue = DispatchQueue(label: "Progress Bar")
    var progressBar = ProgressBar(
      count: learners.count * totalRunCount,
      configuration: [
        ProgressString(string: "Experiment Run:"),
        ProgressIndex(),
        ProgressBarLine(),
        ProgressTimeEstimates()])
    for (learnerName, learner) in learners {
      for run in runs {
        switch run {
        case let .predictorSubsampling(predictorCount, repetitionCount):
          // TODO: resetSeed()
          let predictorSamples = data.predictors.count <= predictorCount ?
            [data.predictors] :
            (0..<repetitionCount).map { _ in sample(from: data.predictors, count: predictorCount) }
          for repetition in 0..<predictorSamples.count {
            if learner.supportsMultiThreading {
              runsDispatchQueue.async(group: runsDispatchGroup) { [data] () in
                progressBarDispatchQueue.sync { progressBar.next() }
                self.runPredictorSubsamplingExperiment(
                  predictorSamples: predictorSamples[repetition],
                  learnerName: learnerName,
                  learner: learner,
                  data: data,
                  callback: callback)
              }
            } else {
              progressBar.next()
              runPredictorSubsamplingExperiment(
                predictorSamples: predictorSamples[repetition],
                learnerName: learnerName,
                learner: learner,
                data: data,
                callback: callback)
            }
          }
        case let .redundancy(maxRedundancy, repetitionCount):
          let actualRepetitionCount = data.predictors.count <= maxRedundancy ? 1 : repetitionCount
          for _ in 0..<actualRepetitionCount {
            if learner.supportsMultiThreading {
              runsDispatchQueue.async(group: runsDispatchGroup) { [data] () in
                progressBarDispatchQueue.sync { progressBar.next() }
                self.runRedundancyExperiment(
                  maxRedundancy: maxRedundancy,
                  learnerName: learnerName,
                  learner: learner,
                  data: data,
                  callback: callback)
              }
            } else {
              progressBar.next()
              runRedundancyExperiment(
                maxRedundancy: maxRedundancy,
                learnerName: learnerName,
                learner: learner,
                data: data,
                callback: callback)
            }
          }
        }
      }
    }
    runsDispatchGroup.wait()
    logger.info("Finished all experiments.")
  }

  private func runPredictorSubsamplingExperiment(
    predictorSamples: [Predictor],
    learnerName: String,
    learner: Learner,
    data: NoisyLabels.Data<Instance, Predictor, Label>,
    callback: ((ExperimentResult) -> ())?
  ) {
    let filteredData = data.filtered(
      predictors: predictorSamples,
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
      callbackSemaphore.wait()
      callback?(ExperimentResult(
        timeStamp: timeStamp,
        learner: learnerName,
        parameterType: .predictorCount,
        parameter: predictorSamples.count,
        metric: metric,
        value: value))
      callbackSemaphore.signal()
    }
  }

  private func runRedundancyExperiment(
    maxRedundancy: Int,
    learnerName: String,
    learner: Learner,
    data: NoisyLabels.Data<Instance, Predictor, Label>,
    callback: ((ExperimentResult) -> ())?
  ) {
    // TODO: resetSeed()
    let filteredData = data.withMaxRedundancy(maxRedundancy)
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
      callbackSemaphore.wait()
      callback?(ExperimentResult(
        timeStamp: timeStamp,
        learner: learnerName,
        parameterType: .redundancy,
        parameter: maxRedundancy,
        metric: metric,
        value: value))
      callbackSemaphore.signal()
    }
  }
}

extension Experiment {
  public struct Learner {
    public let createFn: (NoisyLabels.Data<Instance, Predictor, Label>) -> NoisyLabels.Learner
    public let requiresFeatures: Bool
    public let supportsMultiThreading: Bool
  }
}

public func resultsWriter(at fileURL: URL) -> (ExperimentResult) -> () {
  if !FileManager.default.fileExists(atPath: fileURL.path) {
    let header = "timeStamp\tlearner\tparameterType\tparameter\tmetric\tvalue\n"
    FileManager.default.createFile(atPath: fileURL.path, contents: header.data(using: .utf8))
  }
  return { [fileURL] result in
    let resultParts = [
      "\(result.timeStamp)",
      "\(result.learner)",
      "\(result.parameterType)",
      "\(result.parameter)",
      "\(result.metric)",
      "\(result.value)"]
    let result = resultParts.joined(separator: "\t") + "\n"
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
