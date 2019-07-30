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
import Logging
import NoisyLabels
import TensorFlow
import Utility

#if SNORKEL
import Python
#endif

let logger = Logger(label: "Noisy Labels Experiment")

enum Error: Swift.Error {
  case invalidCommand, datasetNotProvided, invalidDataset
}

enum Command: String {
  case run, plot
}

extension Command: StringEnumArgument {
  public static var completion: ShellCompletion {
    return .values([
      (Command.run.rawValue, "Runs an experiment."),
      (Command.plot.rawValue, "Generates plots with the results of an experiment.")
    ])
  }
}

let parser = ArgumentParser(
  usage: "<options>",
  overview: "This executable can be used to perform 'NoisyLabels' experiments.")
let commandArgument: PositionalArgument<Command> = parser.add(
  positional: "command",
  kind: Command.self,
  usage: "Experiment command to invoke. Can be either `run` or `plot`.")
let dataDirArgument: OptionArgument<PathArgument> = parser.add(
  option: "--data-dir",
  kind: PathArgument.self,
  usage: "Path to the data directory.")
let resultsDirArgument: OptionArgument<PathArgument> = parser.add(
  option: "--results-dir",
  kind: PathArgument.self,
  usage: "Path to the results directory.")
let datasetArgument: OptionArgument<String> = parser.add(
  option: "--dataset",
  kind: String.self,
  usage: "Dataset to use for this experiment.")

// The first argument is always the executable, and so we drop it.
let arguments = Array(ProcessInfo.processInfo.arguments.dropFirst())
let parsedArguments = try parser.parse(arguments)
let currentDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let dataDir: Foundation.URL = {
  if let argument = parsedArguments.get(dataDirArgument) {
    return URL(fileURLWithPath: argument.path.asString)
  }
  return currentDir.appendingPathComponent("temp/data")
}()
let resultsDir: Foundation.URL = {
  if let argument = parsedArguments.get(resultsDirArgument) {
    return URL(fileURLWithPath: argument.path.asString)
  }
  return currentDir.appendingPathComponent("temp/results")
}()

switch parsedArguments.get(commandArgument) {
case .plot:
  try FileManager.default.contentsOfDirectory(at: resultsDir, includingPropertiesForKeys: nil)
    .filter { !$0.hasDirectoryPath }
    .forEach { try ResultsPlotter(forFile: $0).plot() }
  exit(0)
case .run: ()
case _: throw Error.invalidCommand
}

let datasetName: String! = parsedArguments.get(datasetArgument)
if datasetName == nil { throw Error.datasetNotProvided }

func mmceLearner<Instance, Predictor, Label>(
  _ data: NoisyLabels.Data<Instance, Predictor, Label>,
  gamma: Float
) -> Learner {
  let predictor = MinimaxConditionalEntropyPredictor(data: data, gamma: gamma)
  let optimizer = Adam(
    for: predictor,
    learningRate: 1e-3,
    beta1: 0.9,
    beta2: 0.99,
    epsilon: 1e-8,
    decay: 0)
  let model = EMModel(
    predictor: predictor,
    optimizer: optimizer,
    entropyWeight: 0.0,
    useSoftMajorityVote: true,
    useSoftPredictions: false)
  return EMLearner(
    for: model,
    randomSeed: 42,
    batchSize: 128,
    useWarmStarting: true,
    mStepCount: 2000,
    emStepCount: 2,
    marginalStepCount: 0,
    mStepLogCount: 100,
    verbose: false)
}

func lnlLearner<Instance, Predictor, Label>(
  _ data: NoisyLabels.Data<Instance, Predictor, Label>,
  instanceEmbeddingSize: Int?,
  predictorEmbeddingSize: Int?,
  instanceHiddenUnitCounts: [Int],
  predictorHiddenUnitCounts: [Int],
  confusionLatentSize: Int,
  gamma: Float
) -> Learner {
  let predictor = LNLPredictor(
    data: data,
    instanceEmbeddingSize: instanceEmbeddingSize,
    predictorEmbeddingSize: predictorEmbeddingSize,
    instanceHiddenUnitCounts: instanceHiddenUnitCounts,
    predictorHiddenUnitCounts: predictorHiddenUnitCounts,
    confusionLatentSize: confusionLatentSize,
    gamma: gamma)
  let optimizer = Adam(
    for: predictor,
    learningRate: 1e-3,
    beta1: 0.9,
    beta2: 0.99,
    epsilon: 1e-8,
    decay: 0)
  let model = EMModel(
    predictor: predictor,
    optimizer: optimizer,
    entropyWeight: 0.0,
    useSoftMajorityVote: true,
    useSoftPredictions: false)
  return EMLearner(
    for: model,
    randomSeed: 42,
    batchSize: 128,
    useWarmStarting: true,
    mStepCount: 2000,
    emStepCount: 2,
    marginalStepCount: 2000,
    mStepLogCount: 100,
    verbose: false)
}

func learners<Dataset: NoisyLabelsExperiments.Dataset>()
-> [String: Experiment<Dataset>.Learner]
where Dataset.Loader.Predictor: Equatable {
  var learners: [String: Experiment<Dataset>.Learner] = [
    "MAJ": Experiment<Dataset>.Learner(
      createFn: { _ in MajorityVoteLearner(useSoftMajorityVote: false) },
      requiresFeatures: false,
      supportsMultiThreading: true),
    "MAJ-S": Experiment<Dataset>.Learner(
      createFn: { _ in MajorityVoteLearner(useSoftMajorityVote: true) },
      requiresFeatures: false,
      supportsMultiThreading: true),
    // "MMCE-M (γ=0.00)": Experiment<Dataset>.Learner(
    //   createFn: { data in mmceLearner(data, gamma: 0.00) },
    //   requiresFeatures: false,
    //   supportsMultiThreading: true),
    // "MMCE-M (γ=0.25)": Experiment<Dataset>.Learner(
    //   createFn: { data in mmceLearner(data, gamma: 0.25) },
    //   requiresFeatures: false,
    //   supportsMultiThreading: true),
    // "LNL-16-16-4x16-I-1 (γ=0.00)": Experiment<Dataset>.Learner(
    //   createFn: { data in
    //     lnlLearner(
    //       data,
    //       instanceEmbeddingSize: 16,
    //       predictorEmbeddingSize: 16,
    //       instanceHiddenUnitCounts: [16, 16, 16, 16],
    //       predictorHiddenUnitCounts: [],
    //       confusionLatentSize: 1,
    //       gamma: 0.00)
    //   },
    //   requiresFeatures: false,
    //   supportsMultiThreading: true),
    // "LNL-16-16-4x16-I-1 (γ=0.25)": Experiment<Dataset>.Learner(
    //   createFn: { data in
    //     lnlLearner(
    //       data,
    //       instanceEmbeddingSize: 16,
    //       predictorEmbeddingSize: 16,
    //       instanceHiddenUnitCounts: [16, 16, 16, 16],
    //       predictorHiddenUnitCounts: [],
    //       confusionLatentSize: 1,
    //       gamma: 0.25)
    //   },
    //   requiresFeatures: false,
    //   supportsMultiThreading: true),
    // "LNL-F-16-4x16-I-1 (γ=0.00)": Experiment<Dataset>.Learner(
    //   createFn: { data in
    //   lnlLearner(
    //     data,
    //     instanceEmbeddingSize: nil,
    //     predictorEmbeddingSize: 16,
    //     instanceHiddenUnitCounts: [16, 16, 16, 16],
    //     predictorHiddenUnitCounts: [],
    //     confusionLatentSize: 1,
    //     gamma: 0.00)
    //   },
    //   requiresFeatures: true,
    //   supportsMultiThreading: true),
    // "LNL-F-16-4x16-I-1 (γ=0.25)": Experiment<Dataset>.Learner(
    //   createFn: { data in
    //   lnlLearner(
    //     data,
    //     instanceEmbeddingSize: nil,
    //     predictorEmbeddingSize: 16,
    //     instanceHiddenUnitCounts: [16, 16, 16, 16],
    //     predictorHiddenUnitCounts: [],
    //     confusionLatentSize: 1,
    //     gamma: 0.25)
    //   },
    //   requiresFeatures: true,
    //   supportsMultiThreading: true)
  ]

// #if SNORKEL
//   learners["Snorkel"] = Experiment<Dataset>.Learner(
//     createFn: { _ in SnorkelLearner() },
//     requiresFeatures: false,
//     supportsMultiThreading: false)
// #endif

  return learners
}

func runExperiment<Dataset: NoisyLabelsExperiments.Dataset>(dataset: Dataset) throws
where Dataset.Loader.Predictor: Equatable {
  let experiment = try Experiment(dataDir: dataDir, dataset: dataset, learners: learners())
  let resultsURL = resultsDir.appendingPathComponent("\(dataset.description).tsv")
  let callback = try resultsWriter(at: resultsURL)
  experiment.run(
    callback: callback,
    runs: [
      .redundancy(maxRedundancy: 1, repetitionCount: 5),
      .redundancy(maxRedundancy: 2, repetitionCount: 5),
      .redundancy(maxRedundancy: 5, repetitionCount: 5),
      .redundancy(maxRedundancy: 10, repetitionCount: 5)])
}

switch datasetName {
case "rte": try runExperiment(dataset: RTEDataset())
case _: throw Error.invalidDataset
}
