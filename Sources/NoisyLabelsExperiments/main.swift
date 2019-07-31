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

let logger = Logger(label: "Noisy Labels Experiment")

enum Error: Swift.Error {
  case invalidCommand, datasetNotProvided, invalidDataset
}

enum Command: String {
  case run, makeFigures, makeTables
}

extension Command: StringEnumArgument {
  public static var completion: ShellCompletion {
    return .values([
      (Command.run.rawValue, "Runs an experiment."),
      (Command.makeFigures.rawValue, "Generates figures with the results of an experiment."),
      (Command.makeTables.rawValue, "Generates tables with the results of an experiment.")
    ])
  }
}

let parser = ArgumentParser(
  usage: "<options>",
  overview: "This executable can be used to perform 'NoisyLabels' experiments.")
let commandArgument: PositionalArgument<Command> = parser.add(
  positional: "command",
  kind: Command.self,
  usage: "Experiment command to invoke. Can be `run`, `makeFigures`, or `makeTables`.")
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
let parallelismArgument: OptionArgument<Int> = parser.add(
  option: "--parallelism",
  kind: Int.self,
  usage: "Parallelism limit to enforce while running experiments.")

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
let parallelismLimit = parsedArguments.get(parallelismArgument)

switch parsedArguments.get(commandArgument) {
case .makeFigures:
  try FileManager.default.contentsOfDirectory(at: resultsDir, includingPropertiesForKeys: nil)
    .filter { !$0.hasDirectoryPath && $0.pathExtension == "tsv" }
    .forEach { try ResultsPlotter(forFile: $0).plot() }
  exit(0)
case .makeTables:
  try FileManager.default.contentsOfDirectory(at: resultsDir, includingPropertiesForKeys: nil)
    .filter { !$0.hasDirectoryPath && $0.pathExtension == "tsv" }
    .forEach { try ResultsPrinter(forFile: $0).makeTables() }
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
    useSoftMajorityVote: false,
    useSoftPredictions: false,
    learningRateDecayFactor: 1.0)
  return EMLearner(
    for: model,
    randomSeed: 42,
    batchSize: 128,
    useWarmStarting: true,
    mStepCount: 1000,
    emStepCount: 5,
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
    useSoftPredictions: true,
    learningRateDecayFactor: 0.997)
  return EMLearner(
    for: model,
    randomSeed: 42,
    batchSize: 128,
    useWarmStarting: true,
    mStepCount: 1000,
    emStepCount: 10,
    marginalStepCount: 0,
    mStepLogCount: 100,
    verbose: false)
}

// Per-dataset configurations:
//   BlueBirds:
//     - LNL-IE-16-IH-[16, 16, 16, 16]-PE-16-PH-[] (gam=0.00)
//     - LNL-IE-0-IH-[16, 16, 16, 16]-PE-16-PH-[] (gam=0.00)
//   Word Similarity:
//     - LNL-IE-16-IH-[4]-PE-16-PH-[] (gam=0.00)
//     - LNL-IE-0-IH-[4]-PE-16-PH-[] (gam=0.00)

func learners<Dataset: NoisyLabelsExperiments.Dataset>()
-> [String: Experiment<Dataset>.Learner]
where Dataset.Loader.Predictor: Equatable {
  var learners: [String: Experiment<Dataset>.Learner] = [
    "MAJ": Experiment<Dataset>.Learner(
      createFn: { _ in MajorityVoteLearner(useSoftMajorityVote: false) },
      requiresFeatures: false,
      supportsMultiThreading: true),
    "MMCE": Experiment<Dataset>.Learner(
      createFn: { data in mmceLearner(data, gamma: 0.25) },
      requiresFeatures: false,
      supportsMultiThreading: true),
    "LNL-E": Experiment<Dataset>.Learner(
      createFn: { data in
        lnlLearner(
          data,
          instanceEmbeddingSize: 16,
          predictorEmbeddingSize: 16,
          instanceHiddenUnitCounts: [16, 16, 16, 16],
          predictorHiddenUnitCounts: [],
          confusionLatentSize: 1,
          gamma: 0.00)
      },
      requiresFeatures: false,
      supportsMultiThreading: true),
    "LNL": Experiment<Dataset>.Learner(
      createFn: { data in
      lnlLearner(
        data,
        instanceEmbeddingSize: nil,
        predictorEmbeddingSize: 16,
        instanceHiddenUnitCounts: [16, 16, 16, 16],
        predictorHiddenUnitCounts: [],
        confusionLatentSize: 1,
        gamma: 0.00)
      },
      requiresFeatures: true,
      supportsMultiThreading: true)
  ]

#if SNORKEL
  learners["Snorkel"] = Experiment<Dataset>.Learner(
    createFn: { _ in SnorkelLearner() },
    requiresFeatures: false,
    supportsMultiThreading: false)
  learners["MeTaL"] = Experiment<Dataset>.Learner(
    createFn: { _ in MetalLearner(randomSeed: 42) },
    requiresFeatures: false,
    supportsMultiThreading: false)
#endif

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
      .redundancy(maxRedundancy: 1, repetitionCount: 20),
      .redundancy(maxRedundancy: 2, repetitionCount: 20),
      .redundancy(maxRedundancy: 4, repetitionCount: 20),
      .redundancy(maxRedundancy: 6, repetitionCount: 20),
      .redundancy(maxRedundancy: 8, repetitionCount: 20),
      .redundancy(maxRedundancy: 10, repetitionCount: 20),
      // .redundancy(maxRedundancy: 20, repetitionCount: 20),
      // .redundancy(maxRedundancy: 40, repetitionCount: 20),
    ],
    parallelismLimit: parallelismLimit)
}

switch datasetName {
case "bluebirds": try runExperiment(dataset: BlueBirdsDataset())
case "word-similarity": try runExperiment(dataset: WordSimilarityDataset(features: .glove))
case "rte": try runExperiment(dataset: RTEDataset())
case "age": try runExperiment(dataset: AgeDataset())
case "sentiment-popularity": try runExperiment(dataset: SentimentPopularityDataset())
case "weather-sentiment": try runExperiment(dataset: WeatherSentimentDataset())
case "medical-treats": try runExperiment(dataset: MedicalTreatsDataset())
case "medical-causes": try runExperiment(dataset: MedicalCausesDataset())
case _: throw Error.invalidDataset
}
