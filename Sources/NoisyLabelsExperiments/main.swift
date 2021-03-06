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
import SPMUtility
import TensorFlow

let logger = Logger(label: "Noisy Labels Experiment")

enum Error: Swift.Error {
  case invalidCommand, datasetNotProvided, invalidDataset
}

enum Command: String {
  case run, makeFigures, makeTables
}

extension Command: StringEnumArgument {
  public static var completion: ShellCompletion {
    .values([
      (Command.run.rawValue, "Runs an experiment."),
      (Command.makeFigures.rawValue, "Generates figures with the results of an experiment."),
      (Command.makeTables.rawValue, "Generates tables with the results of an experiment.")
    ])
  }
}

let parser = ArgumentParser(
  usage: "<options>",
  overview: "This executable can be used to run Noisy Labels experiments.")
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
let trainDataPortionArgument: OptionArgument<Float> = parser.add(
  option: "--train-data-portion",
  kind: Float.self,
  usage: "Proportion of the data (specifically, instances) to use for training.")
let syntheticPredictorsCountArgument: OptionArgument<Int> = parser.add(
  option: "--synthetic-predictors-count",
  kind: Int.self,
  usage: "Number of synthetic predictors to use.")
let useSyntheticPredictorFeaturesArgument: OptionArgument<Bool> = parser.add(
  option: "--use-synthetic-predictor-features",
  kind: Bool.self,
  usage: "Flag indicating whether to use the synthetic predictor features.")
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
    return URL(fileURLWithPath: argument.path.pathString)
  }
  return currentDir.appendingPathComponent("temp/data")
}()
let resultsDir: Foundation.URL = {
  if let argument = parsedArguments.get(resultsDirArgument) {
    return URL(fileURLWithPath: argument.path.pathString)
  }
  return currentDir.appendingPathComponent("temp/results")
}()
let trainDataPortion = parsedArguments.get(trainDataPortionArgument)
let syntheticPredictorsCount = parsedArguments.get(syntheticPredictorsCountArgument)
let useSyntheticPredictorFeatures = parsedArguments.get(useSyntheticPredictorFeaturesArgument) ?? false
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

let randomSeed: Int64 = 42
var generator = PhiloxRandomNumberGenerator(seed: randomSeed)

let datasetName: String! = parsedArguments.get(datasetArgument)
if datasetName == nil { throw Error.datasetNotProvided }

var currentTensorFlowRandomSeed = randomSeed
func tensorFlowRandomSeed() -> (graph: Int32, op: Int32) {
  let hash = (randomSeed &+ currentTensorFlowRandomSeed).bytes().sha512()
  currentTensorFlowRandomSeed += 1
  let graph = Int32(bytes: [hash[0], hash[1], hash[2], hash[3]], startingAt: 0)
  let op = Int32(bytes: [hash[4], hash[5], hash[6], hash[7]], startingAt: 0)
  return (graph: graph, op: op)
}

func mmceLearner<Instance, Predictor, Label>(
  _ data: NoisyLabels.Data<Instance, Predictor, Label>,
  gamma: Float
) -> Learner {
  withRandomSeedForTensorFlow(tensorFlowRandomSeed()) {
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
      useSoftPredictions: false,
      learningRateDecayFactor: 1.0)
    return EMLearner(
      for: model,
      randomSeed: randomSeed,
      batchSize: 128,
      useWarmStarting: true,
      mStepCount: 1000,
      emStepCount: 5,
      marginalStepCount: 0,
      mStepLogCount: 100,
      verbose: false)
  }
}

func twoStepMmceLearner<Instance, Predictor, Label>(
  _ data: NoisyLabels.Data<Instance, Predictor, Label>,
  instanceEmbeddingSize: Int?,
  predictorEmbeddingSize: Int?,
  instanceHiddenUnitCounts: [Int],
  predictorHiddenUnitCounts: [Int],
  confusionLatentSize: Int,
  gamma: Float
) -> Learner {
  withRandomSeedForTensorFlow(tensorFlowRandomSeed()) {
    let aggregationPredictor = MinimaxConditionalEntropyPredictor(data: data, gamma: 0.25)
    let aggregationOptimizer = Adam(
      for: aggregationPredictor,
      learningRate: 1e-3,
      beta1: 0.9,
      beta2: 0.99,
      epsilon: 1e-8,
      decay: 0)
    let aggregationModel = EMModel(
      predictor: aggregationPredictor,
      optimizer: aggregationOptimizer,
      entropyWeight: 0.0,
      useSoftPredictions: false,
      learningRateDecayFactor: 1.0)
    let aggregationLearner = EMLearner(
      for: aggregationModel,
      randomSeed: randomSeed,
      batchSize: 128,
      useWarmStarting: true,
      mStepCount: 1000,
      emStepCount: 5,
      marginalStepCount: 0,
      mStepLogCount: 100,
      verbose: false)
    let basePredictor = LNLPredictor(
      data: data,
      instanceEmbeddingSize: instanceEmbeddingSize,
      predictorEmbeddingSize: predictorEmbeddingSize,
      instanceHiddenUnitCounts: instanceHiddenUnitCounts,
      predictorHiddenUnitCounts: predictorHiddenUnitCounts,
      confusionLatentSize: confusionLatentSize,
      gamma: gamma)
    let baseOptimizer = Adam(
      for: basePredictor,
      learningRate: 1e-3,
      beta1: 0.9,
      beta2: 0.99,
      epsilon: 1e-8,
      decay: 0)
    let baseModel = EMModel(
      predictor: basePredictor,
      optimizer: baseOptimizer,
      entropyWeight: 0.01,
      useSoftPredictions: true,
      learningRateDecayFactor: 0.995)
    let baseLearner = EMLearner(
      for: baseModel,
      randomSeed: randomSeed,
      batchSize: 512,
      useWarmStarting: true,
      mStepCount: 1000,
      emStepCount: 1,
      marginalStepCount: 0,
      mStepLogCount: 100,
      verbose: true)
    return TwoStepLearner(
      aggregationLearner: aggregationLearner,
      baseLearner: baseLearner,
      verbose: true)
  }
}

func twoStepMmceFeaturizedLearner<Instance, Predictor, Label>(
  _ data: NoisyLabels.Data<Instance, Predictor, Label>,
  predictorEmbeddingSize: Int,
  instanceHiddenUnitCounts: [Int],
  predictorHiddenUnitCounts: [Int],
  confusionLatentSize: Int,
  gamma: Float
) -> Learner {
  withRandomSeedForTensorFlow(tensorFlowRandomSeed()) {
    let aggregationPredictor = MinimaxConditionalEntropyPredictor(data: data, gamma: 0.25)
    let aggregationOptimizer = Adam(
      for: aggregationPredictor,
      learningRate: 1e-3,
      beta1: 0.9,
      beta2: 0.99,
      epsilon: 1e-8,
      decay: 0)
    let aggregationModel = EMModel(
      predictor: aggregationPredictor,
      optimizer: aggregationOptimizer,
      entropyWeight: 0.0,
      useSoftPredictions: false,
      learningRateDecayFactor: 1.0)
    let aggregationLearner = EMLearner(
      for: aggregationModel,
      randomSeed: randomSeed,
      batchSize: 128,
      useWarmStarting: true,
      mStepCount: 1000,
      emStepCount: 5,
      marginalStepCount: 0,
      mStepLogCount: 100,
      verbose: false)
    let basePredictor = FeaturizedLNLPredictor(
      data: data,
      predictorEmbeddingSize: predictorEmbeddingSize,
      instanceHiddenUnitCounts: instanceHiddenUnitCounts,
      predictorHiddenUnitCounts: predictorHiddenUnitCounts,
      confusionLatentSize: confusionLatentSize,
      gamma: gamma)
    let baseOptimizer = Adam(
      for: basePredictor,
      learningRate: 1e-3,
      beta1: 0.9,
      beta2: 0.99,
      epsilon: 1e-8,
      decay: 0)
    let baseModel = EMModel(
      predictor: basePredictor,
      optimizer: baseOptimizer,
      entropyWeight: 0.01,
      useSoftPredictions: true,
      learningRateDecayFactor: 0.995)
    let baseLearner = EMLearner(
      for: baseModel,
      randomSeed: randomSeed,
      batchSize: 512,
      useWarmStarting: true,
      mStepCount: 1000,
      emStepCount: 1,
      marginalStepCount: 0,
      mStepLogCount: 100,
      verbose: true)
    return TwoStepLearner(
      aggregationLearner: aggregationLearner,
      baseLearner: baseLearner,
      verbose: true)
  }
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
  withRandomSeedForTensorFlow(tensorFlowRandomSeed()) {
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
      learningRate: 1e-4,
      beta1: 0.9,
      beta2: 0.99,
      epsilon: 1e-8,
      decay: 0)
    let model = EMModel(
      predictor: predictor,
      optimizer: optimizer,
      entropyWeight: 0.01,
      useSoftPredictions: true,
      learningRateDecayFactor: 0.995)
    return EMLearner(
      for: model,
      randomSeed: randomSeed,
      batchSize: 512,
      useWarmStarting: true,
      mStepCount: 1000,
      emStepCount: 2,
      marginalStepCount: 1000,
      mStepLogCount: 100,
      verbose: true)
  }
}

func featurizedLNLLearner<Instance, Predictor, Label>(
  _ data: NoisyLabels.Data<Instance, Predictor, Label>,
  predictorEmbeddingSize: Int,
  instanceHiddenUnitCounts: [Int],
  predictorHiddenUnitCounts: [Int],
  confusionLatentSize: Int,
  gamma: Float
) -> Learner {
  withRandomSeedForTensorFlow(tensorFlowRandomSeed()) {
    let predictor = FeaturizedLNLPredictor(
      data: data,
      predictorEmbeddingSize: predictorEmbeddingSize,
      instanceHiddenUnitCounts: instanceHiddenUnitCounts,
      predictorHiddenUnitCounts: predictorHiddenUnitCounts,
      confusionLatentSize: confusionLatentSize,
      gamma: gamma)
    let optimizer = Adam(
      for: predictor,
      learningRate: 1e-4,
      beta1: 0.9,
      beta2: 0.99,
      epsilon: 1e-8,
      decay: 0)
    let model = EMModel(
      predictor: predictor,
      optimizer: optimizer,
      entropyWeight: 0.0,
      useSoftPredictions: true,
      learningRateDecayFactor: 1.0)
    return EMLearner(
      for: model,
      randomSeed: randomSeed,
      batchSize: 128,
      useWarmStarting: true,
      mStepCount: 1000,
      emStepCount: 2,
      marginalStepCount: 1000,
      mStepLogCount: 100,
      verbose: true)
  }
}

func fullyFeaturizedLNLLearner<Instance, Predictor, Label>(
  _ data: NoisyLabels.Data<Instance, Predictor, Label>,
  instanceHiddenUnitCounts: [Int],
  predictorHiddenUnitCounts: [Int],
  confusionLatentSize: Int,
  gamma: Float
) -> Learner {
  withRandomSeedForTensorFlow(tensorFlowRandomSeed()) {
    let predictor = FullyFeaturizedLNLPredictor(
      data: data,
      instanceHiddenUnitCounts: instanceHiddenUnitCounts,
      predictorHiddenUnitCounts: predictorHiddenUnitCounts,
      confusionLatentSize: confusionLatentSize,
      gamma: gamma)
    let optimizer = Adam(
      for: predictor,
      learningRate: 1e-4,
      beta1: 0.9,
      beta2: 0.99,
      epsilon: 1e-8,
      decay: 0)
    let model = EMModel(
      predictor: predictor,
      optimizer: optimizer,
      entropyWeight: 0.0,
      useSoftPredictions: true,
      learningRateDecayFactor: 1.0)
    return EMLearner(
      for: model,
      randomSeed: randomSeed,
      batchSize: 128,
      useWarmStarting: true,
      mStepCount: 1000,
      emStepCount: 2,
      marginalStepCount: 1000,
      mStepLogCount: 100,
      verbose: true)
  }
}

func decoupledLNLLearner<Instance, Predictor, Label>(
  _ data: NoisyLabels.Data<Instance, Predictor, Label>,
  predictorEmbeddingSize: Int,
  instanceLHiddenUnitCounts: [Int],
  instanceQHiddenUnitCounts: [Int],
  predictorHiddenUnitCounts: [Int],
  confusionLatentSize: Int,
  gamma: Float
) -> Learner {
  withRandomSeedForTensorFlow(tensorFlowRandomSeed()) {
    let predictor = DecoupledLNLPredictor(
      data: data,
      predictorEmbeddingSize: predictorEmbeddingSize,
      instanceLHiddenUnitCounts: instanceLHiddenUnitCounts,
      instanceQHiddenUnitCounts: instanceQHiddenUnitCounts,
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
      entropyWeight: 0.01,
      useSoftPredictions: true,
      learningRateDecayFactor: 1.0)
    return EMLearner(
      for: model,
      randomSeed: randomSeed,
      batchSize: 128,
      useWarmStarting: true,
      mStepCount: 1000,
      emStepCount: 5,
      marginalStepCount: 0,
      mStepLogCount: 100,
      verbose: true)
  }
}

func fullyDecoupledLNLLearner<Instance, Predictor, Label>(
  _ data: NoisyLabels.Data<Instance, Predictor, Label>,
  instanceLHiddenUnitCounts: [Int],
  instanceQHiddenUnitCounts: [Int],
  predictorHiddenUnitCounts: [Int],
  confusionLatentSize: Int,
  gamma: Float
) -> Learner {
  withRandomSeedForTensorFlow(tensorFlowRandomSeed()) {
    let predictor = FullyDecoupledLNLPredictor(
      data: data,
      instanceLHiddenUnitCounts: instanceLHiddenUnitCounts,
      instanceQHiddenUnitCounts: instanceQHiddenUnitCounts,
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
      entropyWeight: 0.01,
      useSoftPredictions: true,
      learningRateDecayFactor: 1.0)
    return EMLearner(
      for: model,
      randomSeed: randomSeed,
      batchSize: 128,
      useWarmStarting: true,
      mStepCount: 1000,
      emStepCount: 5,
      marginalStepCount: 0,
      mStepLogCount: 100,
      verbose: true)
  }
}

func learners<Dataset: NoisyLabelsExperiments.Dataset>() -> [(String, Experiment<Dataset>.Learner)]
where Dataset.Loader.Predictor: Equatable {
  var learners: [(String, Experiment<Dataset>.Learner)] = [
    ("MAJ", Experiment<Dataset>.Learner(
      createFn: { _ in MajorityVoteLearner(useSoftMajorityVote: false) },
      requiresFeatures: false,
      supportsMultiThreading: true)),
//    ("MMCE", Experiment<Dataset>.Learner(
//      createFn: { data in mmceLearner(data, gamma: 0.25) },
//      requiresFeatures: false,
//      supportsMultiThreading: true)),
//    ("MMCE-ME", Experiment<Dataset>.Learner(
//      createFn: { data in
//        twoStepMmceLearner(
//          data,
//          instanceEmbeddingSize: 512,
//          predictorEmbeddingSize: 512,
//          instanceHiddenUnitCounts: [512],
//          predictorHiddenUnitCounts: [],
//          confusionLatentSize: 1,
//          gamma: 0.00)
//      },
//      requiresFeatures: true,
//      supportsMultiThreading: true)),
   ("MMCE-M", Experiment<Dataset>.Learner(
     createFn: { data in
       twoStepMmceFeaturizedLearner(
         data,
         predictorEmbeddingSize: 32,
         instanceHiddenUnitCounts: [32, 32, 32, 32],
         predictorHiddenUnitCounts: [32, 32, 32, 32],
         confusionLatentSize: 1,
         gamma: 0)
     },
     requiresFeatures: true,
     supportsMultiThreading: true)),
//    ("LNL-E", Experiment<Dataset>.Learner(
//      createFn: { data in
//        lnlLearner(
//          data,
//          instanceEmbeddingSize: 512,
//          predictorEmbeddingSize: 512,
//          instanceHiddenUnitCounts: [512],
//          predictorHiddenUnitCounts: [],
//          confusionLatentSize: 1,
//          gamma: 0.00)
//      },
//      requiresFeatures: false,
//      supportsMultiThreading: true)),
    ("LNL", Experiment<Dataset>.Learner(
      createFn: { data in
        featurizedLNLLearner(
          data,
          predictorEmbeddingSize: 32,
          instanceHiddenUnitCounts: [32, 32, 32, 32],
          predictorHiddenUnitCounts: [32, 32, 32, 32],
          confusionLatentSize: 1,
          gamma: 0.1)
      },
      requiresFeatures: true,
      supportsMultiThreading: true)),
    // ("FullLNL", Experiment<Dataset>.Learner(
    //   createFn: { data in
    //     fullyFeaturizedLNLLearner(
    //       data,
    //       instanceHiddenUnitCounts: [16, 16, 16, 16],
    //       predictorHiddenUnitCounts: [16, 16, 16, 16],
    //       confusionLatentSize: 1,
    //       gamma: 0.1)
    //   },
    //   requiresFeatures: true,
    //   supportsMultiThreading: true))
  ]

#if SNORKEL
  learners["Snorkel"] = Experiment<Dataset>.Learner(
    createFn: { _ in SnorkelLearner() },
    requiresFeatures: false,
    supportsMultiThreading: false)
  learners["MeTaL"] = Experiment<Dataset>.Learner(
    createFn: { _ in MetalLearner(randomSeed: randomSeed) },
    requiresFeatures: false,
    supportsMultiThreading: false)
#endif

  return learners
}

func runExperiment<Dataset: NoisyLabelsExperiments.Dataset>(dataset: Dataset) throws
where Dataset.Loader.Predictor == String {
  let experiment = try withRandomSeedForTensorFlow(tensorFlowRandomSeed()) {
    try Experiment(
      dataDir: dataDir,
      dataset: dataset,
      trainDataPortion: trainDataPortion,
      syntheticPredictorsCount: syntheticPredictorsCount,
      useSyntheticPredictorFeatures: useSyntheticPredictorFeatures,
      learners: learners(),
      using: &generator)
  }
  let resultsURL = resultsDir.appendingPathComponent("\(dataset.description).tsv")
  let callback = try resultsWriter(at: resultsURL)
  experiment.run(
    callback: callback,
    runs: [
      // .redundancy(maxRedundancy: 1, repetitionCount: 1),
      // .redundancy(maxRedundancy: 2, repetitionCount: 3),
      // .redundancy(maxRedundancy: 5, repetitionCount: 3),
      .redundancy(maxRedundancy: 10, repetitionCount: 1),
      // .redundancy(maxRedundancy: 20, repetitionCount: 3),
      // .redundancy(maxRedundancy: 40, repetitionCount: 1),
      // .predictorSubsampling(predictorCount: 100, repetitionCount: 1),
    ],
    parallelismLimit: parallelismLimit,
    using: &generator)
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
