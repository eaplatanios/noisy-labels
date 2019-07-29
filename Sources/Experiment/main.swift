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

let parser = ArgumentParser(
  usage: "<options>",
  overview: "This executable can be used to run 'NoisyLabels' experiments.")
let dataDirArgument: OptionArgument<PathArgument> = parser.add(
  option: "--data-dir",
  kind: PathArgument.self,
  usage: "Path to the data directory.")
let resultsDirArgument: OptionArgument<PathArgument> = parser.add(
  option: "--results-dir",
  kind: PathArgument.self,
  usage: "Path to the results directory.")
let datasetArgument: OptionArgument<Dataset> = parser.add(
  option: "--dataset",
  kind: Dataset.self,
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
let dataset = parsedArguments.get(datasetArgument)!

func mmceLearner(_ data: NoisyLabels.Data<Int, String, Int>, gamma: Float) -> Learner {
  let predictor = MinimaxConditionalEntropyPredictor(data: data, gamma: gamma)
  let optimizer = Adam(
    for: predictor,
    learningRate: 1e-3,
    beta1: 0.9,
    beta2: 0.99,
    epsilon: 1e-8,
    decay: 0)
  let model = MultiLabelEMModel(
    predictor: predictor,
    optimizer: optimizer,
    entropyWeight: 1.0,
    useSoftMajorityVote: true,
    useSoftPredictions: false)
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

func lnlLearner(
  _ data: NoisyLabels.Data<Int, String, Int>,
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
  let model = MultiLabelEMModel(
    predictor: predictor,
    optimizer: optimizer,
    entropyWeight: 1.0,
    useSoftMajorityVote: true,
    useSoftPredictions: false)
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

let experiment = try Experiment(
  dataDir: dataDir,
  dataset: dataset,
  usingFeatures: false,
  learners: [
    // "MAJ": { _ in MajorityVoteLearner(useSoftMajorityVote: false) },
    // "MAJ-S": { _ in MajorityVoteLearner(useSoftMajorityVote: true) },
    // "MMCE-M (γ=0.00)": { data in mmceLearner(data, gamma: 0.00) },
    // "MMCE-M (γ=0.25)": { data in mmceLearner(data, gamma: 0.25) },
    "LNL-4-4-4x16-4x16-1 (γ=0.00)": { data in
      lnlLearner(
        data,
        instanceEmbeddingSize: 4,
        predictorEmbeddingSize: 4,
        instanceHiddenUnitCounts: [16, 16, 16, 16],
        predictorHiddenUnitCounts: [16, 16, 16, 16],
        confusionLatentSize: 1,
        gamma: 0.00)
    },
    "LNL-4-4-4x16-4x16-1 (γ=0.25)": { data in
      lnlLearner(
        data,
        instanceEmbeddingSize: 4,
        predictorEmbeddingSize: 4,
        instanceHiddenUnitCounts: [16, 16, 16, 16],
        predictorHiddenUnitCounts: [16, 16, 16, 16],
        confusionLatentSize: 1,
        gamma: 0.25)
    },
    "LNL-IF-4-4x16-4x16-1 (γ=0.00)": { data in
      lnlLearner(
        data,
        instanceEmbeddingSize: nil,
        predictorEmbeddingSize: 4,
        instanceHiddenUnitCounts: [16, 16, 16, 16],
        predictorHiddenUnitCounts: [16, 16, 16, 16],
        confusionLatentSize: 1,
        gamma: 0.00)
    },
    "LNL-IF-4-4x16-4x16-1 (γ=0.25)": { data in
      lnlLearner(
        data,
        instanceEmbeddingSize: nil,
        predictorEmbeddingSize: 4,
        instanceHiddenUnitCounts: [16, 16, 16, 16],
        predictorHiddenUnitCounts: [16, 16, 16, 16],
        confusionLatentSize: 1,
        gamma: 0.25)
    }
  ])
let results = experiment.run()

let resultsFile = resultsDir.appendingPathComponent("\(dataset.rawValue).csv")
if !FileManager.default.fileExists(atPath: resultsDir.path) {
  try FileManager.default.createDirectory(at: resultsDir, withIntermediateDirectories: true)
}
try results.json(pretty: true).write(to: resultsFile, atomically: false, encoding: .utf8)
