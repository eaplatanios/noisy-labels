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
import SwiftyBeaver
import Utility

let console: ConsoleDestination = ConsoleDestination()
let logger: SwiftyBeaver.Type = {
  console.format = "$DHH:mm:ss.SSS$d $L $M"
  SwiftyBeaver.addDestination(console)
  return SwiftyBeaver.self
}()

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
let dataset = parsedArguments.get(datasetArgument) ?? .rte // fatalError("No dataset was provided.")

func emLearner(_ data: NoisyLabels.Data<Int, String, Int>) -> Learner {
  let predictor = MinimaxConditionalEntropyPredictor(
    instanceCount: data.instances.count,
    predictorCount: data.predictors.count,
    labelCount: data.labels.count,
    avgLabelsPerPredictor: data.avgLabelsPerPredictor,
    avgLabelsPerItem: data.avgLabelsPerItem,
    gamma: 0.00)
  let optimizer = AMSGrad(
    for: predictor,
    learningRate: 1e-3,
    beta1: 0.9,
    beta2: 0.99,
    epsilon: 1e-8,
    decay: 0)
  let model = MultiLabelEMModel(
    predictor: predictor,
    optimizer: optimizer,
    instanceCount: data.instances.count,
    predictorCount: data.predictors.count,
    labelCount: data.labels.count,
    entropyWeight: 0.0,
    useSoftMajorityVote: true,
    useSoftPredictions: false)
  return EMLearner(
    for: model,
    randomSeed: 1234567890,
    batchSize: 1024,
    useWarmStarting: true,
    mStepCount: 1000,
    emStepCount: 10,
    marginalStepCount: 0,
    mStepLogCount: 100)
}

let experiment = try Experiment(
  dataDir: dataDir,
  dataset: dataset,
  usingFeatures: false,
  learners: [
    "MAJ": { _ in MajorityVoteLearner(useSoftMajorityVote: false) },
    "MAJ-S": { _ in MajorityVoteLearner(useSoftMajorityVote: true) },
    "MMCE-M (γ=0.00)": emLearner,
  ])
experiment.run()

// let resultsFile = resultsDir.appendingPathComponent("\(dataset.rawValue).csv")

// if !FileManager.default.fileExists(atPath: resultsDir.path) {
//   try FileManager.default.createDirectory(at: resultsDir)
// }
