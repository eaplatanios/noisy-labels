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
import TensorFlow

let workingDirectory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
  .appendingPathComponent("temp")
let dataDirectory = workingDirectory.appendingPathComponent("data").appendingPathComponent("cora")

let graph = try Graph(loadFromDirectory: dataDirectory)
// let predictor = MLPPredictor(
//   graph: graph,
//   hiddenUnitCounts: [128],
//   confusionLatentSize: 4)
let predictor = DecoupledGCNPredictor(
  graph: graph,
  lHiddenUnitCounts: [128],
  qHiddenUnitCounts: [128])
let optimizerFn = { () in
  Adam(
    for: predictor,
    learningRate: 1e-3,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
    decay: 0)
}
var model = Model(
  predictor: predictor,
  optimizerFn: optimizerFn,
  entropyWeight: 0,
  qualitiesRegularizationWeight: 0,
  randomSeed: 42,
  batchSize: 128,
  useWarmStarting: false,
  mStepCount: 1000,
  emStepCount: 5,
  marginalStepCount: 1000,
  evaluationStepCount: 1,
  mStepLogCount: 100,
  emStepCallback: { dump(evaluate(model: $0, using: graph, usePrior: true)) },
  verbose: true)

dump(evaluate(model: model, using: graph))
model.train(using: graph)
