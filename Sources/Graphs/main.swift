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
let data = try Data(loadFromDirectory: dataDirectory)

let predictor = DecoupledMLPPredictor(
  data: data,
  hiddenUnitCounts: [16],
  confusionLatentSize: 4)
let optimizer = Adam(
  for: predictor,
  learningRate: 1e-4,
  beta1: 0.9,
  beta2: 0.99,
  epsilon: 1e-8,
  decay: 0)
var model = Model(
  predictor: predictor,
  optimizer: optimizer,
  useSoftPredictions: true,
  entropyWeight: 0.01,
  randomSeed: 42,
  batchSize: 128,
  useWarmStarting: true,
  mStepCount: 1000,
  emStepCount: 100,
  mStepLogCount: 100,
  emStepCallback: { m in dump(evaluate(model: m, using: data)) },
  verbose: true)

dump(evaluate(model: model, using: data))
model.train(using: data)
