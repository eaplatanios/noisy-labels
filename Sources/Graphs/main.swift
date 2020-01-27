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
let predictor = MLPPredictor(
   graph: graph,
   hiddenUnitCounts: [128],
   confusionLatentSize: 1)
// let predictor = GCNPredictor(
//    graph: graph,
//    hiddenUnitCounts: [1024],
//    confusionLatentSize: 1)
// let predictor = DecoupledGCNPredictor(
//   graph: graph,
//   lHiddenUnitCounts: [128],
//   qHiddenUnitCounts: [128],
//   confusionLatentSize: 1)

let optimizerFn = { () in
  // RProp(for: predictor)
  Adam(
    for: predictor,
    learningRate: 1e-2,
    beta1: 0.9,
    beta2: 0.99,
    epsilon: 1e-8,
    decay: 0.01)
}

var bestEvaluationResult: Result? = nil
var bestPriorEvaluationResult: Result? = nil
var emStepCallbackInvocationsWithoutImprovement = 0
var emStepCallbackInvocationsWithoutPriorImprovement = 0
func emStepCallback<P: GraphPredictor, O: Optimizer>(model: Model<P, O>) {
  let evaluationResult = evaluate(model: model, using: graph, usePrior: false)
  if let bestResult = bestEvaluationResult {
    if evaluationResult.validationAccuracy > bestResult.validationAccuracy ||
      (evaluationResult.validationAccuracy == bestResult.validationAccuracy &&
       evaluationResult.testAccuracy > bestResult.testAccuracy) {
      emStepCallbackInvocationsWithoutImprovement = 0
      bestEvaluationResult = evaluationResult
    } else {
      emStepCallbackInvocationsWithoutImprovement += 1
    }
  } else {
    bestEvaluationResult = evaluationResult
  }
  let priorEvaluationResult = evaluate(model: model, using: graph, usePrior: true)
  if let bestResult = bestPriorEvaluationResult {
    if priorEvaluationResult.validationAccuracy > bestResult.validationAccuracy ||
      (priorEvaluationResult.validationAccuracy == bestResult.validationAccuracy &&
       priorEvaluationResult.testAccuracy > bestResult.testAccuracy) {
      emStepCallbackInvocationsWithoutPriorImprovement = 0
      bestPriorEvaluationResult = priorEvaluationResult
    } else {
      emStepCallbackInvocationsWithoutPriorImprovement += 1
    }
  } else {
    bestPriorEvaluationResult = priorEvaluationResult
  }
  logger.info("Current Evaluation Result: \(evaluationResult)")
  logger.info("Current Prior Evaluation Result: \(priorEvaluationResult)")
  logger.info("Best Evaluation Result: \(bestEvaluationResult)")
  logger.info("Best Prior Evaluation Result: \(bestPriorEvaluationResult)")
  if emStepCallbackInvocationsWithoutImprovement > 0 {
    logger.info("Evaluation result has not improved in \(emStepCallbackInvocationsWithoutImprovement) EM-step callback invocations.")
  }
  if emStepCallbackInvocationsWithoutPriorImprovement > 0 {
    logger.info("Prior evaluation result has not improved in \(emStepCallbackInvocationsWithoutPriorImprovement) EM-step callback invocations.")
  }
}

var model = Model(
  predictor: predictor,
  optimizerFn: optimizerFn,
  entropyWeight: 0,
  qualitiesRegularizationWeight: 0,
  randomSeed: 42,
  batchSize: 128,
  useWarmStarting: false,
  useThresholdedExpectations: false,
  labelSmoothing: 0.5,
  // resultAccumulator: MovingAverageAccumulator(weight: 0.5),
  mStepCount: 1000,
  emStepCount: 100,
  marginalStepCount: 1000,
  evaluationStepCount: 1,
  mStepLogCount: 100,
  mConvergenceEvaluationCount: 100,
  emStepCallback: { emStepCallback(model: $0) },
  verbose: true)

dump(bestEvaluationResult)
model.train(using: graph)
