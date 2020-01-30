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
import SPMUtility
import TensorFlow

// The first argument is always the executable, and so we drop it.
let arguments = Array(ProcessInfo.processInfo.arguments.dropFirst())

let parser = ArgumentParser(
  usage: "<options>",
  overview: "TODO")
let dataset: OptionArgument<String> = parser.add(
  option: "--dataset",
  shortName: "-d",
  kind: String.self,
  usage: "Dataset to use. Can be one of: 'cora', 'citeseer', 'pubmed', and 'disease'.")
let model: OptionArgument<ModelName> = parser.add(
  option: "--model",
  shortName: "-m",
  kind: ModelName.self,
  usage: "Model to use. Can be one of: 'mlp', 'decoupled-mlp', 'gcn', and 'decoupled-gcn'.")
let lHiddenUnitCounts: OptionArgument<[Int]> = parser.add(
  option: "--l-hidden",
  shortName: "-lh",
  kind: [Int].self,
  usage: "Label hidden unit counts.")
let qHiddenUnitCounts: OptionArgument<[Int]> = parser.add(
  option: "--q-hidden",
  shortName: "-lq",
  kind: [Int].self,
  usage: "Quality hidden unit counts.")
let batchSize: OptionArgument<Int> = parser.add(
  option: "--batch-size",
  shortName: "-bs",
  kind: Int.self,
  usage: "Batch size.")
let labelSmoothing: OptionArgument<Float> = parser.add(
  option: "--label-smoothing",
  shortName: "-ls",
  kind: Float.self,
  usage: "Label smoothing factor.")
let dropout: OptionArgument<Float> = parser.add(
  option: "--dropout-rate",
  shortName: "-dr",
  kind: Float.self,
  usage: "Dropout rate.")
let targetBadEdgeProportion: OptionArgument<Float> = parser.add(
  option: "--target-bad-edge-proportion",
  shortName: "-tbep",
  kind: Float.self,
  usage: "Target bad edge proportion.")
let seed: OptionArgument<Int> = parser.add(
  option: "--seed",
  shortName: "-s",
  kind: Int.self,
  usage: "Random seed.")

let parsedArguments = try! parser.parse(arguments)

let randomSeed = Int64(parsedArguments.get(seed) ?? 123456789)
var generator = PhiloxRandomNumberGenerator(seed: randomSeed)
try withRandomSeedForTensorFlow(randomSeed) {
  let workingDirectory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    .appendingPathComponent("temp")
  let dataDirectory = workingDirectory
    .appendingPathComponent("data")
    .appendingPathComponent(parsedArguments.get(dataset)!)
  var graph = try Graph(loadFromDirectory: dataDirectory)

  if let targetBadEdgeProportion = parsedArguments.get(targetBadEdgeProportion) {
    graph = graph.corrupted(
      targetBadEdgeProportion: targetBadEdgeProportion,
      using: &generator)
  }

  func runExperiment<Predictor: GraphPredictor>(predictor: Predictor)
  where Predictor.TangentVector: VectorProtocol & PointwiseMultiplicative & ElementaryFunctions,
        Predictor.TangentVector.VectorSpaceScalar == Float {
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
      logger.info("Configuration: \(configuration())")
      logger.info("Current Evaluation Result: \(evaluationResult)")
      logger.info("Current Prior Evaluation Result: \(priorEvaluationResult)")
      logger.info("Best Evaluation Result: \(String(describing: bestEvaluationResult))")
      logger.info("Best Prior Evaluation Result: \(String(describing: bestPriorEvaluationResult))")
      if emStepCallbackInvocationsWithoutImprovement > 0 {
        logger.info("Evaluation result has not improved in \(emStepCallbackInvocationsWithoutImprovement) EM-step callback invocations.")
      }
      if emStepCallbackInvocationsWithoutPriorImprovement > 0 {
        logger.info("Prior evaluation result has not improved in \(emStepCallbackInvocationsWithoutPriorImprovement) EM-step callback invocations.")
      }
    }

    let optimizerFn = { () in
      Adam<Predictor>(
        for: predictor,
        learningRate: 1e-2,
        beta1: 0.9,
        beta2: 0.99,
        epsilon: 1e-8,
        decay: 0)
    }

    var model = Model(
      predictor: predictor,
      optimizerFn: optimizerFn,
      entropyWeight: 0,
      qualitiesRegularizationWeight: 0,
      randomSeed: 42,
      batchSize: parsedArguments.get(batchSize) ?? 128,
      useWarmStarting: false,
      useThresholdedExpectations: false,
      labelSmoothing: parsedArguments.get(labelSmoothing) ?? 0.5,
      // resultAccumulator: MovingAverageAccumulator(weight: 0.5),
      mStepCount: 1000,
      emStepCount: 100,
      marginalStepCount: 1000,
      evaluationStepCount: 1,
      mStepLogCount: 100,
      mConvergenceEvaluationCount: 500,
      emStepCallback: { emStepCallback(model: $0) },
      verbose: true)

    dump(bestEvaluationResult)
    model.train(using: graph)
  }

  switch parsedArguments.get(model)! {
  case .mlp: runExperiment(predictor: MLPPredictor(
    graph: graph,
    hiddenUnitCounts: parsedArguments.get(lHiddenUnitCounts)!,
    dropout: parsedArguments.get(dropout) ?? 0.5))
  case .decoupledMLP: runExperiment(predictor: DecoupledMLPPredictorV2(
    graph: graph,
    lHiddenUnitCounts: parsedArguments.get(lHiddenUnitCounts)!,
    qHiddenUnitCounts: parsedArguments.get(qHiddenUnitCounts)!,
    dropout: parsedArguments.get(dropout) ?? 0.5))
  case .gcn: runExperiment(predictor: GCNPredictor(
    graph: graph,
    hiddenUnitCounts: parsedArguments.get(lHiddenUnitCounts)!,
    dropout: parsedArguments.get(dropout) ?? 0.5))
  case .decoupledGCN: runExperiment(predictor: DecoupledGCNPredictorV2(
    graph: graph,
    lHiddenUnitCounts: parsedArguments.get(lHiddenUnitCounts)!,
    qHiddenUnitCounts: parsedArguments.get(qHiddenUnitCounts)!,
    dropout: parsedArguments.get(dropout) ?? 0.5))
  }

  func configuration() -> String {
    var configuration = "\(parsedArguments.get(dataset)!):\(parsedArguments.get(model)!)"
    configuration = "\(configuration):bep-\(graph.badEdgeProportion)"
    if let targetBadEdgeProportion = parsedArguments.get(targetBadEdgeProportion) {
      configuration = "\(configuration):tbep-\(targetBadEdgeProportion)"
    }
    configuration = "\(configuration):lh-\(parsedArguments.get(lHiddenUnitCounts)!.map(String.init).joined(separator: "-"))"
    if let qHiddenUnitCounts = parsedArguments.get(qHiddenUnitCounts) {
      configuration = "\(configuration):qh-\(qHiddenUnitCounts.map(String.init).joined(separator: "-"))"
    }
    configuration = "\(configuration):bs-\(parsedArguments.get(batchSize) ?? 128)"
    configuration = "\(configuration):ls-\(parsedArguments.get(labelSmoothing) ?? 0.5)"
    configuration = "\(configuration):dr-\(parsedArguments.get(dropout) ?? 0.5)"
    configuration = "\(configuration):s-\(parsedArguments.get(seed) ?? 123456789)"
    return configuration
  }
}

extension Float: ArgumentKind {
  public init(argument: String) throws {
    guard let float = Float(argument) else {
      throw ArgumentConversionError.typeMismatch(value: argument, expectedType: Float.self)
    }

    self = float
  }

  public static let completion: ShellCompletion = .none
}

enum ModelName: String, StringEnumArgument {
  case mlp = "mlp"
  case decoupledMLP = "decoupled-mlp"
  case gcn = "gcn"
  case decoupledGCN = "decoupled-gcn"

  public static let completion: ShellCompletion = .none
}
