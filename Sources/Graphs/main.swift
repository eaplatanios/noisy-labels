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

// Example command: swift run -c release Graphs -r 5 -ecsc 5 --dataset citeseer --model decoupled-mlp --l-hidden 128 --q-hidden 128 -ls 0 -tbep 0.2 -sp 0.05 0.1

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
let splitProportions: OptionArgument<[Float]> = parser.add(
  option: "--split-proportions",
  shortName: "-sp",
  kind: [Float].self,
  usage: "Data set split proportions.")
let targetBadEdgeProportion: OptionArgument<Float> = parser.add(
  option: "--target-bad-edge-proportion",
  shortName: "-tbep",
  kind: Float.self,
  usage: "Target bad edge proportion.")
let evaluationConvergenceStepCount: OptionArgument<Int> = parser.add(
  option: "--evaluation-convergence-step-count",
  shortName: "-ecsc",
  kind: Int.self,
  usage: "Maximum number of steps without improvement on the validation set performance.")
let runCount: OptionArgument<Int> = parser.add(
  option: "--run-count",
  shortName: "-r",
  kind: Int.self,
  usage: "Number of runs.")
let seed: OptionArgument<Int> = parser.add(
  option: "--seed",
  shortName: "-s",
  kind: Int.self,
  usage: "Random seed.")

let parsedArguments = try! parser.parse(arguments)

let workingDirectory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
  .appendingPathComponent("temp")
let dataDirectory = workingDirectory
  .appendingPathComponent("data")
  .appendingPathComponent(parsedArguments.get(dataset)!)
let originalGraph = try Graph(loadFromDirectory: dataDirectory)
//let originalGraph = makeSimpleGraph(10, 20)

@discardableResult
func runExperiment<Predictor: GraphPredictor, G: RandomNumberGenerator>(
  predictor: (Graph) -> Predictor,
  randomSeed: Int64,
  using generator: inout G
) -> (
  firstPosteriorResult: Result,
  firstPriorResult: Result,
  posteriorResult: Result,
  priorResult: Result
) where Predictor.TangentVector: VectorProtocol & PointwiseMultiplicative & ElementaryFunctions,
        Predictor.TangentVector.VectorSpaceScalar == Float {
  withRandomSeedForTensorFlow(randomSeed) {
    var graph = originalGraph
    if let splitProportions = parsedArguments.get(splitProportions) {
      assert(splitProportions.count == 2, """
        The split proportions must be two numbers between 0 and 1 that correspond 
        to the train data and validation data proportions, respectively.
        """)
      graph = graph.split(
        trainProportion: splitProportions[0],
        validationProportion: splitProportions[1],
        using: &generator)
    }

    if let targetBadEdgeProportion = parsedArguments.get(targetBadEdgeProportion) {
      graph = graph.corrupted(
        targetBadEdgeProportion: targetBadEdgeProportion,
        using: &generator)
    }

    var stepCount = 0
    var firstEvaluationResult: Result? = nil
    var firstPriorEvaluationResult: Result? = nil
    var bestEvaluationResult: Result? = nil
    var bestPriorEvaluationResult: Result? = nil
    var stepCallbackInvocationsWithoutImprovement = 0
    var stepCallbackInvocationsWithoutPriorImprovement = 0

    func stepCallback<P: GraphPredictor, O: Optimizer>(model: Model<P, O>) -> () {
      stepCount += 1
      if !(stepCount - 1).isMultiple(of: 10) { return }
//      let predictionsMAP = model.labelsApproximateMAP(maxStepCount: 10000)
//      let predictionsMAP = model.labelsGibbsMarginalMAP()
//      let evaluationResult = evaluate(predictions: predictionsMAP, using: graph)
      let evaluationResult = evaluate(model: model, usePrior: false)
      if firstEvaluationResult == nil { firstEvaluationResult = evaluationResult }
      if let bestResult = bestEvaluationResult {
        if evaluationResult.validationAccuracy > bestResult.validationAccuracy ||
             (evaluationResult.validationAccuracy == bestResult.validationAccuracy &&
               evaluationResult.testAccuracy > bestResult.testAccuracy) {
          stepCallbackInvocationsWithoutImprovement = 0
          bestEvaluationResult = evaluationResult
        } else {
          stepCallbackInvocationsWithoutImprovement += 1
        }
      } else {
        bestEvaluationResult = evaluationResult
      }
      let priorEvaluationResult = evaluate(model: model, usePrior: true)
      if firstPriorEvaluationResult == nil { firstPriorEvaluationResult = priorEvaluationResult }
      if let bestResult = bestPriorEvaluationResult {
        if priorEvaluationResult.validationAccuracy > bestResult.validationAccuracy ||
             (priorEvaluationResult.validationAccuracy == bestResult.validationAccuracy &&
               priorEvaluationResult.testAccuracy > bestResult.testAccuracy) {
          stepCallbackInvocationsWithoutPriorImprovement = 0
          bestPriorEvaluationResult = priorEvaluationResult
        } else {
          stepCallbackInvocationsWithoutPriorImprovement += 1
        }
      } else {
        bestPriorEvaluationResult = priorEvaluationResult
      }
      logger.info("Configuration: \(configuration(graph: graph))")
      logger.info("Current Evaluation Result: \(evaluationResult)")
      logger.info("Current Prior Evaluation Result: \(priorEvaluationResult)")
      logger.info("Best Evaluation Result: \(String(describing: bestEvaluationResult))")
      logger.info("Best Prior Evaluation Result: \(String(describing: bestPriorEvaluationResult))")
      if stepCallbackInvocationsWithoutImprovement > 0 {
        logger.info("Evaluation result has not improved in \(stepCallbackInvocationsWithoutImprovement) step callback invocations.")
      }
      if stepCallbackInvocationsWithoutPriorImprovement > 0 {
        logger.info("Prior evaluation result has not improved in \(stepCallbackInvocationsWithoutPriorImprovement) step callback invocations.")
      }
    }

    let predictor = predictor(graph)
    let optimizer = Adam<Predictor>(
      for: predictor,
      learningRate: 1e-3,
      beta1: 0.9,
      beta2: 0.999,
      epsilon: 1e-8,
      decay: 0)

    var model = Model(
      predictor: predictor,
      optimizer: optimizer,
      randomSeed: randomSeed,
      batchSize: parsedArguments.get(batchSize) ?? 128,
      useIncrementalNeighborhoodExpansion: false,
      initializationMethod: .labelPropagation,
      stepCount: 10000,
      evaluationStepCount: nil,
      evaluationConvergenceStepCount: parsedArguments.get(evaluationConvergenceStepCount),
      evaluationResultsAccumulator: ExactAccumulator(),
      // evaluationResultsAccumulator: MovingAverageAccumulator(weight: 0.1),
      stepCallback: { stepCallback(model: $0) },
      logStepCount: 10,
      verbose: true)
    model.train()
    return (
      firstPosteriorResult: firstEvaluationResult!,
      firstPriorResult: firstPriorEvaluationResult!,
      posteriorResult: bestEvaluationResult!,
      priorResult: bestPriorEvaluationResult!)
  }
}

@discardableResult
func runExperiments<Predictor: GraphPredictor>(predictor: (Graph) -> Predictor) -> (
  firstPosteriorResultMean: Result,
  firstPosteriorResultStandardDeviation: Result,
  firstPriorResultMean: Result,
  firstPriorResultStandardDeviation: Result,
  posteriorResultMean: Result,
  posteriorResultStandardDeviation: Result,
  priorResultMean: Result,
  priorResultStandardDeviation: Result
) where Predictor.TangentVector: VectorProtocol & PointwiseMultiplicative & ElementaryFunctions,
        Predictor.TangentVector.VectorSpaceScalar == Float {
  var firstPosteriorResults = [Result]()
  var firstPriorResults = [Result]()
  var posteriorResults = [Result]()
  var priorResults = [Result]()
  for run in 0..<(parsedArguments.get(runCount) ?? 1) {
    logger.info("Starting run \(run)")
    let randomSeed = Int64(parsedArguments.get(seed) ?? 123456789) &+ Int64(run)
    var generator = PhiloxRandomNumberGenerator(seed: randomSeed)
    let (firstPosteriorResult, firstPriorResult, posteriorResult, priorResult) = runExperiment(
      predictor: predictor,
      randomSeed: randomSeed,
      using: &generator)
    firstPosteriorResults.append(firstPosteriorResult)
    firstPriorResults.append(firstPriorResult)
    posteriorResults.append(posteriorResult)
    priorResults.append(priorResult)
    logger.info("First posterior results moments: \(firstPosteriorResults.moments)")
    logger.info("First prior results moments: \(firstPriorResults.moments)")
    logger.info("Posterior results moments: \(posteriorResults.moments)")
    logger.info("Prior results moments: \(priorResults.moments)")
  }
  let firstPosteriorMoments = firstPosteriorResults.moments
  let firstPriorMoments = firstPriorResults.moments
  let posteriorMoments = posteriorResults.moments
  let priorMoments = priorResults.moments
  return (
    firstPosteriorResultMean: firstPosteriorMoments.mean,
    firstPosteriorResultStandardDeviation: firstPosteriorMoments.standardDeviation,
    firstPriorResultMean: firstPriorMoments.mean,
    firstPriorResultStandardDeviation: firstPriorMoments.standardDeviation,
    posteriorResultMean: posteriorMoments.mean,
    posteriorResultStandardDeviation: posteriorMoments.standardDeviation,
    priorResultMean: priorMoments.mean,
    priorResultStandardDeviation: priorMoments.standardDeviation)
}

switch parsedArguments.get(model)! {
case .mlp: runExperiments(predictor: { MLPPredictor(
  graph: $0,
  hiddenUnitCounts: parsedArguments.get(lHiddenUnitCounts)!,
  dropout: parsedArguments.get(dropout) ?? 0.5) })
default: fatalError("The specified model is not supported yet.")
//case .decoupledMLP: runExperiments(predictor: { DecoupledMLPPredictorV2(
//  graph: $0,
//  lHiddenUnitCounts: parsedArguments.get(lHiddenUnitCounts)!,
//  qHiddenUnitCounts: parsedArguments.get(qHiddenUnitCounts)!,
//  dropout: parsedArguments.get(dropout) ?? 0.5) })
//case .gcn: runExperiments(predictor: { GCNPredictor(
//  graph: $0,
//  hiddenUnitCounts: parsedArguments.get(lHiddenUnitCounts)!,
//  dropout: parsedArguments.get(dropout) ?? 0.5) })
//case .decoupledGCN: runExperiments(predictor: { DecoupledGCNPredictorV2(
//  graph: $0,
//  lHiddenUnitCounts: parsedArguments.get(lHiddenUnitCounts)!,
//  qHiddenUnitCounts: parsedArguments.get(qHiddenUnitCounts)!,
//  dropout: parsedArguments.get(dropout) ?? 0.5) })
}

func configuration(graph: Graph) -> String {
  var configuration = "\(parsedArguments.get(dataset)!):\(parsedArguments.get(model)!)"
  if let splitProportions = parsedArguments.get(splitProportions) {
    configuration = "\(configuration):sp-\(splitProportions[0])-\(splitProportions[1])"
  }
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
