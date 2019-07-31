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

import TensorFlow

public protocol Learner {
  mutating func train<Instance, Predictor, Label>(using data: Data<Instance, Predictor, Label>)
  func labelProbabilities(_ instances: [Int]) -> [Tensor<Float>]
  func qualities(_ instances: [Int], _ predictors: [Int], _ labels: [Int]) -> Tensor<Float>
}

public struct MajorityVoteLearner: Learner {
  public let useSoftMajorityVote: Bool
  public let verbose: Bool

  /// Array over instances and labels, which contains the estimated label probabilities.
  private var estimatedLabelProbabilities: [[Float]] = [[Float]]()
  private var estimatedQualities: [[[Float]]] = [[[Float]]]()

  public init(useSoftMajorityVote: Bool, verbose: Bool = false) {
    self.useSoftMajorityVote = useSoftMajorityVote
    self.verbose = verbose
  }

  public mutating func train<Instance, Predictor, Label>(
    using data: Data<Instance, Predictor, Label>
  ) {
    // Count the "votes".
    if verbose { logger.info("Counting the \"votes\".") }
    let instanceCount: Int = data.instances.count
    let labelCount: Int = data.labels.count
    for l in 0..<data.labels.count {
      var values = [Float](repeating: 0.0, count: instanceCount)
      var counts = [Int](repeating: 0, count: instanceCount)
      for predictions in data.predictedLabels[l]!.values {
        for (i, v) in zip(predictions.instances, predictions.values) {
          values[i] += useSoftMajorityVote ? v : (v >= 0.5 ? 1.0 : 0.0)
          counts[i] += 1
        }
      }
      for i in 0..<instanceCount {
        if counts[i] > 0 {
          values[i] /= Float(counts[i])
        } else {
          values[i] = 0.5
        }
      }
      estimatedLabelProbabilities.append(values)
    }

    estimatedQualities = [[[Float]]](
      repeating: [[Float]](
        repeating: [Float](repeating: -1.0, count: data.predictors.count),
        count: labelCount),
      count: instanceCount)
    for l in 0..<labelCount {
      for (p, predictions) in data.predictedLabels[l]! {
        for (i, v) in zip(predictions.instances, predictions.values) {
          if (v >= 0.5) == (estimatedLabelProbabilities[l][i] >= 0.5) {
            estimatedQualities[i][l][p] = 1.0
          } else {
            estimatedQualities[i][l][p] = 0.0
          }
        }
      }
    }

    if verbose { logger.info("Finished counting the \"votes\".") }
  }

  public func labelProbabilities(_ instances: [Int]) -> [Tensor<Float>] {
    estimatedLabelProbabilities.map { p in
      Tensor<Float>(instances.map { p[$0] })
    }
  }

  public func qualities(
    _ instances: [Int],
    _ predictors: [Int],
    _ labels: [Int]
  ) -> Tensor<Float> {
    Tensor<Float>(instances.map { i in
      Tensor<Float>(labels.map { l in
        Tensor<Float>(predictors.map { p in estimatedQualities[i][l][p] })
      })
    })
  }
}

// TODO: !!!! Remove `Optimizer.Scalar == Float` once formal support for learning rate decay lands.
public struct EMLearner<
  Predictor: NoisyLabels.Predictor,
  Optimizer: TensorFlow.Optimizer
>: Learner where Optimizer.Model == Predictor, Optimizer.Scalar == Float {
  public private(set) var model: EMModel<Predictor, Optimizer>

  public let randomSeed: Int64
  public let batchSize: Int
  public let useWarmStarting: Bool
  public let mStepCount: Int
  public let emStepCount: Int
  public let marginalStepCount: Int
  public let mStepLogCount: Int?
  public let emStepCallback: (EMLearner) -> Void
  public let verbose: Bool

  public init(
    for model: EMModel<Predictor, Optimizer>,
    randomSeed: Int64,
    batchSize: Int = 128,
    useWarmStarting: Bool = true,
    mStepCount: Int = 1000,
    emStepCount: Int = 100,
    marginalStepCount: Int = 0,
    mStepLogCount: Int? = 100,
    emStepCallback: @escaping (EMLearner) -> Void = { _ in },
    verbose: Bool = false
  ) {
    self.model = model
    self.randomSeed = randomSeed
    self.batchSize = batchSize
    self.useWarmStarting = useWarmStarting
    self.mStepCount = mStepCount
    self.emStepCount = emStepCount
    self.marginalStepCount = marginalStepCount
    self.mStepLogCount = mStepLogCount
    self.emStepCallback = emStepCallback
    self.verbose = verbose
  }

  public mutating func train<Instance, Predictor, Label>(
    using data: Data<Instance, Predictor, Label>
  ) {
    let dataset = Dataset(elements: data.trainingData())

    for emStep in 0..<emStepCount {
      // E-Step
      if verbose { logger.info("Iteration \(emStep) - Running E-Step") }
      model.prepareForEStep()
      for batch in dataset.batched(batchSize) {
        model.executeEStep(using: batch, majorityVote: emStep == 0)
      }
      model.finalizeEStep(majorityVote: emStep == 0)

      // M-Step
      if verbose { logger.info("Iteration \(emStep) - Running M-Step") }
      if !useWarmStarting { model.prepareForMStep() }
      var accumulatedNLL = Float(0.0)
      var accumulatedSteps = 0
      var datasetIterator = dataset.repeated()
        // .shuffled(sampleCount: 10000, randomSeed: randomSeed)
        .batched(batchSize)
        .prefetched(count: 100)
        .makeIterator()

      for mStep in 0..<mStepCount {
        accumulatedNLL += model.executeMStep(
          using: datasetIterator.next()!,
          majorityVote: emStep == 0)
        accumulatedSteps += 1
        if verbose {
          if let logSteps = mStepLogCount, mStep % logSteps == 0 || mStep == mStepCount - 1 {
            let nll = accumulatedNLL / Float(accumulatedSteps)
            let message = "M-Step \(String(format: "%5d", mStep)) | " + 
              "Negative Log-Likelihood: \(String(format: "%.8f", nll))"
            logger.info("\(message)")
            accumulatedNLL = 0.0
            accumulatedSteps = 0
          }
        }
      }

      emStepCallback(self)
    }

    // Marginal Likelihood Optimization
    if marginalStepCount > 0 {
      if verbose { logger.info("Optimizing marginal likelihood.") }
      var accumulatedNLL = Float(0.0)
      var accumulatedSteps = 0
      var datasetIterator = dataset.repeated()
        // .shuffled(sampleCount: 10000, randomSeed: randomSeed)
        .batched(batchSize).makeIterator()
      for step in 0..<marginalStepCount {
        accumulatedNLL += model.executeMarginalStep(using: datasetIterator.next()!)
        accumulatedSteps += 1
        if verbose {
          if let logSteps = mStepLogCount, step % logSteps == 0 || step == marginalStepCount - 1 {
            let nll = accumulatedNLL / Float(accumulatedSteps)
            let message = "Marginal-Step \(String(format: "%5d", step)) | " + 
              "Negative Log-Likelihood: \(String(format: "%.8f", nll))"
            logger.info("\(message)")
            accumulatedNLL = 0.0
            accumulatedSteps = 0
          }
        }
      }
    }
  }

  public func negativeLogLikelihood(for data: TrainingData) -> Float {
    var nll = Float(0.0)
    for batch in Dataset(elements: data).batched(batchSize) {
      nll += model.negativeLogLikelihood(for: batch)
    }
    return nll
  }

  public func labelProbabilities(_ instances: [Int]) -> [Tensor<Float>] {
    let instances = Tensor<Int32>(instances.map(Int32.init))
    var labelProbabilities = [[Tensor<Float>]]()
    for batch in Dataset(elements: instances).batched(batchSize) {
      let predictions = model.labelProbabilities(batch)
      if labelProbabilities.isEmpty {
        for p in predictions {
          labelProbabilities.append([p])
        }
      } else {
        for (l, p) in predictions.enumerated() {
          labelProbabilities[l].append(p)
        }
      }
    }
    return labelProbabilities.map {
      $0.count > 1 ? Tensor<Float>(concatenating: $0, alongAxis: 0) : $0[0]
    }
  }

  public func qualities(
    _ instances: [Int],
    _ predictors: [Int],
    _ labels: [Int]
  ) -> Tensor<Float> {
    // TODO: Make the following batched and lazy.
    // Compute the Cartesian product of instances, predictors, and labels.
    let productCount = instances.count * predictors.count * labels.count
    var productInstances = [Int]()
    var productPredictors = [Int]()
    var productLabels = [Int]()
    productInstances.reserveCapacity(productCount)
    productPredictors.reserveCapacity(productCount)
    productLabels.reserveCapacity(productCount)
    for i in 0..<instances.count {
      for p in 0..<predictors.count {
        for l in 0..<labels.count {
          productInstances.append(instances[i])
          productPredictors.append(predictors[p])
          productLabels.append(labels[l])
        }
      }
    }

    let dataset = Dataset(elements: InferenceData(
      instances: Tensor<Int32>(productInstances.map(Int32.init)),
      predictors: Tensor<Int32>(productPredictors.map(Int32.init)),
      labels: Tensor<Int32>(productLabels.map(Int32.init))))

    // Collect the qualities for each batch.
    var qualities = [[Tensor<Float>]]()
    for batch in dataset.batched(batchSize) {
      let predictions = model.qualities(batch.instances, batch.predictors, batch.labels)
      if qualities.isEmpty {
        for p in predictions {
          qualities.append([p])
        }
      } else {
        for (i, p) in predictions.enumerated() {
          qualities[i].append(p)
        }
      }
    }

    // Aggregate the collected qualities into a single matrix with shape:
    // [InstanceCount, LabelCount, PredictorCount].
    let concatenatedQualities = qualities.map { qualities in
      qualities.count > 1 ? 
        exp(Tensor<Float>(concatenating: qualities, alongAxis: 0)) :
        exp(qualities[0])
    }
    let aggregatedQualities = concatenatedQualities.count > 1 ? 
      Tensor<Float>(concatenating: concatenatedQualities, alongAxis: 0) :
      concatenatedQualities[0]
    return aggregatedQualities
      .reshaped(to: [labels.count, instances.count, predictors.count])
      .transposed(withPermutations: 1, 0, 2)
  }
}

#if SNORKEL

import Foundation
import Python

// TODO: The Python interface is not thread-safe.
internal let pythonDispatchSemaphore = DispatchSemaphore(value: 1)

public struct SnorkelLearner: Learner {
  private var snorkelMarginals: [Tensor<Float>]!
  private var snorkelQualities: [QualitiesKey: Float]!

  public init() {}

  public mutating func train<Instance, Predictor, Label>(
    using data: Data<Instance, Predictor, Label>
  ) {
    pythonDispatchSemaphore.wait()
    defer { pythonDispatchSemaphore.signal() }

    // We first suppress all Python printing because Snorkel prints some messages and it is
    // impossible to disable that. We also disable the `tqdm` progress bar package that Snorkel is
    // using in a hacky way.
    let sys = Python.import("sys")
    let os = Python.import("os")
    sys.stdout = Python.open(os.devnull, "w")
    let locals = Python.import("__main__").__dict__
    Python.exec("""
      import tqdm
      def nop(*a, **k):
        if len(a) > 0:
          return a[0]
        return None
      tqdm.tqdm = nop
      """, locals, locals)

    let np = Python.import("numpy")
    let snorkel = Python.import("snorkel")
    let snorkelModels = Python.import("snorkel.models")
    let snorkelText = Python.import("snorkel.contrib.models.text")
    let snorkelGenLearning = Python.import("snorkel.learning.gen_learning")
    snorkelQualities = [QualitiesKey: Float]()
    snorkelMarginals = data.labels.indices.map { label in
      let session = snorkel.SnorkelSession()
      let snorkelLabel = snorkelModels.candidate_subclass(
        "NoisyLabelsLabel\(data.labels)",
        ["index"],
        cardinality: data.classCounts[label])

      // Make sure the database is empty.
      session.query(snorkelModels.Candidate).delete()
      session.query(snorkelModels.Context).delete()

      // Add all the candidates.
      for instance in data.instances.indices {
        let instanceIndex = snorkelText.RawText(
          stable_id: String(instance),
          name: String(instance),
          text: String(instance))
        session.add(snorkelLabel(index: instanceIndex))
      }
      session.commit()

      // Dictionary mapping from instance to `(predictor, value)` pairs.
      var predictedLabels = [String: [PythonObject]]()
      for instance in data.instances.indices {
        predictedLabels[String(instance)] = [PythonObject]()
      }
      for (predictor, predictions) in data.predictedLabels[label]! {
        for (instance, value) in zip(predictions.instances, predictions.values) {
          let value = data.classCounts[label] == 2 ? 2 * Int(value) - 1 : Int(value) + 1
          predictedLabels[String(instance)]!.append(Python.tuple([predictor, value]))
        }
      }

      // Create a Snorkel label annotator.
      Python.exec("""
        from snorkel.annotations import LabelAnnotator
        def labeler(predicted_labels):
          def worker_label_generator(t):
            for worker_id, label in predicted_labels[t.index.name]:
              yield worker_id, label
          return LabelAnnotator(label_generator=worker_label_generator)
        """, locals, locals)
      let labelAnnotator = locals["labeler"](predictedLabels)
      let labels = labelAnnotator.apply()

      // Train a generative model on the labels.
      let model = snorkelGenLearning.GenerativeModel(lf_propensity: true)
      model.train(labels, reg_type: 2, reg_param: 0.1, epochs: 30)
      let marginals = Tensor<Float>(numpy: model.marginals(labels).astype(np.float32))!

      // Close the Snorkel session.
      session.close()

      // Estimate the qualities.
      for (predictor, predictions) in data.predictedLabels[label]! {
        for (instance, value) in zip(predictions.instances, predictions.values) {
          // We use a heuristic way to estimate predictor qualities because Snorkel does not
          // provide an explicit way to do so.
          let quality = data.classCounts[label] == 2 ?
            abs(value - marginals[instance].scalarized()) :
            marginals[instance, Int(value)].scalarized()
          snorkelQualities[QualitiesKey(instance, label, predictor)] = quality
        }
      }

      return marginals
    }
  }

  public func labelProbabilities(_ instances: [Int]) -> [Tensor<Float>] {
    snorkelMarginals.map { marginals in
      Tensor<Float>(
        stacking: instances.map { instance in
          let marginals = marginals[instance]
          if marginals.rank == 0 {
            return Tensor<Float>(stacking: [1.0 - marginals, marginals])
          } else {
            return marginals
          }
        })
    }
  }

  public func qualities(
    _ instances: [Int],
    _ predictors: [Int],
    _ labels: [Int]
  ) -> Tensor<Float> {
    var qualities = [Float]()
    qualities.reserveCapacity(instances.count * labels.count * predictors.count)
    var i = 0
    for instance in instances {
      for label in labels {
        for predictor in predictors {
          qualities.append(snorkelQualities[QualitiesKey(instance, label, predictor)] ?? -1)
          i += 1
        }
      }
    }
    return Tensor<Float>(
      shape: [instances.count, labels.count, predictors.count],
      scalars: qualities)
  }
}

extension SnorkelLearner {
  internal struct QualitiesKey: Hashable {
    internal let instance: Int
    internal let label: Int
    internal let predictor: Int

    internal init(_ instance: Int, _ label: Int, _ predictor: Int) {
      self.instance = instance
      self.label = label
      self.predictor = predictor
    }
  }
}

public struct MetalLearner: Learner {
  private var metalMarginals: [Tensor<Float>]!
  private var metalQualities: [QualitiesKey: Float]!
  private var metalRandomSeed: Int?

  public init(randomSeed: Int? = nil) {
    self.metalRandomSeed = randomSeed
  }

  public mutating func train<Instance, Predictor, Label>(
    using data: Data<Instance, Predictor, Label>
  ) {
    pythonDispatchSemaphore.wait()
    defer { pythonDispatchSemaphore.signal() }

    // We first suppress all Python printing because Snorkel prints some messages and it is
    // impossible to disable that. We also disable the `tqdm` progress bar package that Snorkel is
    // using in a hacky way.
    let sys = Python.import("sys")
    let os = Python.import("os")
    sys.stdout = Python.open(os.devnull, "w")

    let np = Python.import("numpy")
    let spSparse = Python.import("scipy.sparse")
    let metalLabelModel = Python.import("metal.label_model")
    metalQualities = [QualitiesKey: Float]()
    metalMarginals = data.labels.indices.map { label in
      let annotations = spSparse.lil_matrix(
        Python.tuple([data.instances.count, data.predictors.count]),
        dtype: np.int32)
      for (predictor, predictions) in data.predictedLabels[label]! {
        for (instance, value) in zip(predictions.instances, predictions.values) {
          annotations[instance, predictor] = Python.int(Int(value) + 1)
        }
      }
      let labelModel = metalLabelModel.LabelModel(
        k: data.classCounts[label],
        seed: metalRandomSeed ?? Int.random(in: 0..<1000000),
        verbose: false)
      labelModel.train_model(annotations)
      let marginals = Tensor<Float>(
        numpy: labelModel.predict_proba(annotations).astype(np.float32))!

      // Estimate the qualities.
      for (predictor, predictions) in data.predictedLabels[label]! {
        for (instance, value) in zip(predictions.instances, predictions.values) {
          // We use a heuristic way to estimate predictor qualities because Snorkel does not
          // provide an explicit way to do so.
          let quality = marginals[instance, Int(value)].scalarized()
          metalQualities[QualitiesKey(instance, label, predictor)] = quality
        }
      }

      return marginals
    }
  }

  public func labelProbabilities(_ instances: [Int]) -> [Tensor<Float>] {
    metalMarginals.map { marginals in
      Tensor<Float>(
        stacking: instances.map { instance in
          let marginals = marginals[instance]
          if marginals.rank == 0 {
            return Tensor<Float>(stacking: [1.0 - marginals, marginals])
          } else {
            return marginals
          }
        })
    }
  }

  public func qualities(
    _ instances: [Int],
    _ predictors: [Int],
    _ labels: [Int]
  ) -> Tensor<Float> {
    var qualities = [Float]()
    qualities.reserveCapacity(instances.count * labels.count * predictors.count)
    var i = 0
    for instance in instances {
      for label in labels {
        for predictor in predictors {
          qualities.append(metalQualities[QualitiesKey(instance, label, predictor)] ?? -1)
          i += 1
        }
      }
    }
    return Tensor<Float>(
      shape: [instances.count, labels.count, predictors.count],
      scalars: qualities)
  }
}

extension MetalLearner {
  internal struct QualitiesKey: Hashable {
    internal let instance: Int
    internal let label: Int
    internal let predictor: Int

    internal init(_ instance: Int, _ label: Int, _ predictor: Int) {
      self.instance = instance
      self.label = label
      self.predictor = predictor
    }
  }
}

#endif
