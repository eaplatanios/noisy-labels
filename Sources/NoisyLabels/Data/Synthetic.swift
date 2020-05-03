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

public protocol SyntheticDataGenerator {
  func generate<Instance, Label, G: RandomNumberGenerator>(
    basedOn data: Data<Instance, String, Label>,
    using generator: inout G
  ) -> Data<Instance, String, Label>
}

public struct OraclesSyntheticDataGenerator: SyntheticDataGenerator {
  public init() {}

  public func generate<Instance, Label, G: RandomNumberGenerator>(
    basedOn data: Data<Instance, String, Label>,
    using generator: inout G
  ) -> Data<Instance, String, Label> {
    let predictors = data.predictors + ["Correct Oracle", "Wrong Oracle", "Random Oracle"]
    let c = data.predictors.count
    let w = c + 1
    let r = w + 1
    var predictedLabels = data.predictedLabels
    for l in predictedLabels.keys {
      let instances = [Int](data.trueLabels[l]!.keys)
      let values = [Int](data.trueLabels[l]!.values).map(Float.init)
      predictedLabels[l]![c] = (instances: instances, values: values)
      predictedLabels[l]![w] = (instances: instances, values: values.map { 1 - $0 })
      predictedLabels[l]![r] = (
        instances: instances,
        values: values.map { _ in Float(Int.random(in: 0...1, using: &generator)) })
    }
    return Data(
      instances: data.instances,
      predictors: predictors,
      labels: data.labels,
      trueLabels: data.trueLabels,
      predictedLabels: predictedLabels,
      classCounts: data.classCounts,
      instanceFeatures: data.instanceFeatures,
      predictorFeatures: data.predictorFeatures, // TODO: Is this right?
      labelFeatures: data.labelFeatures,
      partitions: data.partitions)
  }
}

public struct SyntheticPredictorsDataGenerator: SyntheticDataGenerator {
  public let predictorCount: Int
  public let usePredictorFeatures: Bool

  public init(predictorCount: Int, usePredictorFeatures: Bool) {
    self.predictorCount = predictorCount
    self.usePredictorFeatures = usePredictorFeatures
  }

  public func generate<Instance, Label, G: RandomNumberGenerator>(
    basedOn data: Data<Instance, String, Label>,
    using generator: inout G
  ) -> Data<Instance, String, Label> {
    precondition(
      data.labels.count == 1,
      "This synthetic data generator only works with single-label datasets.")

    let predictorEmbeddingSize = Int(ceil(Float(predictorCount) / 2)) + 1

    // Create the predictors, their features, and their confusion matrices.
    let C = data.classCounts[0]
    var predictors = [String]()
    var predictorFeatures = [Tensor<Float>]()
    var predictorConfusionMatrices = [Tensor<Float>]()
    for m in 0..<predictorCount {
      // var features = [Float](repeating: 0, count: predictorEmbeddingSize)
      // if m < predictorEmbeddingSize {
      //   features[m] = 1
      // } else {
      //   let n = m - predictorEmbeddingSize
      //   features[n] = 1
      //   features[n + 1] = 1
      // }
      // predictors.append(features.map { String($0) }.joined())
      // predictorFeatures.append(Tensor<Float>(features))

      // Create the confusion matrices.
      var features = [Float](repeating: 0, count: 2 * C)
      var confusionMatrix: Tensor<Float> = eye(rowCount: C, columnCount: C)
      if m < predictorEmbeddingSize {
        let labelPair = inverseCantor(2 * m)
        let l = labelPair.0 % C
        let k = labelPair.1 % C
        features[l] = -1
        features[C + k] = 1
        confusionMatrix[l, l] = Tensor<Float>(0)
        confusionMatrix[l, k] = Tensor<Float>(1)
      } else {
        let n = m - predictorEmbeddingSize
        let labelPair = inverseCantor(2 * n)
        let l = labelPair.0 % C
        let k = labelPair.1 % C
        features[l] = -0.5
        features[C + k] = 0.5
        features[k] = -0.5
        features[C + l] = 0.5
        // let labelPair1 = inverseCantor(2 * n)
        // let labelPair2 = inverseCantor(2 * n + 1)
        confusionMatrix[l, l] = Tensor<Float>(0.5)
        confusionMatrix[l, k] = Tensor<Float>(0.5)
        confusionMatrix[k, k] = Tensor<Float>(0.5)
        confusionMatrix[k, l] = Tensor<Float>(0.5)
        // confusionMatrix[labelPair1.0 % C, labelPair1.0 % C] = Tensor<Float>(0.75)
        // confusionMatrix[labelPair1.0 % C, labelPair1.1 % C] = Tensor<Float>(0.25)
        // confusionMatrix[labelPair2.0 % C, labelPair2.0 % C] = Tensor<Float>(0.75)
        // confusionMatrix[labelPair2.0 % C, labelPair2.1 % C] = Tensor<Float>(0.25)
      }
      predictors.append(features.map { String($0) }.joined())
      predictorFeatures.append(Tensor<Float>(features))
      predictorConfusionMatrices.append(confusionMatrix)
    }

    // Create the noisy predictions.
    var predictedLabels = [Int: [Int: (instances: [Int], values: [Float])]]()
    for l in 0..<data.labels.count {
      var predictions = [Int: (instances: [Int], values: [Float])]()
      for m in 0..<predictorCount {
        let confusionMatrix = predictorConfusionMatrices[m]
        var instances = [Int]()
        var values = [Float]()
        for (i, label) in data.trueLabels[l]! {
          instances.append(i)
          values.append(Float(Tensor<Int32>(
            randomCategorialLogits: confusionMatrix[label].expandingShape(at: 0),
            sampleCount: 1).scalarized()))
        }
        predictions[m] = (instances: instances, values: values)
      }
      predictedLabels[l] = predictions
    }

    return Data(
      instances: data.instances,
      predictors: predictors,
      labels: data.labels,
      trueLabels: data.trueLabels,
      predictedLabels: predictedLabels,
      classCounts: data.classCounts,
      instanceFeatures: data.instanceFeatures,
      predictorFeatures: usePredictorFeatures ? predictorFeatures : nil,
      labelFeatures: data.labelFeatures,
      partitions: data.partitions)
  }
}

@inlinable
internal func inverseCantor(_ z: Int) -> (Int, Int) {
  let w = Int(floor(((8 * Float(z) + 1).squareRoot() - 1) / 2))
  let t = (w * w + w) / 2
  let y = z - t
  let x = w - y
  return (x, y)
}
