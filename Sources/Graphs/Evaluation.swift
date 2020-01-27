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

public struct Result {
  public let trainAccuracy: Float
  public let validationAccuracy: Float
  public let testAccuracy: Float

  public init(trainAccuracy: Float, validationAccuracy: Float, testAccuracy: Float) {
    self.trainAccuracy = trainAccuracy
    self.validationAccuracy = validationAccuracy
    self.testAccuracy = testAccuracy
  }
}

extension Result {
  public func scaled(by scale: Float) -> Result {
    Result(
      trainAccuracy: trainAccuracy * scale,
      validationAccuracy: validationAccuracy * scale,
      testAccuracy: testAccuracy * scale)
  }

  public func adding(_ result: Result) -> Result {
    Result(
      trainAccuracy: trainAccuracy + result.trainAccuracy,
      validationAccuracy: validationAccuracy + result.validationAccuracy,
      testAccuracy: testAccuracy + result.testAccuracy)
  }
}

public func evaluate<P: GraphPredictor, O: Optimizer>(
  model: Model<P, O>,
  using graph: Graph,
  usePrior: Bool = false
) -> Result where O.Model == P {
  func evaluate(_ indices: [Int32], _ labels: [Int]) -> Float {
    let probabilities = model.labelProbabilities(for: indices, usePrior: usePrior)
    let predictions = probabilities.argmax(squeezingAxis: -1).scalars.map(Int.init)
    return zip(predictions, labels).map {
      $0 == $1 ? 1.0 : 0.0
    }.reduce(0, +) / Float(predictions.count)
  }
  return Result(
    trainAccuracy: evaluate(
      graph.trainNodes,
      graph.trainNodes.map { graph.labels[$0]! }),
    validationAccuracy: evaluate(
      graph.validationNodes,
      graph.validationNodes.map { graph.labels[$0]! }),
    testAccuracy: evaluate(
      graph.testNodes,
      graph.testNodes.map { graph.labels[$0]! }))
}
