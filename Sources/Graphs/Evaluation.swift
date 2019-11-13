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

public func evaluate<P: Predictor, O: Optimizer>(
  model: Model<P, O>,
  using data: Data
) -> (trainAccuracy: Float, validationAccuracy: Float, testAccuracy: Float) where O.Model == P {
  func evaluate(_ indices: [Int], _ labels: [Int]) -> Float {
    let probabilities = model.labelProbabilities(for: indices)
    let predictions = probabilities.argmax(squeezingAxis: -1).scalars.map(Int.init)
    return zip(predictions, labels).map {
      $0 == $1 ? 1.0 : 0.0
    }.reduce(0, +) / Float(predictions.count)
  }
  return (
    trainAccuracy: evaluate(
      data.trainNodes,
      data.trainNodes.map { data.nodeLabels[$0]! }),
    validationAccuracy: evaluate(
      data.validationNodes,
      data.validationNodes.map { data.nodeLabels[$0]! }),
    testAccuracy: evaluate(
      data.testNodes,
      data.testNodes.map { data.nodeLabels[$0]! }))
}
