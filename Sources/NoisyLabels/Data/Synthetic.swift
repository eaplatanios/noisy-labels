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

extension Data {
  public func withSyntheticPredictors<G: RandomNumberGenerator>(using generator: inout G) -> Data {
    let predictors = self.predictors + [
      "Correct Oracle" as! Predictor,
      "Wrong Oracle" as! Predictor,
      "Random Oracle" as! Predictor]
    let c = self.predictors.count
    let w = c + 1
    let r = w + 1
    var predictedLabels = self.predictedLabels
    for l in predictedLabels.keys {
      let instances = [Int](trueLabels[l]!.keys)
      let values = [Int](trueLabels[l]!.values).map(Float.init)
      predictedLabels[l]![c] = (instances: instances, values: values)
      predictedLabels[l]![w] = (instances: instances, values: values.map { 1 - $0 })
      predictedLabels[l]![r] = (
        instances: instances,
        values: values.map { _ in Float(Int.random(in: 0...1, using: &generator)) })
    }
    return Data(
      instances: instances,
      predictors: predictors,
      labels: labels,
      trueLabels: trueLabels,
      predictedLabels: predictedLabels,
      classCounts: classCounts,
      instanceFeatures: instanceFeatures,
      predictorFeatures: predictorFeatures,
      labelFeatures: labelFeatures)
  }
}
