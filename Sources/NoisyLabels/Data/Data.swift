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

public struct TrainingData: TensorGroup {
  public let instances: Tensor<Int32>
  public let predictors: Tensor<Int32>
  public let labels: Tensor<Int32>
  public let values: Tensor<Float>

  public init(
    instances: Tensor<Int32>,
    predictors: Tensor<Int32>,
    labels: Tensor<Int32>,
    values: Tensor<Float>
  ) {
    self.instances = instances
    self.predictors = predictors
    self.labels = labels
    self.values = values
  }

  public init<C: RandomAccessCollection>(_handles: C) where C.Element: _AnyTensorHandle {
    precondition(_handles.count == 4)
    let iIndex = _handles.startIndex
    let pIndex = _handles.index(iIndex, offsetBy: 1)
    let lIndex = _handles.index(pIndex, offsetBy: 1)
    let vIndex = _handles.index(lIndex, offsetBy: 1)
    instances = Tensor<Int32>(handle: TensorHandle<Int32>(handle: _handles[iIndex]))
    predictors = Tensor<Int32>(handle: TensorHandle<Int32>(handle: _handles[pIndex]))
    labels = Tensor<Int32>(handle: TensorHandle<Int32>(handle: _handles[lIndex]))
    values = Tensor<Float>(handle: TensorHandle<Float>(handle: _handles[vIndex]))
  }

  public var _tensorHandles: [_AnyTensorHandle] {
    instances._tensorHandles + predictors._tensorHandles + 
      labels._tensorHandles + values._tensorHandles
  }
}

public struct InferenceData: TensorGroup {
  public let instances: Tensor<Int32>
  public let predictors: Tensor<Int32>
  public let labels: Tensor<Int32>

  public init(
    instances: Tensor<Int32>,
    predictors: Tensor<Int32>,
    labels: Tensor<Int32>
  ) {
    self.instances = instances
    self.predictors = predictors
    self.labels = labels
  }

  public init<C: RandomAccessCollection>(_handles: C) where C.Element: _AnyTensorHandle {
    precondition(_handles.count == 3)
    let iIndex = _handles.startIndex
    let pIndex = _handles.index(iIndex, offsetBy: 1)
    let lIndex = _handles.index(pIndex, offsetBy: 1)
    instances = Tensor<Int32>(handle: TensorHandle<Int32>(handle: _handles[iIndex]))
    predictors = Tensor<Int32>(handle: TensorHandle<Int32>(handle: _handles[pIndex]))
    labels = Tensor<Int32>(handle: TensorHandle<Int32>(handle: _handles[lIndex]))
  }

  public var _tensorHandles: [_AnyTensorHandle] {
    instances._tensorHandles + predictors._tensorHandles + labels._tensorHandles
  }
}

public struct Data<Instance, Predictor, Label> {
  public let instances: [Instance]
  public let predictors: [Predictor]
  public let labels: [Label]
  
  /// Contains the true label value per label and per instance.
  public let trueLabels: [Int: [Int: Int]]

  /// Contains a tuple of two lists (instance indices and predicted values) 
  /// per label and per predictor.
  public let predictedLabels: [Int: [Int: (instances: [Int], values: [Float])]]

  /// Number of classes per label.
  public let classCounts: [Int]

  public let instanceFeatures: [Tensor<Float>]?
  public let predictorFeatures: [Tensor<Float>]?
  public let labelFeatures: [Tensor<Float>]?

  public init(
    instances: [Instance], 
    predictors: [Predictor], 
    labels: [Label],
    trueLabels: [Int: [Int: Int]], 
    predictedLabels: [Int: [Int: (instances: [Int], values: [Float])]], 
    classCounts: [Int], 
    instanceFeatures: [Tensor<Float>]? = nil, 
    predictorFeatures: [Tensor<Float>]? = nil, 
    labelFeatures: [Tensor<Float>]? = nil
  ) {
    self.instances = instances
    self.predictors = predictors
    self.labels = labels
    self.trueLabels = trueLabels
    self.predictedLabels = predictedLabels
    self.classCounts = classCounts
    self.instanceFeatures = instanceFeatures
    self.predictorFeatures = predictorFeatures
    self.labelFeatures = labelFeatures
  }

  public var instanceIndices: [Int] {
    return [Int](0..<instances.count)
  }

  public var predictorIndices: [Int] {
    return [Int](0..<predictors.count)
  }

  public var labelIndices: [Int] {
    return [Int](0..<labels.count)
  }

  public var avgLabelsPerPredictor: Float {
    var numLabels = 0
    for l in 0..<labels.count {
      for predictions in predictedLabels[l]!.values {
        numLabels += predictions.instances.count
      }
    }
    return Float(numLabels) / Float(predictors.count)
  }

  public var avgLabelsPerItem: Float {
    var numLabels = 0
    for l in 0..<labels.count {
      for predictions in predictedLabels[l]!.values {
        numLabels += predictions.instances.count
      }
    }
    return Float(numLabels) / Float(instances.count)
  }

  public static func join(_ datasets: Data...) -> Data {
    var iIndices = [Int: Int]()
    var pIndices = [Int: Int]()
    var lIndices = [Int: Int]()

    var instances = [Instance]()
    var predictors = [Predictor]()
    var labels = [Label]()

    var trueLabels = [Int: [Int: Int]]()
    var predictedLabels = [Int: [Int: (instances: [Int], values: [Float])]]()
    var classCounts = [Int]()

    var hasIF = true
    var hasPF = true
    var hasLF = true
    for dataset in datasets {
      hasIF = hasIF && dataset.instanceFeatures != nil
      hasPF = hasPF && dataset.predictorFeatures != nil
      hasLF = hasPF && dataset.labelFeatures != nil
    }
    var instanceFeatures = hasIF ? [Tensor<Float>]() : nil
    var predictorFeatures = hasPF ? [Tensor<Float>]() : nil
    var labelFeatures = hasLF ? [Tensor<Float>]() : nil

    for dataset in datasets {
      for (lOld, label) in dataset.labels.enumerated() {
        let l = lIndices[lOld] ?? {
          let value = labels.count
          labels.append(label)
          lIndices[lOld] = value
          if hasLF {
            labelFeatures!.append(dataset.labelFeatures![lOld])
          }
          return value
        }()
        
        // Process the true labels.
        if !trueLabels.keys.contains(l) {
          trueLabels[l] = [Int: Int]()
        }
        for (iOld, trueLabel) in dataset.trueLabels[lOld]! {
          let instance = dataset.instances[iOld]
          let i = iIndices[iOld] ?? {
            let value = instances.count
            instances.append(instance)
            iIndices[iOld] = value
            if hasIF {
              instanceFeatures!.append(dataset.instanceFeatures![iOld])
            }
            return value
          }()
          trueLabels[l]![i] = trueLabel
        }

        // Process the predicted labels.
        if !predictedLabels.keys.contains(l) {
          predictedLabels[l] = [Int: (instances: [Int], values: [Float])]()
        }
        for (pOld, predictions) in dataset.predictedLabels[lOld]! {
          let predictor = dataset.predictors[pOld]
          let p = pIndices[pOld] ?? {
            let value = predictors.count
            predictors.append(predictor)
            pIndices[pOld] = value
            if hasPF {
              predictorFeatures!.append(dataset.predictorFeatures![pOld])
            }
            return value
          }()

          if !predictedLabels[l]!.keys.contains(p) {
            predictedLabels[l]![p] = (instances: [Int](), values: [Float]())
          }
          for (iOld, value) in zip(predictions.instances, predictions.values) {
            let instance = dataset.instances[iOld]
            let i = iIndices[iOld] ?? {
              let value = instances.count
              instances.append(instance)
              iIndices[iOld] = value
              if hasIF {
                instanceFeatures!.append(dataset.instanceFeatures![iOld])
              }
              return value
            }()
            predictedLabels[l]![p]!.instances.append(i)
            predictedLabels[l]![p]!.values.append(value)
          }
        }

        // Process the number of classes.
        classCounts = zip(classCounts, dataset.classCounts).map(+)
      }
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

  // TODO: Support shuffling.
  public func trainingData() -> TrainingData {
    var instances = [Int]()
    var predictors = [Int]()
    var labels = [Int]()
    var values = [Float]()
    for (l, allPredictions) in self.predictedLabels {
      for (p, predictions) in allPredictions {
        instances.append(contentsOf: predictions.instances)
        predictors.append(contentsOf: [Int](repeating: p, count: predictions.instances.count))
        labels.append(contentsOf: [Int](repeating: l, count: predictions.instances.count))
        values.append(contentsOf: predictions.values)
      }
    }
    return TrainingData(
      instances: Tensor<Int32>(instances.map { Int32($0) }),
      predictors: Tensor<Int32>(predictors.map { Int32($0) }),
      labels: Tensor<Int32>(labels.map { Int32($0) }),
      values: Tensor<Float>(values))
  }
}

public extension Data {
  func computeBinaryQualities() -> Tensor<Float> {
    var numCorrect = [Float](repeating: 0.0, count: labels.count * predictors.count)
    var numTotal = [Float](repeating: 0.0, count: labels.count * predictors.count)
    for l in labels.indices {
      for (p, predictions) in predictedLabels[l]! {
        for (i, v) in zip(predictions.instances, predictions.values) {
          if trueLabels[l]![i]! == (v >= 0.5 ? 1 : 0) {
            numCorrect[l * predictors.count + p] += 1
          }
          numTotal[l * predictors.count + p] += 1
        }
      }
    }
    let l = labels.count
    let p = predictors.count
    let numCorrectTensor = Tensor<Float>(shape: [l, p], scalars: numCorrect)
    let numTotalTensor = Tensor<Float>(shape: [l, p], scalars: numTotal)
    return numCorrectTensor / numTotalTensor
  }

  func computeBinaryQualitiesFullConfusion() -> Tensor<Float> {
    var numConfused = [Float](repeating: 0.0, count: labels.count * predictors.count * 4)
    var numTotal = [Float](repeating: 0.0, count: labels.count * predictors.count)
    for l in labels.indices {
      for (p, predictions) in predictedLabels[l]! {
        for (i, v) in zip(predictions.instances, predictions.values) {
          if trueLabels[l]![i]! == 0 {
            if v >= 0.5 {
              numConfused[l * predictors.count + p * 4 + 1] += 1
            } else {
              numConfused[l * predictors.count + p * 4] += 1
            }
          } else {
            if v >= 0.5 {
              numConfused[l * predictors.count + p * 4 + 3] += 1
            } else {
              numConfused[l * predictors.count + p * 4 + 2] += 1
            }
          }
          if trueLabels[l]![i]! == (v >= 0.5 ? 1 : 0) {
            numConfused[l * predictors.count + p] += 1
          }
          numTotal[l * predictors.count + p] += 1
        }
      }
    }
    let l = labels.count
    let p = predictors.count
    let numConfusedTensor = Tensor<Float>(shape: [l, p, 2, 2], scalars: numConfused)
    let numTotalTensor = Tensor<Float>(shape: [l, p], scalars: numTotal)
    return numConfusedTensor / numTotalTensor
  }
}

public extension Data where Label: Equatable {
  func filtered(labels: [Label]) -> Data {
    var iIndices = [Int: Int]()
    var pIndices = [Int: Int]()

    var instances = [Instance]()
    var predictors = [Predictor]()

    var trueLabels = [Int: [Int: Int]]()
    var predictedLabels = [Int: [Int: (instances: [Int], values: [Float])]]()
    var classCounts = [Int]()

    let hasIF = instanceFeatures != nil
    let hasPF = predictorFeatures != nil
    let hasLF = labelFeatures != nil
    var instanceFeatures = hasIF ? [Tensor<Float>]() : nil
    var predictorFeatures = hasPF ? [Tensor<Float>]() : nil
    var labelFeatures = hasLF ? [Tensor<Float>]() : nil

    for (l, label) in labels.enumerated() {
      let lOld = self.labels.firstIndex(of: label)!
      classCounts.append(self.classCounts[lOld])
      if hasLF {
        labelFeatures!.append(self.labelFeatures![lOld])
      }

      // Filter the true labels.
      trueLabels[l] = [Int: Int]()
      for (iOld, trueLabel) in self.trueLabels[lOld]! {
        let i = iIndices[iOld] ?? {
          let value = instances.count
          instances.append(self.instances[iOld])
          iIndices[iOld] = value
          if hasIF {
            instanceFeatures!.append(self.instanceFeatures![iOld])
          }
          return value
        }()
        trueLabels[l]![i] = trueLabel
      }

      // Filter the predicted labels.
      predictedLabels[l] = [Int: (instances: [Int], values: [Float])]()
      for (pOld, predictions) in self.predictedLabels[lOld]! {
        let p = pIndices[pOld] ?? {
          let value = predictors.count
          predictors.append(self.predictors[pOld])
          pIndices[pOld] = value
          if hasPF {
            predictorFeatures!.append(self.predictorFeatures![pOld])
          }
          return value
        }()
        predictedLabels[l]![p] = (instances: [Int](), values: [Float]())
        for (iOld, value) in zip(predictions.instances, predictions.values) {
          let i = iIndices[iOld] ?? {
            let value = instances.count
            instances.append(self.instances[iOld])
            iIndices[iOld] = value
            if hasIF {
              instanceFeatures!.append(self.instanceFeatures![iOld])
            }
            return value
          }()
          predictedLabels[l]![p]!.instances.append(i)
          predictedLabels[l]![p]!.values.append(value)
        }
      }
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

public extension Data where Predictor: Equatable {
  func filtered(predictors: [Predictor], keepInstances: Bool = true) -> Data {
    var iIndices = [Int: Int]()
    var pIndices = [Int: Int]()

    var newInstances = [Instance]()
    var newPredictors = [Predictor]()

    var trueLabels = [Int: [Int: Int]]()
    var predictedLabels = [Int: [Int: (instances: [Int], values: [Float])]]()
    var classCounts = [Int]()

    let hasIF = instanceFeatures != nil
    let hasPF = predictorFeatures != nil
    let hasLF = labelFeatures != nil
    var instanceFeatures = hasIF ? [Tensor<Float>]() : nil
    var predictorFeatures = hasPF ? [Tensor<Float>]() : nil
    var labelFeatures = hasLF ? [Tensor<Float>]() : nil

    if keepInstances {
      newInstances = instances
      instanceFeatures = self.instanceFeatures
    }

    for (l, _) in labels.enumerated() {
      classCounts.append(self.classCounts[l])
      if hasLF {
        labelFeatures!.append(self.labelFeatures![l])
      }

      // Filter the true labels.
      trueLabels[l] = [Int: Int]()
      for (iOld, trueLabel) in self.trueLabels[l]! {
        let i: Int = {
          if keepInstances {
            return iOld
          } else {
            return iIndices[iOld] ?? -1
          }
        }()
        if i == -1 {
          continue
        } else {
          trueLabels[l]![i] = trueLabel
        }
      }

      // Filter the predicted labels.
      predictedLabels[l] = [Int: (instances: [Int], values: [Float])]()
      for (pOld, predictions) in self.predictedLabels[l]! {
        let predictor = self.predictors[pOld]
        if !predictors.contains(predictor) {
          continue
        }
        let p = pIndices[pOld] ?? {
          let value = newPredictors.count
          newPredictors.append(predictor)
          pIndices[pOld] = value
          if hasPF {
            predictorFeatures!.append(self.predictorFeatures![pOld])
          }
          return value
        }()
        predictedLabels[l]![p] = (instances: [Int](), values: [Float]())
        for (iOld, value) in zip(predictions.instances, predictions.values) {
          let i: Int = {
            if keepInstances {
              return iOld
            } else {
              return iIndices[iOld] ?? {
                let value = newInstances.count
                newInstances.append(self.instances[iOld])
                iIndices[iOld] = value
                if hasIF {
                  instanceFeatures!.append(self.instanceFeatures![iOld])
                }
                return value
              }()
            }
          }()
          predictedLabels[l]![p]!.instances.append(i)
          predictedLabels[l]![p]!.values.append(value)
        }
      }
    }

    return Data(
      instances: newInstances,
      predictors: newPredictors,
      labels: labels,
      trueLabels: trueLabels,
      predictedLabels: predictedLabels,
      classCounts: classCounts,
      instanceFeatures: instanceFeatures,
      predictorFeatures: predictorFeatures,
      labelFeatures: labelFeatures)
  }
}
