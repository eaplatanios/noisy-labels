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

public enum DataPartition: String {
  case all, train, test
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
  public let partitions: [DataPartition: [Int]]?

  public init(
    instances: [Instance], 
    predictors: [Predictor], 
    labels: [Label],
    trueLabels: [Int: [Int: Int]], 
    predictedLabels: [Int: [Int: (instances: [Int], values: [Float])]], 
    classCounts: [Int], 
    instanceFeatures: [Tensor<Float>]? = nil, 
    predictorFeatures: [Tensor<Float>]? = nil, 
    labelFeatures: [Tensor<Float>]? = nil,
    partitions: [DataPartition: [Int]]? = nil
  ) {
    precondition(
      partitions == nil || partitions!.keys.contains(.train),
      "If data partitions are provided, a 'train' partition must be included.")
    self.instances = instances
    self.predictors = predictors
    self.labels = labels
    self.trueLabels = trueLabels
    self.predictedLabels = predictedLabels
    self.classCounts = classCounts
    self.instanceFeatures = instanceFeatures
    self.predictorFeatures = predictorFeatures
    self.labelFeatures = labelFeatures
    self.partitions = partitions
  }

  public var instanceIndices: [Int] {
    [Int](0..<instances.count)
  }

  public var predictorIndices: [Int] {
    [Int](0..<predictors.count)
  }

  public var labelIndices: [Int] {
    [Int](0..<labels.count)
  }

  public var avgLabelsPerPredictor: Float {
    var labelCounts = [Float](repeating: 0, count: predictors.count)
    for l in 0..<labels.count {
      for (p, predictions) in predictedLabels[l]! {
        labelCounts[p] += Float(predictions.instances.count)
      }
    }
    return labelCounts.mean
  }

  public var avgLabelsPerItem: Float {
    var labelCounts = [Float](repeating: 0, count: instances.count)
    for l in 0..<labels.count {
      for predictions in predictedLabels[l]!.values {
        for i in predictions.instances {
          labelCounts[i] += 1
        }
      }
    }
    return labelCounts.mean
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
    var hasP = true
    for dataset in datasets {
      hasIF = hasIF && dataset.instanceFeatures != nil
      hasPF = hasPF && dataset.predictorFeatures != nil
      hasLF = hasPF && dataset.labelFeatures != nil
      hasP = hasP && dataset.partitions != nil
    }
    var instanceFeatures = hasIF ? [Tensor<Float>]() : nil
    var predictorFeatures = hasPF ? [Tensor<Float>]() : nil
    var labelFeatures = hasLF ? [Tensor<Float>]() : nil
    var partitions = hasP ? [DataPartition: [Int]]() : nil

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
        // Note: The following sorting operation is necessary to guarantee reproducibility.
        for (iOld, trueLabel) in dataset.trueLabels[lOld]!.sorted(by: { $0.key < $1.key }) {
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
        // Note: The following sorting operation is necessary to guarantee reproducibility.
        for (pOld, predictions) in dataset.predictedLabels[lOld]!.sorted(by: { $0.key < $1.key }) {
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

      if hasP {
        for (partition, indices) in dataset.partitions! {
          if !partitions!.keys.contains(partition) {
            partitions![partition] = [Int]()
          }
          for iOld in indices {
            partitions![partition]!.append(iIndices[iOld]!)
          }
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
      labelFeatures: labelFeatures,
      partitions: partitions)
  }
  
  public func trainingData() -> TrainingData {
    var instances = [Int]()
    var predictors = [Int]()
    var labels = [Int]()
    var values = [Float]()
    // The following sorting operations are necessary to guarantee reproducibility.
    for (l, allPredictions) in predictedLabels.sorted(by: { $0.key < $1.key }) {
      for (p, predictions) in allPredictions.sorted(by: { $0.key < $1.key }) {
        let filtered = { () -> (instances: [Int], values: [Float]) in
          if let partitions = self.partitions {
            let instanceSet = Set<Int>(partitions[.train]!)
            return zip(predictions.instances, predictions.values)
              .filter { instanceSet.contains($0.0) }
              .reduce(into: (instances: [Int](), values: [Float]())) {
                $0.instances.append($1.0)
                $0.values.append($1.1)
              }
          } else {
            return (instances: predictions.instances, values: predictions.values)
          }
        }()
        instances.append(contentsOf: filtered.instances)
        predictors.append(contentsOf: [Int](repeating: p, count: filtered.instances.count))
        labels.append(contentsOf: [Int](repeating: l, count: filtered.instances.count))
        values.append(contentsOf: filtered.values)
      }
    }
    return TrainingData(
      instances: Tensor<Int32>(instances.map { Int32($0) }),
      predictors: Tensor<Int32>(predictors.map { Int32($0) }),
      labels: Tensor<Int32>(labels.map { Int32($0) }),
      values: Tensor<Float>(values))
  }
}

extension Data {
  public func computeBinaryQualities() -> [DataPartition: Tensor<Float>] {
    let partitions = self.partitions ?? [.all: [Int](0..<instances.count)]
    return [DataPartition: Tensor<Float>](uniqueKeysWithValues:
      partitions.map { (partition, instances) -> (DataPartition, Tensor<Float>) in
        let instanceSet = Set<Int>(instances)
        var numCorrect = [Float](repeating: 0.0, count: labels.count * predictors.count)
        var numTotal = [Float](repeating: 0.0, count: labels.count * predictors.count)
        for l in labels.indices {
          for (p, predictions) in predictedLabels[l]! {
            for (i, v) in zip(predictions.instances, predictions.values) {
              if instanceSet.contains(i) {
                if let trueLabel = trueLabels[l]?[i] {
                  if trueLabel == (v >= 0.5 ? 1 : 0) {
                    numCorrect[l * predictors.count + p] += 1
                  }
                  numTotal[l * predictors.count + p] += 1
                }
              }
            }
          }
        }
        let l = labels.count
        let p = predictors.count
        let numCorrectTensor = Tensor<Float>(shape: [l, p], scalars: numCorrect)
        let numTotalTensor = Tensor<Float>(shape: [l, p], scalars: numTotal)
        return (partition, numCorrectTensor / numTotalTensor)
      })
  }

  public func computeBinaryQualitiesFullConfusion() -> [DataPartition: Tensor<Float>] {
    let partitions = self.partitions ?? [.all: [Int](0..<instances.count)]
    return [DataPartition: Tensor<Float>](uniqueKeysWithValues:
      partitions.map { (partition, instances) -> (DataPartition, Tensor<Float>) in
        let instanceSet = Set<Int>(instances)
        var numConfused = [Float](repeating: 0.0, count: labels.count * predictors.count * 4)
        var numTotal = [Float](repeating: 0.0, count: labels.count * predictors.count)
        for l in labels.indices {
          for (p, predictions) in predictedLabels[l]! {
            for (i, v) in zip(predictions.instances, predictions.values) {
              if instanceSet.contains(i) {
                if let trueLabel = trueLabels[l]?[i] {
                  if trueLabel == 0 {
                    if v >= 0.5 {
                      numConfused[l * predictors.count * 4 + p * 4 + 1] += 1
                    } else {
                      numConfused[l * predictors.count * 4 + p * 4] += 1
                    }
                  } else {
                    if v >= 0.5 {
                      numConfused[l * predictors.count * 4 + p * 4 + 3] += 1
                    } else {
                      numConfused[l * predictors.count * 4 + p * 4 + 2] += 1
                    }
                  }
                  numTotal[l * predictors.count + p] += 1
                }
              }
            }
          }
        }
        let l = labels.count
        let p = predictors.count
        let numConfusedTensor = Tensor<Float>(shape: [l, p, 2, 2], scalars: numConfused)
        let numTotalTensor = Tensor<Float>(shape: [l, p, 1, 1], scalars: numTotal)
        return (partition, numConfusedTensor / numTotalTensor)
      })
  }
}

extension Data where Label: Equatable {
  public func filtered(labels: [Label]) -> Data {
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
    let hasP = partitions != nil
    var instanceFeatures = hasIF ? [Tensor<Float>]() : nil
    var predictorFeatures = hasPF ? [Tensor<Float>]() : nil
    var labelFeatures = hasLF ? [Tensor<Float>]() : nil
    var partitions = hasP ? [DataPartition: [Int]]() : nil

    for (l, label) in labels.enumerated() {
      let lOld = self.labels.firstIndex(of: label)!
      classCounts.append(self.classCounts[lOld])
      if hasLF {
        labelFeatures!.append(self.labelFeatures![lOld])
      }

      // Filter the true labels.
      trueLabels[l] = [Int: Int]()
      // Note: The following sorting operation is necessary to guarantee reproducibility.
      for (iOld, trueLabel) in self.trueLabels[lOld]!.sorted(by: { $0.key < $1.key }) {
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
      // Note: The following sorting operation is necessary to guarantee reproducibility.
      for (pOld, predictions) in self.predictedLabels[lOld]!.sorted(by: { $0.key < $1.key }) {
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

    if hasP {
      for (partition, indices) in self.partitions! {
        if !partitions!.keys.contains(partition) {
          partitions![partition] = [Int]()
        }
        for iOld in indices {
          if let i = iIndices[iOld] {
            partitions![partition]!.append(i)
          }
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
      labelFeatures: labelFeatures,
      partitions: partitions)
  }
}

extension Data where Predictor: Equatable {
  public func filtered(predictors: [Predictor], keepInstances: Bool = true) -> Data {
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
    let hasP = partitions != nil
    var instanceFeatures = hasIF ? [Tensor<Float>]() : nil
    var predictorFeatures = hasPF ? [Tensor<Float>]() : nil
    var labelFeatures = hasLF ? [Tensor<Float>]() : nil
    var partitions = hasP ? [DataPartition: [Int]]() : nil

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
      // Note: The following sorting operation is necessary to guarantee reproducibility.
      for (iOld, trueLabel) in self.trueLabels[l]!.sorted(by: { $0.key < $1.key }) {
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
      // Note: The following sorting operation is necessary to guarantee reproducibility.
      for (pOld, predictions) in self.predictedLabels[l]!.sorted(by: { $0.key < $1.key }) {
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

    if hasP {
      for (partition, indices) in self.partitions! {
        if !partitions!.keys.contains(partition) {
          partitions![partition] = [Int]()
        }
        for iOld in indices {
          if let i = iIndices[iOld] {
            partitions![partition]!.append(i)
          }
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
      labelFeatures: labelFeatures,
      partitions: partitions)
  }
}

extension Data {
  public func withMaxRedundancy<G: RandomNumberGenerator>(
    _ maxRedundancy: Int,
    using generator: inout G
  ) -> Data {
    var pIndices = [Int: Int]()
    var newPredictors = [Predictor]()
    var predictedLabels = [Int: [Int: (instances: [Int], values: [Float])]]()

    let hasPF = predictorFeatures != nil
    var predictorFeatures = hasPF ? [Tensor<Float>]() : nil

    // Create dictionary mapping from labels to dictionaries from instances to their annotations.
    var instanceAnnotations = [Int: [Int: [(predictor: Int, value: Float)]]]()
    for l in labels.indices {
      instanceAnnotations[l] = [Int: [(predictor: Int, value: Float)]]()
      for i in instances.indices {
        instanceAnnotations[l]![i] = [(predictor: Int, value: Float)]()
      }
      // Note: The following sorting operation is necessary to guarantee reproducibility.
      for (pOld, predictions) in self.predictedLabels[l]!.sorted(by: { $0.key < $1.key }) {
        for (i, value) in zip(predictions.instances, predictions.values) {
          instanceAnnotations[l]![i]!.append((predictor: pOld, value: value))
        }
      }

      // Discard some annotations to enforce the requested redundancy limit.
      // Note: The following sorting operation is necessary to guarantee reproducibility.
      for (i, annotations) in instanceAnnotations[l]!.sorted(by: { $0.key < $1.key }) {
        if annotations.count > maxRedundancy {
          instanceAnnotations[l]![i] = sample(
            from: annotations,
            count: maxRedundancy,
            using: &generator)
        }
      }

      // Filter the predicted labels.
      predictedLabels[l] = [Int: (instances: [Int], values: [Float])]()
      // Note: The following sorting operation is necessary to guarantee reproducibility.
      for (i, annotations) in instanceAnnotations[l]!.sorted(by: { $0.key < $1.key }) {
        for (pOld, value) in annotations {
          let predictor = self.predictors[pOld]
          let p = pIndices[pOld] ?? {
            let value = newPredictors.count
            newPredictors.append(predictor)
            pIndices[pOld] = value
            if hasPF {
              predictorFeatures!.append(self.predictorFeatures![pOld])
            }
            return value
          }()
          if !predictedLabels[l]!.keys.contains(p) {
            predictedLabels[l]![p] = (instances: [Int](), values: [Float]())
          }
          predictedLabels[l]![p]!.instances.append(i)
          predictedLabels[l]![p]!.values.append(value)
        }
      }
    }

    return Data(
      instances: instances,
      predictors: newPredictors,
      labels: labels,
      trueLabels: trueLabels,
      predictedLabels: predictedLabels,
      classCounts: classCounts,
      instanceFeatures: instanceFeatures,
      predictorFeatures: predictorFeatures,
      labelFeatures: labelFeatures,
      partitions: partitions)
  }
}

extension Data {
  public func partitioned<G: RandomNumberGenerator>(
    trainPortion: Float,
    using generator: inout G
  ) -> Data {
    precondition(trainPortion > 0 && trainPortion <= 1)
    let instances = [Int](0..<self.instances.count).shuffled(using: &generator)
    let trainCount = Int(Float(instances.count) * trainPortion)
    let testCount = instances.count - trainCount
    let partitions = { () -> [DataPartition: [Int]]? in
      if testCount == 0 {
        return nil
      } else {
        var partitions = [DataPartition: [Int]]()
        partitions[.all] = instances
        partitions[.train] = [Int](instances.prefix(trainCount))
        partitions[.test] = [Int](instances.suffix(testCount))
        return partitions
      }
    }()
    return Data(
      instances: self.instances,
      predictors: self.predictors,
      labels: self.labels,
      trueLabels: self.trueLabels,
      predictedLabels: self.predictedLabels,
      classCounts: self.classCounts,
      instanceFeatures: self.instanceFeatures,
      predictorFeatures: self.predictorFeatures,
      labelFeatures: self.labelFeatures,
      partitions: partitions)
  }
}
