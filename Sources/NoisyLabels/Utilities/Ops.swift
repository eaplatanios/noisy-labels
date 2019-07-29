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

public struct Pair<Element1: Differentiable, Element2: Differentiable>: Differentiable {
  @differentiable public var first: Element1
  @differentiable public var second: Element2

  @inlinable
  @differentiable
  public init(first: Element1, second: Element2) {
    self.first = first
    self.second = second
  }
}

@inlinable
@differentiable(vjp: _vjpDifferentiableZip)
internal func differentiableZip<Element1: Differentiable, Element2: Differentiable>(
  _ array1: [Element1],
  _ array2: [Element2]
) -> [Pair<Element1, Element2>] {
  zip(array1, array2).map { Pair(first: $0, second: $1) }
}

@inlinable
internal func _vjpDifferentiableZip<Element1: Differentiable, Element2: Differentiable>(
  _ array1: [Element1],
  _ array2: [Element2]
) -> ([Pair<Element1, Element2>], (Array<Pair<Element1, Element2>>.TangentVector) -> (
  Array<Element1>.TangentVector,
  Array<Element2>.TangentVector
)) {
  (differentiableZip(array1, array2), { v in
    var array1 = [Element1.TangentVector](repeating: Element1.TangentVector.zero, count: array1.count)
    var array2 = [Element2.TangentVector](repeating: Element2.TangentVector.zero, count: array2.count)
    for i in v.base.indices {
      if i < array1.count { array1[i] = v[i].first }
      if i < array2.count { array2[i] = v[i].second }
    }
    return (Array<Element1>.TangentVector(array1), Array<Element2>.TangentVector(array2))
  })
}

public extension Array where Element: Differentiable {
  @differentiable(wrt: (self, context), vjp: _vjpDifferentiableContextualMap)
  func differentiableMap<Context: Differentiable, Result: Differentiable>(
    _ context: Context,
    _ body: @differentiable (Context, Element) -> Result
  ) -> [Result] {
    map { body(context, $0) }
  }

  @usableFromInline
  internal func _vjpDifferentiableContextualMap<Context: Differentiable, Result: Differentiable>(
    _ context: Context,
    _ body: @differentiable (Context, Element) -> Result
  ) -> ([Result], (Array<Result>.TangentVector) -> (Array.TangentVector, Context.TangentVector)) {
    var values: [Result] = []
    var pullbacks: [(Result.TangentVector) -> (Element.TangentVector, Context.TangentVector)] = []
    for x in self {
      let (y, pb) = Swift.valueWithPullback(at: x, context) { body($1, $0) }
      values.append(y)
      pullbacks.append(pb)
    }
    return (values, { v in
      var array = [Element.TangentVector]()
      var context = Context.TangentVector.zero
      for (element, pullback) in zip(v.base, pullbacks) {
        let (e, c) = pullback(element)
        array.append(e)
        context += c
      }
      return (Array.TangentVector(array), context)
    })
  }
}

public struct ModelParameters: Differentiable {
  @noDerivative public let labelMask: Tensor<Bool>
  @noDerivative public var eStepAccumulator: Tensor<Float>
  @differentiable public var labelProbabilities: Tensor<Float>
  @differentiable public var qualities: Tensor<Float>

  @inlinable
  @differentiable(wrt: (labelProbabilities, qualities))
  public init(
    labelMask: Tensor<Bool>,
    eStepAccumulator: Tensor<Float>,
    labelProbabilities: Tensor<Float>,
    qualities: Tensor<Float>
  ) {
    self.labelMask = labelMask
    self.eStepAccumulator = eStepAccumulator
    self.labelProbabilities = labelProbabilities
    self.qualities = qualities
  }
}

@inlinable
@differentiable(wrt: (labelProbabilities, qualities), vjp: _vjpModelZip)
internal func modelZip(
  labelMasks: [Tensor<Bool>],
  eStepAccumulators: [Tensor<Float>],
  labelProbabilities: [Tensor<Float>],
  qualities: [Tensor<Float>]
) -> [ModelParameters] {
  var result = [ModelParameters]()
  result.reserveCapacity(labelMasks.count)
  for l in labelMasks.indices {
    result.append(ModelParameters(
      labelMask: labelMasks[l],
      eStepAccumulator: eStepAccumulators[l],
      labelProbabilities: labelProbabilities[l],
      qualities: qualities[l]))
  }
  return result
}

@inlinable
internal func _vjpModelZip(
  labelMasks: [Tensor<Bool>],
  eStepAccumulators: [Tensor<Float>],
  labelProbabilities: [Tensor<Float>],
  qualities: [Tensor<Float>]
) -> ([ModelParameters], (Array<ModelParameters>.TangentVector) -> (
  Array<Tensor<Float>>.TangentVector,
  Array<Tensor<Float>>.TangentVector
)) {
  (
    modelZip(
      labelMasks: labelMasks,
      eStepAccumulators: eStepAccumulators,
      labelProbabilities: labelProbabilities,
      qualities: qualities),
    { v in
      var p = [Tensor<Float>](repeating: Tensor<Float>.zero, count: labelProbabilities.count)
      var q = [Tensor<Float>](repeating: Tensor<Float>.zero, count: qualities.count)
      for i in v.base.indices {
        if i < labelProbabilities.count { p[i] = v[i].labelProbabilities }
        if i < qualities.count { q[i] = v[i].qualities }
      }
      return (Array<Tensor<Float>>.TangentVector(p), Array<Tensor<Float>>.TangentVector(q))
    })
}

// public class AMSGrad<Model: Differentiable & KeyPathIterable>: Optimizer
//     where Model.TangentVector: VectorProtocol & PointwiseMultiplicative & 
//                                ElementaryFunctions & KeyPathIterable,
//           Model.TangentVector.VectorSpaceScalar == Float,
//           Model.AllDifferentiableVariables == Model.TangentVector {
//     public typealias Model = Model
//     /// The learning rate.
//     public var learningRate: Float
//     /// A coefficient used to calculate the first and second moments of the gradients.
//     public var beta1: Float
//     /// A coefficient used to calculate the first and second moments of the gradients.
//     public var beta2: Float
//     /// A small scalar added to the denominator to improve numerical stability.
//     public var epsilon: Float
//     /// The learning rate decay.
//     public var decay: Float
//     /// The current step.
//     public var step: Int = 0
//     /// The first moments of the weights.
//     public var firstMoments: Model.TangentVector = .zero
//     /// The second moments of the weights.
//     public var secondMoments: Model.TangentVector = .zero
//     /// The maximum of the second moments of the weights.
//     public var secondMomentsMax: Model.TangentVector = .zero

//     public init(
//         for model: __shared Model,
//         learningRate: Float = 1e-3,
//         beta1: Float = 0.9,
//         beta2: Float = 0.999,
//         epsilon: Float = 1e-8,
//         decay: Float = 0
//     ) {
//         precondition(learningRate >= 0, "Learning rate must be non-negative")
//         precondition(0 <= beta1 && beta1 <= 1, "Beta parameter must be between 0 and 1")
//         precondition(0 <= beta2 && beta2 <= 1, "Beta parameter must be between 0 and 1")
//         precondition(decay >= 0, "Learning rate decay must be non-negative")

//         self.learningRate = learningRate
//         self.beta1 = beta1
//         self.beta2 = beta2
//         self.epsilon = epsilon
//         self.decay = decay
//     }

//     public func update(_ model: inout Model, along direction: Model.TangentVector) {
//         update(&model.allDifferentiableVariables, along: direction)
//     }

//     // TODO: Deprecate this when `Differentiable.AllDifferentiableVariables` is removed.
//     public func update(
//         _ model: inout Model.AllDifferentiableVariables,
//         along direction: Model.TangentVector
//     ) {
//         self.step += 1
//         let step = Float(self.step)
//         let beta1Power = pow(beta1, step)
//         let beta2Power = pow(beta2, step)
//         let learningRate = self.learningRate * 1 / (1 + decay * step)
//         // Note: `stepSize` and `secondMoments` are split into two lines to avoid the "compiler is 
//         // unable to type-check this expression in reasonable time" error.
//         var stepSize = learningRate * sqrt(1 - pow(beta2Power, step))
//         stepSize = stepSize / (1 - pow(beta1Power, step))
//         firstMoments = firstMoments * beta1 + direction * (1 - beta1)
//         secondMoments = secondMoments * beta2
//         secondMoments += direction .* direction * (1 - beta2)

//         // Update `secondMomentsMax` using a key path approach because `max(_:_:)` cannot be 
//         // currently applied in a simpler manner.
//         for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
//             secondMomentsMax[keyPath: kp] = max(
//                 secondMomentsMax[keyPath: kp], secondMoments[keyPath: kp])
//         }
//         for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
//             secondMomentsMax[keyPath: kp] = max(
//                 secondMomentsMax[keyPath: kp], secondMoments[keyPath: kp])
//         }

//         let denominator = Model.TangentVector.sqrt(secondMomentsMax) + epsilon
//         model.move(along: -stepSize * firstMoments ./ denominator)
//     }
// }
