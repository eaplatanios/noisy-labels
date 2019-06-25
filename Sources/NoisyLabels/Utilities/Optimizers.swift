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

fileprivate extension Tensor where Scalar: Numeric {
  mutating func resetToZero() {
    self = Tensor(zeros: shape)
  }
}

/// Adam optimizer.
///
/// Reference: ["Adam - A Method for Stochastic Optimization"](
/// https://arxiv.org/abs/1412.6980v8)
public class Adam<Model: Differentiable & KeyPathIterable>: Optimizer
where Model.AllDifferentiableVariables: KeyPathIterable,
      Model.AllDifferentiableVariables == Model.CotangentVector {
  /// The learning rate.
  public var learningRate: Float
  /// A coefficient used to calculate the first and second moments of
  /// gradients.
  public var beta1: Float
  /// A coefficient used to calculate the first and second moments of
  /// gradients.
  public var beta2: Float
  /// A small scalar added to the denominator to improve numerical stability.
  public var epsilon: Float
  /// The weight decay.
  public var decay: Float
  /// The current step.
  public var step: Int = 0
  /// The first moments of the weights.
  public var firstMoments: Model.AllDifferentiableVariables
  /// The second moments of the weights.
  public var secondMoments: Model.AllDifferentiableVariables

  public init(
    for model: __shared Model,
    learningRate: Float = 1e-3,
    beta1: Float = 0.9,
    beta2: Float = 0.999,
    epsilon: Float = 1e-8,
    decay: Float = 0
  ) {
    precondition(learningRate >= 0, "Learning rate must be non-negative")
    precondition(0 <= beta1 && beta1 <= 1, "Beta parameter must be between 0 and 1")
    precondition(0 <= beta2 && beta2 <= 1, "Beta parameter must be between 0 and 1")
    precondition(decay >= 0, "Weight decay must be non-negative")

    self.learningRate = learningRate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.decay = decay

    // Initialize first & second moments to be zeros of the same shape.
    // We can't use `Model.AllDifferentiableVariables.zero` due to the
    // interaction between Key Paths and Differentiable Arrays.
    firstMoments = model.allDifferentiableVariables
    secondMoments = model.allDifferentiableVariables
    for kp in firstMoments.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
      firstMoments[keyPath: kp].resetToZero()
      secondMoments[keyPath: kp].resetToZero()
    }
    for kp in firstMoments.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
      firstMoments[keyPath: kp].resetToZero()
      secondMoments[keyPath: kp].resetToZero()
    }
  }

  public func update(
    _ model: inout Model.AllDifferentiableVariables,
    along direction: Model.AllDifferentiableVariables
  ) {
    step += 1
    let learningRate = self.learningRate * 1 / (1 + decay * Float(step))
    let stepSize = learningRate * (
      sqrt(1 - pow(beta2, Float(step))) / (1 - pow(beta1, Float(step))))
    // Update Float & Double Tensor variables.
    for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
      firstMoments[keyPath: kp] = 
        firstMoments[keyPath: kp] * beta1 + (1 - beta1) * direction[keyPath: kp]
      secondMoments[keyPath: kp] = 
        secondMoments[keyPath: kp] * beta2 + (1 - beta2) * 
        direction[keyPath: kp] * direction[keyPath: kp]
      model[keyPath: kp] -= stepSize * firstMoments[keyPath: kp] / 
        (sqrt(secondMoments[keyPath: kp]) + epsilon)
    }
    for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
      firstMoments[keyPath: kp] = firstMoments[keyPath: kp] * Double(beta1) + 
        Double((1 - beta1)) * direction[keyPath: kp]
      secondMoments[keyPath: kp] = secondMoments[keyPath: kp] * Double(beta2) + 
        Double(1 - beta2) * direction[keyPath: kp] * direction[keyPath: kp]
      model[keyPath: kp] -= Double(stepSize) * firstMoments[keyPath: kp] /
        sqrt(secondMoments[keyPath: kp]) + Double(epsilon)
    }
  }
}

/// Adam optimizer.
///
/// Reference: ["Adam - A Method for Stochastic Optimization"](
/// https://arxiv.org/abs/1412.6980v8)
public class AMSGrad<Model: Differentiable & KeyPathIterable>: Optimizer
where Model.AllDifferentiableVariables: KeyPathIterable,
      Model.AllDifferentiableVariables == Model.CotangentVector {
  /// The learning rate.
  public var learningRate: Float
  /// A coefficient used to calculate the first and second moments of
  /// gradients.
  public var beta1: Float
  /// A coefficient used to calculate the first and second moments of
  /// gradients.
  public var beta2: Float
  /// A small scalar added to the denominator to improve numerical stability.
  public var epsilon: Float
  /// The weight decay.
  public var decay: Float

  public var beta1Power: Float
  public var beta2Power: Float

  /// The current step.
  public var step: Int = 0

  /// The first moments of the weights.
  public var firstMoments: Model.AllDifferentiableVariables
  /// The second moments of the weights.
  public var secondMoments: Model.AllDifferentiableVariables

  public var vHat: Model.AllDifferentiableVariables

  public init(
    for model: __shared Model,
    learningRate: Float = 1e-3,
    beta1: Float = 0.9,
    beta2: Float = 0.99,
    epsilon: Float = 1e-8,
    decay: Float = 0
  ) {
    precondition(learningRate >= 0, "Learning rate must be non-negative")
    precondition(0 <= beta1 && beta1 <= 1, "Beta parameter must be between 0 and 1")
    precondition(0 <= beta2 && beta2 <= 1, "Beta parameter must be between 0 and 1")
    precondition(decay >= 0, "Weight decay must be non-negative")

    self.learningRate = learningRate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.decay = decay

    self.beta1Power = beta1
    self.beta2Power = beta2

    // Initialize first & second moments to be zeros of the same shape.
    // We can't use `Model.AllDifferentiableVariables.zero` due to the
    // interaction between Key Paths and Differentiable Arrays.
    firstMoments = model.allDifferentiableVariables
    secondMoments = model.allDifferentiableVariables
    vHat = model.allDifferentiableVariables
    for kp in firstMoments.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
      firstMoments[keyPath: kp].resetToZero()
      secondMoments[keyPath: kp].resetToZero()
      vHat[keyPath: kp].resetToZero()
    }
    for kp in firstMoments.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
      firstMoments[keyPath: kp].resetToZero()
      secondMoments[keyPath: kp].resetToZero()
      vHat[keyPath: kp].resetToZero()
    }
  }

  public func update(
    _ model: inout Model.AllDifferentiableVariables,
    along direction: Model.AllDifferentiableVariables
  ) {
    step += 1
    let learningRate = self.learningRate * 1 / (1 + decay * Float(step))
    let stepSize = learningRate * (sqrt(1 - beta2Power) / (1 - beta1Power))
    // Update Float & Double Tensor variables.
    for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
      firstMoments[keyPath: kp] = 
        firstMoments[keyPath: kp] * beta1 + (1 - beta1) * direction[keyPath: kp]
      secondMoments[keyPath: kp] = 
        secondMoments[keyPath: kp] * beta2 + (1 - beta2) * 
        direction[keyPath: kp] * direction[keyPath: kp]
      vHat[keyPath: kp] = max(vHat[keyPath: kp], secondMoments[keyPath: kp])
      model[keyPath: kp] -= stepSize * firstMoments[keyPath: kp] / 
        (sqrt(vHat[keyPath: kp]) + epsilon)
    }
    for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
      firstMoments[keyPath: kp] = firstMoments[keyPath: kp] * Double(beta1) + 
        Double((1 - beta1)) * direction[keyPath: kp]
      secondMoments[keyPath: kp] = secondMoments[keyPath: kp] * Double(beta2) + 
        Double(1 - beta2) * direction[keyPath: kp] * direction[keyPath: kp]
      vHat[keyPath: kp] = max(vHat[keyPath: kp], secondMoments[keyPath: kp])
      model[keyPath: kp] -= Double(stepSize) * firstMoments[keyPath: kp] /
        sqrt(vHat[keyPath: kp]) + Double(epsilon)
    }
    beta1Power *= beta1
    beta2Power *= beta2
  }
}
