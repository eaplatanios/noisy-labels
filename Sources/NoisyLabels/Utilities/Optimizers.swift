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

public class SignSGD<Model: Differentiable & KeyPathIterable>: Optimizer where
  Model.TangentVector: VectorProtocol & KeyPathIterable,
  Model.TangentVector.VectorSpaceScalar == Float {
  /// The learning rate.
  public var learningRate: Float

  /// The weight decay.
  public var decay: Float

  /// The set of steps taken.
  public var step: Int = 0

  public init(
    for model: __shared Model,
    learningRate: Float = 0.01,
    decay: Float = 0
  ) {
    precondition(learningRate >= 0, "Learning rate must be non-negative")
    precondition(decay >= 0, "Weight decay must be non-negative")
    self.learningRate = learningRate
    self.decay = decay
  }

  public func update(_ model: inout Model, along direction: Model.TangentVector) {
    step += 1
    let learningRate = self.learningRate * 1 / (1 + decay * Float(step))
    for (directionKp, modelKp) in zip(
      direction.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self),
      model.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self)
    ) {
      let update = sign(direction[keyPath: directionKp]) // * abs(model[keyPath: modelKp])
      model[keyPath: modelKp].move(along: update.scaled(by: -learningRate))
    }
  }
}

public class RProp<Model: Differentiable & KeyPathIterable>: Optimizer where
  Model.TangentVector: VectorProtocol & KeyPathIterable,
  Model.TangentVector.VectorSpaceScalar == Float {
  /// The learning rate.
  public var learningRate: Float

  public let alpha: Float
  public let beta: Float

  /// The set of steps taken.
  public var step: Int = 0

  /// The learning rate.
  public var currentLearningRate: Model.TangentVector = .zero

  /// The previous gradient of the model.
  private var previousDirection: Model.TangentVector = .zero

  public init(
    for model: __shared Model,
    initialLearningRate: Float = 0.01,
    alpha: Float = 1.2,
    beta: Float = 0.5
  ) {
    precondition(initialLearningRate >= 0, "Initial learning rate must be non-negative")
    self.learningRate = initialLearningRate
    self.currentLearningRate = currentLearningRate.adding(initialLearningRate)
    self.alpha = alpha
    self.beta = beta
  }

  public func update(_ model: inout Model, along direction: Model.TangentVector) {
    for ((directionKp, modelKp), lrKp) in zip(
      zip(
        direction.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self),
        model.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self)),
      currentLearningRate.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self)
    ) {
      if step == 0 {
        currentLearningRate[keyPath: lrKp] = Tensor<Float>(
          zerosLike: direction[keyPath: directionKp]) + learningRate
      } else {
        let change = direction[keyPath: directionKp] .* previousDirection[keyPath: directionKp]
        currentLearningRate[keyPath: lrKp] = currentLearningRate[keyPath: lrKp].replacing(
          with: currentLearningRate[keyPath: lrKp] * alpha,
          where: change .> 0
        ).replacing(
          with: currentLearningRate[keyPath: lrKp] * beta,
          where: change .< 0)
      }
      let update = -sign(direction[keyPath: directionKp])
      let lr = currentLearningRate[keyPath: lrKp]
      model[keyPath: modelKp].move(along: update .* lr)
    }
    previousDirection = direction
    step += 1
  }
}
