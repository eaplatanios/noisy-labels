import TensorFlow

public class RAdam<Model: Differentiable>: Optimizer where
  Model.TangentVector: VectorProtocol & PointwiseMultiplicative & ElementaryFunctions,
  Model.TangentVector.VectorSpaceScalar == Float {
    public typealias Model = Model
    /// The learning rate.
    public var learningRate: Float
    /// A coefficient used to calculate the first and second moments of the gradients.
    public var beta1: Float
    /// A coefficient used to calculate the first and second moments of the gradients.
    public var beta2: Float
    /// TODO: Document.
    public var degeneratedToSGD: Bool
    /// A small scalar added to the denominator to improve numerical stability.
    public var epsilon: Float
    /// The learning rate decay.
    public var decay: Float
    /// The current step.
    public var step: Int = 0
    /// The first moments of the weights.
    public var firstMoments: Model.TangentVector = .zero
    /// The second moments of the weights.
    public var secondMoments: Model.TangentVector = .zero

    public init(
        for model: __shared Model,
        learningRate: Float = 1e-3,
        beta1: Float = 0.9,
        beta2: Float = 0.999,
        degeneratedToSGD: Bool = true,
        epsilon: Float = 1e-8,
        decay: Float = 0
    ) {
        precondition(learningRate >= 0, "Learning rate must be non-negative")
        precondition(0 <= beta1 && beta1 <= 1, "Beta parameter must be between 0 and 1")
        precondition(0 <= beta2 && beta2 <= 1, "Beta parameter must be between 0 and 1")
        precondition(decay >= 0, "Learning rate decay must be non-negative")

        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.degeneratedToSGD = degeneratedToSGD
        self.epsilon = epsilon
        self.decay = decay
    }

    public func update(_ model: inout Model, along direction: Model.TangentVector) {
        step += 1
        let step = Float(self.step)
        let beta1Power = pow(beta1, step)
        let beta2Power = pow(beta2, step)
        secondMoments = secondMoments.scaled(by: beta2)
        secondMoments += (direction .* direction).scaled(by: 1 - beta2)
        firstMoments = firstMoments.scaled(by: beta1) + direction.scaled(by: 1 - beta1)
        // Compute maximum length SMA, bias-corrected moving average and approximate length.
        let N_sma_inf =  2 / (1 - beta2) - 1
        let N_sma_t = N_sma_inf - 2 * step * beta2Power / (1 - beta2Power)

        var stepSize = Float(0)
        if N_sma_t >= 5 {
          stepSize = sqrt(
            (N_sma_t - 4) * (N_sma_t - 2) * N_sma_inf / (
                  (N_sma_inf - 4) * (N_sma_inf - 2) * (N_sma_t)
            )).scaled(by: 1 / (1 - beta1Power))
          // model.move(along: firstMoments.scaled(by: -stepSize * sqrt(1 - beta2Power)) ./ secondMoments_h)
        } else if degeneratedToSGD {
          stepSize = 1 / (1 - beta1Power)
        } else {
          stepSize = -1
        }

        if N_sma_t >= 5 {
          let denominator = Model.TangentVector.sqrt(secondMoments).adding(epsilon)
          model.move(along: firstMoments.scaled(by: -stepSize * learningRate) ./ denominator)
        } else if stepSize > 0 {
          model.move(along: firstMoments.scaled(by: -stepSize * learningRate))
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
