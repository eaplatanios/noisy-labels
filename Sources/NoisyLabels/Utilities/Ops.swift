
import TensorFlow

public struct Pair<Element: Differentiable>: Differentiable {
  public var first: Element
  public var second: Element

  @inlinable
  public init(first: Element, second: Element) {
    self.first = first
    self.second = second
  }
}

@inlinable
@differentiable(vjp: _vjpDifferentiableZip)
internal func differentiableZip<Element: Differentiable>(
  _ array1: [Element],
  _ array2: [Element]
) -> [Pair<Element>] {
  zip(array1, array2).map { Pair(first: $0, second: $1) }
}

@inlinable
internal func _vjpDifferentiableZip<Element: Differentiable>(
  _ array1: [Element],
  _ array2: [Element]
) -> ([Pair<Element>], (Array<Pair<Element>>.TangentVector) -> (
  Array<Element>.TangentVector,
  Array<Element>.TangentVector
)) {
  (differentiableZip(array1, array2), { v in
    var array1 = [Element.TangentVector](repeating: Element.TangentVector.zero, count: array1.count)
    var array2 = [Element.TangentVector](repeating: Element.TangentVector.zero, count: array2.count)
    for i in v.base.indices {
      if i < array1.count { array1[i] = v[i].first }
      if i < array2.count { array2[i] = v[i].second }
    }
    return (Array<Element>.TangentVector(array1), Array<Element>.TangentVector(array2))
  })
}

// public extension Array where Element: Differentiable {
//   @differentiable(wrt: self, vjp: _vjpDifferentiableMap1)
//   func differentiableMap<Result: Differentiable>(
//     _ body: @differentiable (Element) -> Result
//   ) -> [Result] {
//     map(body)
//   }

//   @usableFromInline
//   internal func _vjpDifferentiableMap1<Result: Differentiable>(
//     _ body: @differentiable (Element) -> Result
//   ) -> ([Result], (Array<Result>.TangentVector) -> Array.TangentVector) {
//     var values: [Result] = []
//     var pullbacks: [(Result.TangentVector) -> Element.TangentVector] = []
//     for x in self {
//       let (y, pb) = Swift.valueWithPullback(at: x, in: body)
//       values.append(y)
//       pullbacks.append(pb)
//     }
//     func pullback(_ tans: Array<Result>.TangentVector) -> Array.TangentVector {
//       .init(zip(tans.base, pullbacks).map { tan, pb in pb(tan) })
//     }
//     return (values, pullback)
//   }
// }
