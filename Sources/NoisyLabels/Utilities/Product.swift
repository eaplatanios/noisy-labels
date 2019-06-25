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

public struct Product<T, C: Collection>: IteratorProtocol, Sequence where C.Iterator.Element == T {
  /// Contains the indices of the next elements for each collection.
  private var indices: [C.Index]

  /// Contains all the collections that need to be combined.
  private var pools: [C]

  /// Indicates whether to terminate the generation.
  private var done: Bool = false

  /// Constructs a product generator for the provided collections.
  public init(_ collections: [C]) {
    self.pools = collections
    self.indices = collections.map{ $0.startIndex }
    self.done = pools.reduce(true, { $0 && $1.count == 0 })
  }

  /// Constructs a product generator for `collection` repeated `count` times.
  public init(repeating collection: C, count: Int) {
    precondition(count >= 0, "'count' must be >= 0.")
    self.init([C](repeating: collection, count: count))
  }

  /// Generates the next element in the product and returns it.
  public mutating func next() -> [T]? {
    if done { return nil }
    let element = pools.enumerated().map { $1[indices[$0]] }
    incrementLocationInPool(poolIndex: pools.count - 1)
    return element
  }

  /// Increments the indices of the collections in the pools. The index corresponding to location in 
  /// the pool for the provided `poolIndex` will be incremented. If the new location is out of 
  /// range, the location is set to zero and the location in the preceding pool is incremented. If 
  /// `poolIndex` is less than zero, then the generation is complete.
  mutating private func incrementLocationInPool(poolIndex: Int) {
    guard pools.indices.contains(poolIndex) else { done = true; return }
    indices[poolIndex] = pools[poolIndex].index(after: indices[poolIndex])
    if indices[poolIndex] == pools[poolIndex].endIndex {
      indices[poolIndex] = pools[poolIndex].startIndex
      incrementLocationInPool(poolIndex: poolIndex - 1)
    }
  }
}
