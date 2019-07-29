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

import Foundation
import NoisyLabels
import Utility

public enum Dataset: String {
  case blueBirds, rte, wordSimilarity

  public var loader: (Foundation.URL) -> DataLoader {
    switch self {
    case .rte: return { RTELoader(dataDir: $0) }
    case _: fatalError("The selected dataset is not supported yet.")
    }
  }

  public var runs: [Experiment.Run] {
    switch self {
    case .blueBirds: return [
      .simple(predictorCount: 1, repetitionCount: 10),
      .simple(predictorCount: 2, repetitionCount: 10),
      .simple(predictorCount: 5, repetitionCount: 10),
      .simple(predictorCount: 10, repetitionCount: 10),
      .simple(predictorCount: 20, repetitionCount: 10),
      .simple(predictorCount: 39, repetitionCount: 10),
      .redundancy(max: 1, repetitionCount: 10),
      .redundancy(max: 2, repetitionCount: 10),
      .redundancy(max: 5, repetitionCount: 10),
      .redundancy(max: 10, repetitionCount: 10),
      .redundancy(max: 20, repetitionCount: 10),
      .redundancy(max: 39, repetitionCount: 10)]
    case .rte: return [
      .simple(predictorCount: 1, repetitionCount: 50),
      .simple(predictorCount: 10, repetitionCount: 50),
      .simple(predictorCount: 20, repetitionCount: 50),
      .simple(predictorCount: 50, repetitionCount: 50),
      .simple(predictorCount: 100, repetitionCount: 20),
      .simple(predictorCount: 164, repetitionCount: 10),
      .redundancy(max: 1, repetitionCount: 10),
      .redundancy(max: 2, repetitionCount: 10),
      .redundancy(max: 3, repetitionCount: 10),
      .redundancy(max: 4, repetitionCount: 10),
      .redundancy(max: 5, repetitionCount: 10),
      .redundancy(max: 6, repetitionCount: 10),
      .redundancy(max: 7, repetitionCount: 10),
      .redundancy(max: 8, repetitionCount: 10),
      .redundancy(max: 9, repetitionCount: 10),
      .redundancy(max: 10, repetitionCount: 10)]
    case .wordSimilarity: return [
      .simple(predictorCount: 1, repetitionCount: 50),
      .simple(predictorCount: 2, repetitionCount: 50),
      .simple(predictorCount: 5, repetitionCount: 50),
      .simple(predictorCount: 10, repetitionCount: 10),
      .redundancy(max: 1, repetitionCount: 10),
      .redundancy(max: 2, repetitionCount: 10),
      .redundancy(max: 3, repetitionCount: 10),
      .redundancy(max: 4, repetitionCount: 10),
      .redundancy(max: 5, repetitionCount: 10),
      .redundancy(max: 6, repetitionCount: 10),
      .redundancy(max: 7, repetitionCount: 10),
      .redundancy(max: 8, repetitionCount: 10),
      .redundancy(max: 9, repetitionCount: 10),
      .redundancy(max: 10, repetitionCount: 10)]
    }
  }
}

extension Dataset: StringEnumArgument {
  public static var completion: ShellCompletion {
    return .values([
      (Dataset.rte.rawValue, "")
    ])
  }
}

extension Experiment {
  public enum Run {
    case simple(predictorCount: Int, repetitionCount: Int)
    case redundancy(max: Int, repetitionCount: Int)
  }
}
