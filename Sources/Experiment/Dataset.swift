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

public enum Dataset: String, StringEnumArgument {
  case rte

  public static var completion: ShellCompletion {
    return .values([
      (Dataset.rte.rawValue, "")
    ])
  }

  public func loader(dataDir: Foundation.URL) -> DataLoader {
    switch self {
    case .rte: return RTELoader(dataDir: dataDir)
    }
  }

  public func predictorCounts() -> [Int] {
    switch self {
    case .rte: return [1, 10, 20, 50, 100, 164]
    }
  }

  public func repetitionCounts() -> [Int] {
    switch self {
    case .rte: return [20, 10, 10, 5, 3, 1]
    }
  }
}
