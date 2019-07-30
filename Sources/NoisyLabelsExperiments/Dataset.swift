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

public protocol Dataset: CustomStringConvertible {
  associatedtype Loader: DataLoader
  var loader: (Foundation.URL) -> Loader { get }
  var runs: [ExperimentRun] { get }
}

public struct RTEDataset: Dataset {
  public var description: String = "rte"
  public var loader: (Foundation.URL) -> RTELoader = { RTELoader(dataDir: $0) }
  public var runs: [ExperimentRun] = [
    .predictorSubsampling(predictorCount: 1, repetitionCount: 50),
    .predictorSubsampling(predictorCount: 10, repetitionCount: 50),
    .predictorSubsampling(predictorCount: 20, repetitionCount: 50),
    .predictorSubsampling(predictorCount: 50, repetitionCount: 50),
    .predictorSubsampling(predictorCount: 100, repetitionCount: 20),
    .predictorSubsampling(predictorCount: 164, repetitionCount: 10),
    .redundancy(maxRedundancy: 1, repetitionCount: 10),
    .redundancy(maxRedundancy: 2, repetitionCount: 10),
    .redundancy(maxRedundancy: 3, repetitionCount: 10),
    .redundancy(maxRedundancy: 4, repetitionCount: 10),
    .redundancy(maxRedundancy: 5, repetitionCount: 10),
    .redundancy(maxRedundancy: 6, repetitionCount: 10),
    .redundancy(maxRedundancy: 7, repetitionCount: 10),
    .redundancy(maxRedundancy: 8, repetitionCount: 10),
    .redundancy(maxRedundancy: 9, repetitionCount: 10),
    .redundancy(maxRedundancy: 10, repetitionCount: 10)]
}
