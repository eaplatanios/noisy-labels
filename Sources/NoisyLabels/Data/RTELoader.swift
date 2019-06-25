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
import ZIPFoundation

/// PASCAL RTE Amazon Mechanical Turk dataset loader.
///
/// Sources:
/// - https://sites.google.com/site/nlpannotations
/// - https://www.kaggle.com/nltkdata/rte-corpus
///
/// - Author: Emmanouil Antonios Platanios
///
public struct RTELoader: DataLoader {
  private let url: URL = URL(
    string: "https://dl.dropboxusercontent.com/s/ebkpj9a5ndy7gh5/rte.zip")!
  private let featuresUrl: URL = URL(
    string: "https://dl.dropboxusercontent.com/s/bc5dr440k6olt79/rte_features.zip")!

  public let dataDir: URL

  public init(dataDir: URL) {
    self.dataDir = dataDir
  }

  public func load(withFeatures: Bool = true) throws -> Data<Int, String, Int> {
    let dataDir = self.dataDir.appendingPathComponent("rte")
    let compressedFile = dataDir.appendingPathComponent("rte.zip")

    // Download the data, if necessary.
    try maybeDownload(from: url, to: compressedFile)

    // Extract the data, if necessary.
    let extractedDir = compressedFile.deletingPathExtension()
    if !FileManager.default.fileExists(atPath: extractedDir.path) {
      try FileManager.default.unzipItem(at: compressedFile, to: extractedDir)
    }

    // Read the original data file.
    var instances = [Int]()
    var predictors = [String]()
    var trueLabels = [Int: Int]()
    var predictedLabels = [Int: (instances: [Int], values: [Float])]()

    var instanceIds = [Int: Int]()
    var predictorIds = [String: Int]()

    let originalFile = extractedDir.appendingPathComponent("original.tsv")
    let originalContents = try String(contentsOfFile: originalFile.path, encoding: .utf8)
    for line in originalContents.split(separator: "\n").dropFirst() {
      let parts = line.split(separator: "\t")
      let instance = Int(parts[2])!
      let predictor = String(parts[1])
      let value = Float(parts[3])!
      let trueLabel = Int(parts[4])!
      
      let instanceId = instanceIds[instance] ?? {
        let id = instances.count
        instances.append(instance)
        instanceIds[instance] = id
        return id
      }()
      
      let predictorId = predictorIds[predictor] ?? {
        let id = predictors.count
        predictors.append(predictor)
        predictorIds[predictor] = id
        return id
      }()

      trueLabels[instanceId] = trueLabel
      
      if !predictedLabels.keys.contains(predictorId) {
        predictedLabels[predictorId] = (instances: [instanceId], values: [value])
      } else {
        predictedLabels[predictorId]!.instances.append(instanceId)
        predictedLabels[predictorId]!.values.append(value)
      }
    }

    return Data(
      instances: instances,
      predictors: predictors,
      labels: [0],
      trueLabels: [0: trueLabels],
      predictedLabels: [0: predictedLabels],
      numClasses: [2])
  }
}
