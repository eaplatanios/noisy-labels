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
import TensorFlow
import ZIPFoundation

/// Sentiment popularity Amazon Mechanical Turk dataset loader.
///
/// Source: https://eprints.soton.ac.uk/376544/1/SP_amt.csv
public struct SentimentPopularityLoader: DataLoader {
  private let url: URL = URL(
    string: "https://dl.dropboxusercontent.com/s/yau8sn7oo2n7288/sentiment_popularity.zip")!

  public let dataDir: URL

  public init(dataDir: URL) {
    self.dataDir = dataDir
  }

  public func load(withFeatures: Bool = false) throws -> Data<Int, String, Int> {
    precondition(
      withFeatures == false,
      "The sentiment popularity dataset does not provide instance features.")

    logger.info("Loading the sentiment popularity dataset.")

    let dataDir = self.dataDir.appendingPathComponent("sentiment-popularity")
    let compressedFile = dataDir.appendingPathComponent("sentiment-popularity.zip")

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

    let originalFile = extractedDir.appendingPathComponent("original.csv")
    let originalContents = try String(contentsOfFile: originalFile.path, encoding: .utf8)
    for line in originalContents.split(separator: "\n") {
      let parts = line.split(separator: ",")
      let instance = Int(parts[1])!
      let predictor = String(parts[0])
      let value = Float(parts[2])!
      let trueLabel = Int(parts[3])!

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
      classCounts: [2])
  }
}
