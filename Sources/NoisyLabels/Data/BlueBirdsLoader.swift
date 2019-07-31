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

/// BlueBirds dataset loader.
///
/// Source: https://github.com/welinder/cubam/tree/public/demo/bluebirds
public struct BlueBirdsLoader: DataLoader {
  private let url: URL = URL(
    string: "https://dl.dropboxusercontent.com/s/n5l3x6bdb9ihlon/bluebirds.zip")!
  private let featuresURL: URL = URL(
    string: "https://dl.dropboxusercontent.com/s/c6svvrowgekbwmd/bluebirds_features.zip")!

  public let dataDir: URL

  public init(dataDir: URL) {
    self.dataDir = dataDir
  }

  public func load(withFeatures: Bool = true) throws -> Data<Int, String, Int> {
    logger.info("Loading the BlueBirds dataset.")

    let dataDir = self.dataDir.appendingPathComponent("bluebirds")
    let compressedFile = dataDir.appendingPathComponent("bluebirds.zip")

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

    var instanceFeatures: [Tensor<Float>]? = nil
    if withFeatures {
      logger.info("Loading the BlueBirds dataset features.")
      let compressedFeaturesFile = dataDir.appendingPathComponent("bluebirds_features.zip")
      try maybeDownload(from: featuresURL, to: compressedFeaturesFile)
      let extractedFeaturesDir = compressedFeaturesFile.deletingPathExtension()
      if !FileManager.default.fileExists(atPath: extractedFeaturesDir.path) {
        try FileManager.default.unzipItem(at: compressedFeaturesFile, to: extractedFeaturesDir)
      }
      let featuresFile = extractedFeaturesDir.appendingPathComponent("features.txt")
      let featuresString = try String(contentsOf: featuresFile, encoding: .utf8)
      var features = [Int: Tensor<Float>]()
      for line in featuresString.components(separatedBy: .newlines).filter({ !$0.isEmpty }) {
        let lineParts = line.components(separatedBy: "\t")
        let instance = Int(lineParts[0])!
        let values = lineParts[1].components(separatedBy: " ").map { Float($0)! }
        features[instance] = Tensor(values)
      }
      instanceFeatures = instances.map { features[$0]! }
    }

    return Data(
      instances: instances,
      predictors: predictors,
      labels: [0],
      trueLabels: [0: trueLabels],
      predictedLabels: [0: predictedLabels],
      classCounts: [2],
      instanceFeatures: instanceFeatures)
  }
}
