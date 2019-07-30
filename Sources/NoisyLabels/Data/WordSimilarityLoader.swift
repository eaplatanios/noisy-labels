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

/// Word similarity Amazon Mechanical Turk dataset loader.
///
/// Source:
/// - https://sites.google.com/site/nlpannotations
public struct WordSimilarityLoader: DataLoader {
  private let url: URL = URL(
    string: "https://dl.dropboxusercontent.com/s/wbgaob9t42cas37/wordsim.zip")!
  private let gloveFeaturesURL: URL = URL(
    string: "https://dl.dropboxusercontent.com/s/tnyr4hsaf5q4va9/wordsim_glove_features.zip")!
  private let bertFeaturesURL: URL = URL(
    string: "https://dl.dropboxusercontent.com/s/wuuwmzghm75wrov/wordsim_bert_features.zip")!

  public let dataDir: URL
  public let features: Features

  public init(dataDir: URL, features: Features = .glove) {
    self.dataDir = dataDir
    self.features = features
  }

  public func load(withFeatures: Bool = true) throws -> Data<Int, String, Int> {
    logger.info("Loading the word similarity dataset.")

    let dataDir = self.dataDir.appendingPathComponent("wordsim")
    let compressedFile = dataDir.appendingPathComponent("wordsim.zip")

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
      let value = Float(parts[3])! / 10.0
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

    var instanceFeatures: [Tensor<Float>]? = nil
    if withFeatures {
      logger.info("Loading the word similarity dataset features.")
      let compressedFeaturesFile = dataDir.appendingPathComponent(
        "wordsim_\(features.rawValue)_features.zip")
      let featuresURL = { () -> URL in
        switch self.features {
        case .glove: return gloveFeaturesURL
        case .bert: return bertFeaturesURL
        }
      }()
      try maybeDownload(from: featuresURL, to: compressedFeaturesFile)
      let extractedFeaturesDir = compressedFeaturesFile.deletingPathExtension()
      if !FileManager.default.fileExists(atPath: extractedFeaturesDir.path) {
        try FileManager.default.unzipItem(at: compressedFeaturesFile, to: extractedFeaturesDir)
      }
      let featuresFile = extractedFeaturesDir.appendingPathComponent(
        "\(features.rawValue)_features.txt")
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

extension WordSimilarityLoader {
  public enum Features: String {
    case glove, bert
  }
}
