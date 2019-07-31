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

/// Medical relations dataset loader.
///
/// The task is: given a sentence with 2 medical terms, assign potentially multiple binary labels
/// that indicate relations between the two terms. Notes:
///   - We ignore directionality of the relations.
///   - There are multiple relations that crowdworkers could pick from. However, we have ground
///     truth labels for only a subset of relations (i.e., "CAUSES" and "TREATS") and so we only
///     load data relevant for these two relations.
///
/// Source: https://github.com/CrowdTruth/Medical-Relation-Extraction
public struct MedicalRelationsLoader: DataLoader {
  private let url: URL = URL(
    string: "https://dl.dropboxusercontent.com/s/3q97sd499e2z991/medical_relations.zip")!
  private let featuresURL: URL = URL(
    string: "https://dl.dropboxusercontent.com/s/tnyr4hsaf5q4va9/wordsim_glove_features.zip")!

  public let dataDir: URL
  public let labels: [Label]

  public init(dataDir: URL, labels: [Label] = Label.allCases) {
    self.dataDir = dataDir
    self.labels = labels
  }

  public func load(withFeatures: Bool = true) throws -> Data<String, String, Label> {
    logger.info("Loading the medical relations dataset.")

    let dataDir = self.dataDir.appendingPathComponent("medical-relations")
    let compressedFile = dataDir.appendingPathComponent("medical-relations.zip")

    // Download the data, if necessary.
    try maybeDownload(from: url, to: compressedFile)

    // Extract the data, if necessary.
    let extractedDir = compressedFile.deletingPathExtension()
    if !FileManager.default.fileExists(atPath: extractedDir.path) {
      try FileManager.default.unzipItem(at: compressedFile, to: extractedDir)
    }

    // Read the original data file.
    var instances = [String]()
    var predictors = [String]()
    var trueLabels = [Int: [Int: Int]]()
    var predictedLabels = [Int: [Int: (instances: [Int], values: [Float])]]()
    var instanceIds = [String: Int]()
    var predictorIds = [String: Int]()

    for (l, label) in labels.enumerated() {
      trueLabels[l] = [Int: Int]()
      predictedLabels[l] = [Int: (instances: [Int], values: [Float])]()
      let originalFile = extractedDir.appendingPathComponent("original_\(label).tsv")
      let originalContents = try String(contentsOfFile: originalFile.path, encoding: .utf8)
      for line in originalContents.split(separator: "\n") {
        let parts = line.split(separator: "\t")
        let instance = String(parts[1])
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

        if trueLabel >= 0 { trueLabels[l]![instanceId] = trueLabel }
        if !predictedLabels[l]!.keys.contains(predictorId) {
          predictedLabels[l]![predictorId] = (instances: [instanceId], values: [value])
        } else {
          predictedLabels[l]![predictorId]!.instances.append(instanceId)
          predictedLabels[l]![predictorId]!.values.append(value)
        }
      }
    }

    var instanceFeatures: [Tensor<Float>]? = nil
    if withFeatures {
      logger.info("Loading the medical relations dataset features.")
      fatalError("The medical relations dataset does not support instance features yet.")
      // let compressedFeaturesFile = dataDir.appendingPathComponent(
      //   "wordsim_\(features.rawValue)_features.zip")
      // let featuresURL = { () -> URL in
      //   switch self.features {
      //   case .glove: return gloveFeaturesURL
      //   case .bert: return bertFeaturesURL
      //   }
      // }()
      // try maybeDownload(from: featuresURL, to: compressedFeaturesFile)
      // let extractedFeaturesDir = compressedFeaturesFile.deletingPathExtension()
      // if !FileManager.default.fileExists(atPath: extractedFeaturesDir.path) {
      //   try FileManager.default.unzipItem(at: compressedFeaturesFile, to: extractedFeaturesDir)
      // }
      // let featuresFile = extractedFeaturesDir.appendingPathComponent(
      //   "\(features.rawValue)_features.txt")
      // let featuresString = try String(contentsOf: featuresFile, encoding: .utf8)
      // var features = [String: Tensor<Float>]()
      // for line in featuresString.components(separatedBy: .newlines).filter({ !$0.isEmpty }) {
      //   let lineParts = line.components(separatedBy: "\t")
      //   let instance = String(lineParts[0])
      //   let values = lineParts[1].components(separatedBy: " ").map { Float($0)! }
      //   features[instance] = Tensor(values)
      // }
      // instanceFeatures = instances.map { features[$0]! }
    }

    return Data(
      instances: instances,
      predictors: predictors,
      labels: labels,
      trueLabels: trueLabels,
      predictedLabels: predictedLabels,
      classCounts: [Int](repeating: 2, count: labels.count),
      instanceFeatures: instanceFeatures)
  }
}

extension MedicalRelationsLoader {
  public enum Label: String, CaseIterable {
    case causes, treats
  }
}
