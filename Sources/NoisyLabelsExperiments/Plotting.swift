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
import Python

fileprivate let mpl = Python.import("matplotlib")
fileprivate let plt = Python.import("matplotlib.pyplot")
fileprivate let sns = Python.import("seaborn")

fileprivate let redPalette = sns.color_palette("Reds", 4)
fileprivate let greenPalette = sns.color_palette("Greens", 4)
fileprivate let bluePalette = sns.color_palette("Blues_d", 4)

public struct ResultsPlotter {
  public let resultsFile: URL

  public var dataset: String { resultsFile.deletingPathExtension().lastPathComponent }

  public init(forFile resultsFile: URL) {
    self.resultsFile = resultsFile
  }

  public func plot() throws {
    let results = try String(
      contentsOf: resultsFile,
      encoding: .utf8
    ).components(separatedBy: .newlines).dropFirst()

    // First, we parse the loaded results into a dictionary that looks like this:
    // [
    //   .redundancy: [
    //     .accuracy: [
    //       "MAJ": [1: [0.80, 0.78, 0.81, 0.83], 2: [...]), ...],
    //       "MAJ-S": [1: [0.82, 0.79, 0.80, 0.85], 2: [...]), ...],
    //       ...
    //     ],
    //     .auc: ...,
    //     ...
    //   ],
    //   .predictorCount: ...
    // ]
    var parsed = [ExperimentResult.ParameterType: [Metric: [String: [Int: [Float]]]]]()
    for result in results {
      let parts = result.components(separatedBy: "\t")
      if parts.count < 6 { continue }
      let l = parts[1]
      let t = ExperimentResult.ParameterType(rawValue: parts[2])!
      let p = Int(parts[3])!
      let m = Metric(rawValue: parts[4])!
      let v = Float(parts[5])!
      if !parsed.keys.contains(t) { parsed[t] = [Metric: [String: [Int: [Float]]]]() }
      if !parsed[t]!.keys.contains(m) { parsed[t]![m] = [String: [Int: [Float]]]()}
      if !parsed[t]![m]!.keys.contains(l) { parsed[t]![m]![l] = [Int: [Float]]() }
      if !parsed[t]![m]![l]!.keys.contains(p) { parsed[t]![m]![l]![p] = [Float]() }
      parsed[t]![m]![l]![p]!.append(v)
    }

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    sns.set()
    sns.set_context("paper")
    sns.set_style("white")
    sns.set_style("ticks")

    let plotsFolder = resultsFile
      .deletingLastPathComponent()
      .appendingPathComponent("figures", isDirectory: true)
    if !FileManager.default.fileExists(atPath: plotsFolder.path) {
      try FileManager.default.createDirectory(
        atPath: plotsFolder.path,
        withIntermediateDirectories: false)
    }

    // Plot all the results and save them in PDF files alongside the original results file.
    for (type, typeResults) in parsed {
      for (metric, metricResults) in typeResults {
        // Create a new figure.
        let figure = plt.figure(figsize: [5.0, 4.0])
        let ax = plt.gca()

        // Plot the curves for all the learners.
        for (learner, results) in metricResults.sorted(by: { compareLearners($0.key, $1.key) }) {
          let results = results.sorted(by: { $0.0 < $1.0 })
          let x = results.map { $0.0 }
          let y = results.map { $0.1 }
          let yMean = y.map { $0.mean }
          let yStandardDeviation = y.map { $0.standardDeviation }
          let color = learnerColor(learner)
          ax.plot(x, yMean, label: learner, color: color, linewidth: 2)
          ax.fill_between(
            x,
            zip(yMean, yStandardDeviation).map(-),
            zip(yMean, yStandardDeviation).map(+),
            color: color,
            alpha: 0.1,
            linewidth: 0)
          ax.axhline(y: yMean.max(), linestyle: "dashed", alpha: 0.7, color: color)
        }

        // Add axis labels.
        ax.set_xlabel(
          type.axisTitle,
          color: "grey",
          fontname: "Lato",
          fontsize: 18,
          fontweight: "light")
        ax.set_ylabel(
          metric.axisTitle,
          color: "grey",
          fontname: "Lato",
          fontsize: 18,
          fontweight: "light")
        ax.yaxis.set_tick_params(labelbottom: true)

        // Change the tick label sizes.
        plt.setp(ax.get_xticklabels(), fontname: "Lato", fontsize: 18, fontweight: "regular")
        plt.setp(ax.get_yticklabels(), fontname: "Lato", fontsize: 18, fontweight: "regular")

        // Set the figure title.
        figure.suptitle(
          figureTitle(for: dataset),
          x: 0.5,
          y: 1,
          fontname: "Lato",
          fontsize: 22,
          fontweight: "black")

        // Remove the grid.
        sns.despine()

        // Add a legend.
        plt.legend()

        // Save the figure.
        plt.savefig(
          plotsFolder.appendingPathComponent("\(dataset)-\(type)-\(metric).pdf").path,
          bbox_inches: "tight")
      }
    }
  }
}

extension ExperimentResult.ParameterType {
  fileprivate var axisTitle: String {
    switch self {
    case .predictorCount: return "#Predictors"
    case .redundancy: return "Redundancy"
    }
  }
}

extension Metric {
  fileprivate var axisTitle: String {
    switch self {
    case .madError: return "Error MAD"
    case .madErrorRank: return "Error Rank MAD"
    case .accuracy: return "Accuracy"
    case .auc: return "AUC"
    }
  }
}

fileprivate func figureTitle(for dataset: String) -> String {
  switch dataset {
  case "bluebirds": return "BlueBirds"
  case "word-similarity": return "Word Similarity"
  case "rte": return "RTE"
  case "age": return "Age"
  case _: return "Unknown"
  }
}

fileprivate func compareLearners(_ learner1: String, _ learner2: String) -> Bool {
  func score(_ learner: String) -> Int {
    if learner.hasPrefix("LNL-E") { return 5 }
    if learner.hasPrefix("LNL") { return 6 }
    if learner.hasPrefix("MeTaL") { return 4 }
    if learner.hasPrefix("Snorkel") { return 3 }
    if learner.hasPrefix("MMCE") { return 2 }
    if learner.hasPrefix("MAJ-S") { return 1 }
    if learner.hasPrefix("MAJ") { return 0 }
    return -1
  }

  return score(learner1) < score(learner2)
}

fileprivate func learnerColor(_ learner: String) -> PythonObject {
  if learner.hasPrefix("LNL-E") { return redPalette[0] }
  if learner.hasPrefix("LNL") { return redPalette[2] }
  if learner.hasPrefix("Snorkel") { return greenPalette[0] }
  if learner.hasPrefix("MeTaL") { return greenPalette[2] }
  if learner.hasPrefix("MMCE") { return bluePalette[1] }
  if learner.hasPrefix("MAJ-S") { return bluePalette[2] }
  if learner.hasPrefix("MAJ") { return bluePalette[3] }
  return "black"
}
