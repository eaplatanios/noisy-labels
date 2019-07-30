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

    mpl.use("TkAgg")
    mpl.rc("text", usetex: false)

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
        let figure = plt.subplots()
        let ax = figure[1]
        for (learner, results) in metricResults {
          let results = results.sorted(by: { $0.0 < $1.0 })
          let x = results.map { $0.0 }
          let y = results.map { $0.1 }
          let yMean = y.map { $0.mean }
          let yStandardDeviation = y.map { $0.standardDeviation }
          ax.plot(x, yMean, label: learner)
          ax.fill_between(
            x,
            zip(yMean, yStandardDeviation).map(-),
            zip(yMean, yStandardDeviation).map(+),
            alpha: 0.35)
        }
        plt.legend()
        plt.savefig(plotsFolder.appendingPathComponent("\(dataset)-\(type)-\(metric).pdf").path)
      }
    }
  }
}
