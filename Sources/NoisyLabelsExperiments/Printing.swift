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

public struct ResultsPrinter {
  public let resultsFile: URL

  public var dataset: String { resultsFile.deletingPathExtension().lastPathComponent }

  public init(forFile resultsFile: URL) {
    self.resultsFile = resultsFile
  }

  public func makeTables() throws {
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

    let tablesFolder = resultsFile
      .deletingLastPathComponent()
      .appendingPathComponent("tables", isDirectory: true)
    if !FileManager.default.fileExists(atPath: tablesFolder.path) {
      try FileManager.default.createDirectory(
        atPath: tablesFolder.path,
        withIntermediateDirectories: false)
    }

    // Plot all the results and save them in PDF files alongside the original results file.
    for (type, typeResults) in parsed {
      for (metric, metricResults) in typeResults {
        var tableRows = [String]()
        let types = metricResults.values.map { $0.keys }.max(by: { $0.count < $1.count })!.sorted()
        let firstColWidth = max(typeResults.keys.map { $0.tableTitle.count }.max() ?? 0, 7)
        let colWidth = 11  // Example: " 0.99±0.02 "

        // Example:
        // ╔═════════════════════════════════════════ ... ═════════════════════════════════╗
        // ║                                        Accuracy                               ║
        // ╟────────────────┬──────────────────────── ... ─────────────────────────────────╢
        var rowParts = [String]()
        rowParts.append("╔═\(String(repeating: "═", count: firstColWidth))══")
        rowParts.append([String](
          repeating: String(repeating: "═", count: colWidth),
          count: types.count
        ).joined(separator: "═"))
        rowParts.append("╗")
        tableRows.append(rowParts.joined())
        let tableWidth = 4 + firstColWidth + (colWidth + 1) * types.count
        let titlePosition = tableWidth / 2 - metric.tableTitle.count / 2
        var beforeSpaceCount = titlePosition - 1
        var afterSpaceCount = (tableWidth + 1) / 2 - (metric.tableTitle.count + 1) / 2 - 1
        rowParts = [String]()
        rowParts.append("║\(String(repeating: " ", count: beforeSpaceCount))")
        rowParts.append(metric.tableTitle)
        rowParts.append("\(String(repeating: " ", count: afterSpaceCount))║")
        tableRows.append(rowParts.joined())
        rowParts = [String]()
        rowParts.append("╟─\(String(repeating: "─", count: firstColWidth))─┬")
        rowParts.append([String](
          repeating: String(repeating: "─", count: colWidth),
          count: types.count
        ).joined(separator: "─"))
        rowParts.append("╢")
        tableRows.append(rowParts.joined())

        // Example:
        // ║                │                          Redundancy                          ║
        // ║     Learner    ├───────────────────────────── ... ────────────────────────────╢
        // ║                │     1           2            ...           8          10     ║
        // ╟────────────────┼───────────────────────────── ... ────────────────────────────╢
        rowParts = [String]()
        rowParts.append("║ \(String(repeating: " ", count: firstColWidth)) │")
        let width = tableWidth - firstColWidth - 5
        var position = width / 2 - type.typeTitle.count / 2
        beforeSpaceCount = position
        afterSpaceCount = (width + 1) / 2 - (type.typeTitle.count + 1) / 2
        rowParts.append("\(String(repeating: " ", count: beforeSpaceCount))")
        rowParts.append(type.typeTitle)
        rowParts.append("\(String(repeating: " ", count: afterSpaceCount))║")
        tableRows.append(rowParts.joined())
        rowParts = [String]()
        position = firstColWidth / 2 - 2
        beforeSpaceCount = position
        afterSpaceCount = (firstColWidth + 1) / 2 - 3
        rowParts.append("║\(String(repeating: " ", count: beforeSpaceCount))")
        rowParts.append("Learner")
        rowParts.append("\(String(repeating: " ", count: afterSpaceCount))├")
        rowParts.append([String](
          repeating: String(repeating: "─", count: colWidth),
          count: types.count
        ).joined(separator: "─"))
        rowParts.append("╢")
        tableRows.append(rowParts.joined())
        rowParts = [String]()
        rowParts.append("║ \(String(repeating: " ", count: firstColWidth)) ")
        var first = true
        for type in types {
          let t = String(type)
          position = colWidth / 2 - t.count / 2
          beforeSpaceCount = position
          afterSpaceCount = (colWidth + 1) / 2 - (t.count + 1) / 2
          if first {
            rowParts.append("│\(String(repeating: " ", count: beforeSpaceCount))")
            first = false
          } else {
            rowParts.append(" \(String(repeating: " ", count: beforeSpaceCount))")
          }
          rowParts.append(t)
          rowParts.append("\(String(repeating: " ", count: afterSpaceCount))")
        }
        rowParts.append("║")
        tableRows.append(rowParts.joined())
        rowParts = [String]()
        rowParts.append("╟─\(String(repeating: "─", count: firstColWidth))─┼")
        rowParts.append([String](
          repeating: String(repeating: "─", count: colWidth),
          count: types.count
        ).joined(separator: "─"))
        rowParts.append("╢")
        tableRows.append(rowParts.joined())

        // Example:
        // ║            MAJ │ 0.59±0.01   0.58±0.01        ...       0.63±0.00   0.63±0.00 ║
        // ║           MMCE │ 0.16±0.01   0.21±0.01        ...       0.58±0.01   0.61±0.01 ║
        // ║        SNORKEL │ 0.56±0.01   0.58±0.02        ...       0.65±0.01   0.65±0.00 ║
        for (learner, results) in metricResults.sorted(by: { compareLearners($0.key, $1.key) }) {
          rowParts = [String]()
          position = firstColWidth - learner.count + 1
          rowParts.append("║\(String(repeating: " ", count: position))")
          rowParts.append("\(learner) ")
          let results = results.sorted(by: { $0.0 < $1.0 })
          let x = results.map { $0.0 }
          let y = results.map { $0.1 }
          let yMean = y.map { String(format: "%.2f", $0.mean) }
          let yStandardDeviation = y.map { String(format: "%.2f", $0.standardDeviation) }
          let ys = zip(yMean, yStandardDeviation).map { "\($0)±\($1)" }
          first = true
          var i = 0
          for type in types {
            if i < x.count && x[i] == type {
              position = colWidth / 2 - ys[i].count / 2
              beforeSpaceCount = position
              afterSpaceCount = (colWidth + 1) / 2 - (ys[i].count + 1) / 2
              if first {
                rowParts.append("│\(String(repeating: " ", count: beforeSpaceCount))")
                first = false
              } else {
                rowParts.append(" \(String(repeating: " ", count: beforeSpaceCount))")
              }
              rowParts.append(ys[i])
              rowParts.append("\(String(repeating: " ", count: afterSpaceCount))")
              i += 1
            } else {
              if first {
                rowParts.append("│\(String(repeating: " ", count: colWidth))")
                first = false
              } else {
                rowParts.append(" \(String(repeating: " ", count: colWidth))")
              }
            }
          }
          rowParts.append("║")
          tableRows.append(rowParts.joined())
        }

        // Example:
        // ╚════════════════╧═════════════════════════════ ... ════════════════════════════╝
        rowParts = [String]()
        rowParts.append("╚═\(String(repeating: "═", count: firstColWidth))═╧")
        rowParts.append([String](
          repeating: String(repeating: "═", count: colWidth),
          count: types.count
        ).joined(separator: "═"))
        rowParts.append("╝")
        tableRows.append(rowParts.joined())

        // Finally, write the table to the appropriate file.
        let table = tableRows.joined(separator: "\n")
        let file = tablesFolder.appendingPathComponent("\(dataset)-\(type)-\(metric).txt")
        try table.write(to: file, atomically: true, encoding: String.Encoding.utf8)
      }
    }
  }
}

extension ExperimentResult.ParameterType {
  fileprivate var typeTitle: String {
    switch self {
    case .predictorCount: return "#Predictors"
    case .redundancy: return "Redundancy"
    }
  }
}

extension Metric {
  fileprivate var tableTitle: String {
    switch self {
    case .madError: return "Error MAD"
    case .madErrorRank: return "Error Rank MAD"
    case .accuracy: return "Accuracy"
    case .auc: return "AUC"
    }
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
