// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "NoisyLabels",
  platforms: [.macOS(.v10_12)],
  products: [
    .library(name: "NoisyLabels", targets: ["NoisyLabels"]),
    .executable(name: "Experiment", targets: ["Experiment"])
  ],
  dependencies: [
    .package(url: "https://github.com/weichsel/ZIPFoundation/", .branch("master")),
    .package(url: "https://github.com/SwiftyBeaver/SwiftyBeaver.git", from: "1.7.0"),
    .package(url: "https://github.com/jkandzi/Progress.swift", from: "0.4.0"),
    .package(url: "https://github.com/apple/swift-package-manager.git", from: "0.1.0")
  ],
  targets: [
    .target(
      name: "NoisyLabels",
      dependencies: ["SwiftyBeaver", "ZIPFoundation"],
      linkerSettings: [.linkedLibrary("tensorflow")]),
    .target(name: "Experiment", dependencies: ["NoisyLabels", "Progress", "Utility"]),
  ]
)
