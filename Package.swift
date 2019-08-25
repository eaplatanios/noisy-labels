// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "NoisyLabels",
  platforms: [.macOS(.v10_12)],
  products: [
    .library(name: "NoisyLabels", targets: ["NoisyLabels"]),
    .executable(name: "NoisyLabelsExperiments", targets: ["NoisyLabelsExperiments"])
  ],
  dependencies: [
    .package(url: "https://github.com/apple/swift-log.git", from: "1.0.0"),
    .package(url: "https://github.com/weichsel/ZIPFoundation/", .branch("master")),
    .package(url: "https://github.com/jkandzi/Progress.swift", from: "0.4.0"),
    .package(url: "https://github.com/apple/swift-package-manager.git", from: "0.4.0")
  ],
  targets: [
    .target(name: "NoisyLabels", dependencies: ["Logging", "ZIPFoundation"]),
    .target(name: "NoisyLabelsExperiments", dependencies: ["NoisyLabels", "Progress", "SPMUtility"]),
  ]
)
