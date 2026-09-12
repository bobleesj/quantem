// swift-tools-version: 6.0
import Foundation
import PackageDescription

let infrastructure: Package.Dependency =
  ProcessInfo.processInfo.environment["QUANTEM_GPU_PACKAGE"].map {
    .package(path: $0)
  } ?? .package(url: "https://github.com/bobleesj/quantem.gpu.git", branch: "main")
let infrastructureName =
  ProcessInfo.processInfo.environment["QUANTEM_GPU_PACKAGE"].map {
    URL(fileURLWithPath: $0).lastPathComponent
  } ?? "quantem.gpu"

let package = Package(
  name: "QuantEMNative",
  platforms: [.macOS(.v14)],
  products: [
    .library(name: "QuantEMMAPED", targets: ["QuantEMMAPED"]),
    .executable(name: "maped-native-benchmark", targets: ["MAPEDNativeBenchmark"]),
  ],
  dependencies: [infrastructure],
  targets: [
    .target(
      name: "QuantEMMAPED",
      dependencies: [
        .product(name: "MetalScientificNumerics", package: infrastructureName),
        .product(name: "Metal4DSTEMStreamingIO", package: infrastructureName),
      ],
      path: "native/Sources/QuantEMMAPED"),
    .executableTarget(
      name: "MAPEDNativeBenchmark", dependencies: ["QuantEMMAPED"],
      path: "native/Benchmarks/MAPEDNativeBenchmark"),
    .testTarget(
      name: "QuantEMMAPEDTests", dependencies: ["QuantEMMAPED"],
      path: "native/Tests/QuantEMMAPEDTests", resources: [.copy("Fixtures")]),
  ])
