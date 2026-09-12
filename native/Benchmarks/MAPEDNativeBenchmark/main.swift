import Foundation
import Metal
import Metal4DSTEMStreamingIO
import MetalCountResources
import MetalScientificNumerics
import Native4DSTEMIO
import QuantEMMAPED

let arguments = CommandLine.arguments
if arguments.count == 3, arguments[1] == "--precision" {
  let bytes = try Data(contentsOf: URL(fileURLWithPath: arguments[2]), options: .mappedIfSafe)
  let operations = try MetalImageOperations()
  let precision = try MetalPrecision(device: operations.device)
  let values = operations.device.makeBuffer(length: bytes.count, options: .storageModeShared)!
  bytes.withUnsafeBytes { _ = memcpy(values.contents(), $0.baseAddress!, $0.count) }
  let count = bytes.count / 4
  try precision.includeRange(values, count: count)
  try precision.calibrate(shape: [1, 1, 1, count])
  _ = try precision.convert(values, count: count)
  var seconds: [Double] = []
  for _ in 0..<5 {
    try autoreleasepool {
      let started = Date.timeIntervalSinceReferenceDate
      _ = try precision.convert(values, count: count)
      seconds.append(Date.timeIntervalSinceReferenceDate - started)
    }
  }
  print("Precision region: \(count) float32 values; seconds \(seconds)")
  exit(0)
}
guard arguments.count >= 3 else {
  fatalError("Usage: maped-native-benchmark INPUT_DIRECTORY REPORT_JSON")
}
let input = URL(fileURLWithPath: arguments[1])
let report = URL(fileURLWithPath: arguments[2])
let files = try FileManager.default.contentsOfDirectory(at: input, includingPropertiesForKeys: nil)
  .filter { $0.lastPathComponent.hasSuffix("_master.h5") }.sorted { $0.path < $1.path }
guard files.count == 7 else { fatalError("The seven-tilt benchmark requires seven master files.") }
// Keep all seven acquisitions resident while timing the normal sequential loader.
// This diagnostic mode excludes alignment, merging, validation exports, and UI.
if let requested = ProcessInfo.processInfo.environment["MAPED_LOAD_PASSES"],
  let passes = Int(requested), passes > 0
{
  var runs: [[String: Any]] = []
  for pass in 0..<passes {
    try autoreleasepool {
      let start = Date.timeIntervalSinceReferenceDate
      let operations = try MetalImageOperations()
      let setupSeconds = Date.timeIntervalSinceReferenceDate - start
      let index = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("QuantEM/Index", isDirectory: true)
      let catalog = Native4DSTEMCatalogBuilder(cacheDirectory: index)
      var sources: [MetalEncodedSource] = []
      defer { sources.forEach { $0.releaseResidentStorage() } }
      var tilts: [[String: Any]] = []
      for (tilt, file) in files.enumerated() {
        let begin = Date.timeIntervalSinceReferenceDate
        let prepared = try catalog.prepare(input: file)
        guard prepared.datasets.count == 1 else {
          fatalError("Expected one acquisition per master file.")
        }
        let indexed = try Native4DSTEMIndexedSource.open(dataset: prepared.datasets[0])
        let indexSeconds = Date.timeIntervalSinceReferenceDate - begin
        let loading = Date.timeIntervalSinceReferenceDate
        let source = try MetalEncodedSource.load(source: indexed, device: operations.device)
        sources.append(source)
        tilts.append([
          "tilt": tilt, "index_seconds": indexSeconds,
          "resident_preparation_seconds": Date.timeIntervalSinceReferenceDate - loading,
          "total_seconds": Date.timeIntervalSinceReferenceDate - begin,
          "resident_bytes": source.residentBytes, "source_read_passes": source.sourceReadPasses,
        ])
      }
      let run: [String: Any] = [
        "pass": pass, "total_seconds": Date.timeIntervalSinceReferenceDate - start,
        "setup_seconds": setupSeconds, "tilts": tilts,
        "input_resident_bytes": sources.reduce(0) { $0 + $1.residentBytes },
        "peak_metal_bytes": sources.map(\.peakAllocatedBytes).max() ?? 0,
        "device": operations.device.name,
      ]
      runs.append(run)
      print(
        String(
          data: try JSONSerialization.data(withJSONObject: run, options: [.sortedKeys]),
          encoding: .utf8)!)
    }
  }
  try JSONSerialization.data(
    withJSONObject: ["load_runs": runs], options: [.prettyPrinted, .sortedKeys]
  )
  .write(to: report)
  exit(0)
}
let started = Date.timeIntervalSinceReferenceDate
let maped = try MAPEDNative.from_files(files)
print(
  "Encoded inputs ready: \(maped.sources.reduce(0) { $0 + $1.residentBytes }) bytes",
  terminator: "\n")
try maped.preprocess()
try maped.diffraction_origin(sigma: 1)
try maped.diffraction_align(edge_blend: 2)
try maped.real_space_align(num_iter: 20, edge_blend: 5, padding: 2, hanning_filter: true)
let alignmentWall = Date.timeIntervalSinceReferenceDate - started
var document: [String: Any] = [
  "timings": maped.timings, "alignment_wall_seconds": alignmentWall,
  "peak_metal_bytes": maped.peak_metal_bytes, "shape": maped.shape,
  "input_resident_bytes": maped.sources.reduce(0) { $0 + $1.residentBytes },
  "source_read_passes": maped.sources.map {
    ($0 as? MetalEncodedSource)?.sourceReadPasses as Any? ?? NSNull()
  },
  "diffraction_origins": maped.diffraction_origins,
  "diffraction_shifts": maped.diffraction_shifts!.values(),
  "real_space_shifts": maped.real_space_shifts!.values(),
]
if ProcessInfo.processInfo.environment["MAPED_FULL_PARITY"] == "1" {
  let original = ProcessInfo.processInfo.environment["QUANTEM_GPU_SAMPLING_REFERENCE"]
  setenv("QUANTEM_GPU_SAMPLING_REFERENCE", "1", 1)
  let referenceOperations = try MetalImageOperations()
  if let original {
    setenv("QUANTEM_GPU_SAMPLING_REFERENCE", original, 1)
  } else {
    unsetenv("QUANTEM_GPU_SAMPLING_REFERENCE")
  }
  let reference = try MAPEDNative.from_resident(maped.sources, operations: referenceOperations)
  defer { reference.close() }
  try reference.preprocess()
  try reference.diffraction_origin(sigma: 1)
  try reference.diffraction_align(edge_blend: 2)
  try reference.real_space_align(num_iter: 20, edge_blend: 5, padding: 2, hanning_filter: true)
  guard
    reference.diffraction_shifts!.values().map(\.bitPattern)
      == maped.diffraction_shifts!.values().map(\.bitPattern),
    reference.real_space_shifts!.values().map(\.bitPattern)
      == maped.real_space_shifts!.values().map(\.bitPattern)
  else { fatalError("Reference and candidate alignment differ.") }
  let library = try maped.operations.device.makeLibrary(
    source: MetalCountResources.source("resident_utilities"), options: nil)
  let pipeline = try maped.operations.device.makeComputePipelineState(
    function: library.makeFunction(name: "count_verify")!)
  let error = maped.operations.device.makeBuffer(length: 4, options: .storageModeShared)!
  error.contents().storeBytes(of: UInt32(0), as: UInt32.self)
  let began = Date.timeIntervalSinceReferenceDate
  for row in stride(from: 0, to: maped.shape[0], by: max(1, 4096 / maped.shape[1])) {
    try autoreleasepool {
      let rows = row..<min(maped.shape[0], row + max(1, 4096 / maped.shape[1]))
      let expected = try reference.merged_region(rows)
      let actual = try maped.merged_region(rows)
      let command = maped.operations.queue.makeCommandBuffer()!
      let encoder = command.makeComputeCommandEncoder()!
      encoder.setComputePipelineState(pipeline)
      encoder.setBuffer(expected.buffer, offset: 0, index: 0)
      encoder.setBuffer(actual.buffer, offset: 0, index: 1)
      encoder.setBuffer(error, offset: 0, index: 2)
      var bytes = UInt32(expected.rows * expected.columns * 4)
      encoder.setBytes(&bytes, length: 4, index: 3)
      encoder.dispatchThreads(
        MTLSize(width: Int(bytes), height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
      encoder.endEncoding()
      command.commit()
      command.waitUntilCompleted()
      guard command.status == .completed, error.contents().load(as: UInt32.self) == 0 else {
        fatalError("Float32 bit parity failed in scan rows \(rows).")
      }
    }
  }
  document["full_float32_parity"] =
    [
      "values": maped.shape.reduce(1, *), "exact": true,
      "seconds": Date.timeIntervalSinceReferenceDate - began,
    ] as [String: Any]
}
if let requested = ProcessInfo.processInfo.environment["MAPED_PROCESSING_PASSES"],
  let passes = Int(requested), passes > 0
{
  var seconds: [Double] = []
  for _ in 0..<passes {
    let start = Date.timeIntervalSinceReferenceDate
    for row in stride(from: 0, to: maped.shape[0], by: max(1, 4096 / maped.shape[1])) {
      try autoreleasepool {
        _ = try maped.merged_region(row..<min(maped.shape[0], row + max(1, 4096 / maped.shape[1])))
      }
    }
    seconds.append(Date.timeIntervalSinceReferenceDate - start)
  }
  document["processing_pass_seconds"] = seconds
}
// Diagnostic only: measure the existing count codec on the exact byte planes
// of float32 regions. This is not a public floating-point resident API.
if ProcessInfo.processInfo.environment["MAPED_FLOAT_CACHE_PROBE"] == "1" {
  var records: [[String: Any]] = []
  for first in [0, 248, 504] {
    try autoreleasepool {
      let region = try maped.merged_region(first..<(first + 8))
      let cache = try MetalEncodedSource(
        shape: [4096, 1, maped.shape[2], maped.shape[3] * 4],
        itemBytes: 1, device: maped.operations.device)
      let began = Date.timeIntervalSinceReferenceDate
      try cache.append(region.buffer, frames: 4096)
      let encodedSeconds = Date.timeIntervalSinceReferenceDate - began
      let readStarted = Date.timeIntervalSinceReferenceDate
      let restored = try cache.read(0..<4096)
      let readSeconds = Date.timeIntervalSinceReferenceDate - readStarted
      let library = try maped.operations.device.makeLibrary(
        source: MetalCountResources.source("resident_utilities"), options: nil)
      let pipeline = try maped.operations.device.makeComputePipelineState(
        function: library.makeFunction(name: "count_verify")!)
      let error = maped.operations.device.makeBuffer(length: 4, options: .storageModeShared)!
      error.contents().storeBytes(of: UInt32(0), as: UInt32.self)
      let command = maped.operations.queue.makeCommandBuffer()!
      let encoder = command.makeComputeCommandEncoder()!
      encoder.setComputePipelineState(pipeline)
      encoder.setBuffer(region.buffer, offset: 0, index: 0)
      encoder.setBuffer(restored, offset: 0, index: 1)
      encoder.setBuffer(error, offset: 0, index: 2)
      var bytes = UInt32(region.rows * region.columns * 4)
      encoder.setBytes(&bytes, length: 4, index: 3)
      encoder.dispatchThreads(
        MTLSize(width: Int(bytes), height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
      encoder.endEncoding()
      command.commit()
      command.waitUntilCompleted()
      guard command.status == .completed, error.contents().load(as: UInt32.self) == 0
      else { fatalError("Float byte cache round-trip failed.") }
      records.append([
        "first_row": first, "encoded_bytes": cache.residentBytes,
        "raw_bytes": Int(bytes), "encode_seconds": encodedSeconds,
        "read_seconds": readSeconds, "exact": true,
      ])
      cache.releaseResidentStorage()
    }
  }
  document["float_byte_cache_probe"] = records
}
let validationStarted = Date.timeIntervalSinceReferenceDate
if ProcessInfo.processInfo.environment["MAPED_VALIDATE_WEIGHTS"] == "1" {
  let operations = maped.operations
  let scan = try operations.interiorWindow(rows: 512, columns: 512)
  let detector = try operations.image(rows: 192, columns: 192, value: 1)
  let ramp = try operations.image(
    values: (0..<(192 * 192)).map(Float.init), rows: 192, columns: 192)
  for index in 0..<7 {
    let weights = [
      (
        "scan_weight",
        try operations.shiftedScanMask(scan, shifts: maped.real_space_shifts!, index: index)
      ),
      (
        "detector_weight",
        try operations.shifted(detector, shifts: maped.diffraction_shifts!, index: index)
      ),
      (
        "detector_ramp",
        try operations.shifted(ramp, shifts: maped.diffraction_shifts!, index: index)
      ),
    ]
    for (name, weight) in weights {
      try weight.values().withUnsafeBytes {
        try Data($0).write(
          to: report.deletingPathExtension().appendingPathExtension("\(name)-\(index).f32"))
      }
    }
  }
}
for (name, images) in [("dp_mean", maped.dp_mean), ("im_bf", maped.im_bf)] {
  for (index, image) in images.enumerated() {
    let values = image.values()
    try values.withUnsafeBytes {
      try Data($0).write(
        to: report.deletingPathExtension().appendingPathExtension("\(name)-\(index).f32"))
    }
  }
}
let mergeStarted = Date.timeIntervalSinceReferenceDate
try autoreleasepool {
  let region = try maped.merged_region(248..<256)
  try region.values().withUnsafeBytes {
    try Data($0).write(to: report.deletingPathExtension().appendingPathExtension("region.f32"))
  }
}
document["region_seconds"] = Date.timeIntervalSinceReferenceDate - mergeStarted
let validationSeconds = Date.timeIntervalSinceReferenceDate - validationStarted
document["validation_export_seconds"] = validationSeconds
if arguments.count > 3 {
  let result = try maped.merge_datasets(save_to: URL(fileURLWithPath: arguments[3]))
  document["timings"] = maped.timings
  document["output_resident_bytes"] = result.residentBytes
  document["precision"] = result.metadata
  document["total_wall_seconds"] = Date.timeIntervalSinceReferenceDate - started - validationSeconds
}
document["peak_metal_bytes"] = maped.peak_metal_bytes
try JSONSerialization.data(withJSONObject: document, options: [.prettyPrinted, .sortedKeys]).write(
  to: report)
print(
  String(
    data: try JSONSerialization.data(withJSONObject: document, options: [.sortedKeys]),
    encoding: .utf8)!)
maped.close()
