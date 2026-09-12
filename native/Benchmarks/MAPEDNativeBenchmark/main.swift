import Foundation
import Metal4DSTEMStreamingIO
import MetalScientificNumerics
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
  "source_read_passes": maped.sources.map(\.sourceReadPasses),
  "diffraction_origins": maped.diffraction_origins,
  "diffraction_shifts": maped.diffraction_shifts!.values(),
  "real_space_shifts": maped.real_space_shifts!.values(),
]
let validationStarted = Date.timeIntervalSinceReferenceDate
if ProcessInfo.processInfo.environment["MAPED_VALIDATE_WEIGHTS"] == "1" {
  let operations = maped.operations
  let scan = try operations.interiorWindow(rows: 512, columns: 512)
  let detector = try operations.image(rows: 192, columns: 192, value: 1)
  let ramp = try operations.image(values: (0..<(192 * 192)).map(Float.init), rows: 192, columns: 192)
  for index in 0..<7 {
    let weights = [
      ("scan_weight", try operations.shiftedScanMask(scan, shifts: maped.real_space_shifts!, index: index)),
      ("detector_weight", try operations.shifted(detector, shifts: maped.diffraction_shifts!, index: index)),
      ("detector_ramp", try operations.shifted(ramp, shifts: maped.diffraction_shifts!, index: index))
    ]
    for (name, weight) in weights {
      try weight.values().withUnsafeBytes {
        try Data($0).write(to: report.deletingPathExtension().appendingPathExtension("\(name)-\(index).f32"))
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
