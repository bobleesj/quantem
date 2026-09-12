import Foundation
import MetalScientificNumerics
import QuantEMMAPED

// Native hardware test runner. The manifest selects scientific calls, not kernels.
let args = CommandLine.arguments
if args.count == 4 || args.count == 7, args[1] == "--summaries" {
  let prefix = args[2]
  let directory = URL(fileURLWithPath: args[3])
  try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
  let sigma: Double = args.count == 7 ? Double(args[4])! : 2
  let padding = args.count == 7 ? Int(args[5])! : 5
  let hann = args.count == 7 && args[6] == "true"
  try JSONSerialization.data(withJSONObject: ["sigma": sigma, "padding": padding, "hann": hann])
    .write(to: directory.appendingPathComponent("settings.json"))
  let operations = try MetalImageOperations()
  let shifts = try operations.image(rows: 7, columns: 2)
  let window = try operations.window(
    operations.image(rows: 512, columns: 512, value: 1), kind: hann ? 2 : 0, padding: padding)
  var spectra: [GPUImage] = []
  func export(_ image: GPUImage, _ name: String) throws {
    try image.values().withUnsafeBytes {
      try Data($0).write(to: directory.appendingPathComponent(name + ".f32"))
    }
  }
  try export(window, "window")
  for index in 0..<7 {
    let bytes = try Data(contentsOf: URL(fileURLWithPath: prefix + ".im_bf-\(index).f32"))
    let values = bytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    let image = try operations.image(values: values, rows: 512, columns: 512)
    let gradient = try operations.gradientMagnitude(image, sigma: sigma)
    let padded = try operations.window(gradient, padding: padding)
    let shifted = try operations.shifted(padded, shifts: shifts, index: index)
    let centered = try operations.centered(shifted, window: window)
    let spectrum = try operations.fourier(centered)
    for (name, item) in [
      ("gradient", gradient), ("shifted", shifted), ("centered", centered), ("spectrum", spectrum),
    ] {
      try export(item, "\(name)-\(index)")
    }
    spectra.append(spectrum)
  }
  let reference = try operations.mean(spectra)
  try export(reference, "reference")
  for index in 1..<7 {
    try export(operations.correlation(reference, spectra[index]), "shift-\(index)")
  }
  var diffractionReference: GPUImage?
  for index in 0..<7 {
    let bytes = try Data(contentsOf: URL(fileURLWithPath: prefix + ".dp_mean-\(index).f32"))
    let values = bytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    let dp = try operations.image(values: values, rows: 192, columns: 192)
    let weighted = try operations.window(dp, kind: 1, edge_blend: 16)
    let spectrum = try operations.fourier(weighted)
    try export(weighted, "dwindow-\(index)")
    try export(spectrum, "dspectrum-\(index)")
    if let reference = diffractionReference {
      let shift = try operations.correlation(reference, spectrum, upsample_factor: 200)
      try export(shift, "dshift-\(index)")
      diffractionReference = try operations.blendSpectrum(
        reference, spectrum, count: index, shift: shift)
    } else {
      diffractionReference = spectrum
    }
    try export(diffractionReference!, "dreference-\(index)")
  }
  exit(0)
}
guard args.count == 4 else {
  fatalError("Usage: maped-parameter-parity INPUT_DIRECTORY CASES_JSON OUTPUT_DIRECTORY")
}
let input = URL(fileURLWithPath: args[1])
let manifestURL = URL(fileURLWithPath: args[2])
let output = URL(fileURLWithPath: args[3])
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
let manifest =
  try JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as! [String: Any]
let cases = manifest["cases"] as! [[String: Any]]
let positions = manifest["sample_positions"] as! [[Int]]
let files = try FileManager.default.contentsOfDirectory(at: input, includingPropertiesForKeys: nil)
  .filter { $0.lastPathComponent.hasSuffix("_master.h5") }.sorted { $0.path < $1.path }
guard files.count == 7 else { fatalError("Choose the directory containing exactly seven masters.") }
let maped = try MAPEDNative.from_files(files)
defer { maped.close() }
let sourceBytes = maped.sources.map(\.residentBytes)
var results: [[String: Any]] = []
func integer(_ options: [String: Any], _ name: String, _ fallback: Int) -> Int {
  (options[name] as? NSNumber)?.intValue ?? fallback
}
func scalar(_ options: [String: Any], _ name: String, _ fallback: Double) -> Double {
  (options[name] as? NSNumber)?.doubleValue ?? fallback
}
func padValue(_ value: Any?, fallback: String) -> MAPEDPadValue {
  if let number = value as? NSNumber { return .value(number.doubleValue) }
  return .statistic(value as? String ?? fallback)
}
for test in cases {
  let name = test["id"] as! String
  let started = Date.timeIntervalSinceReferenceDate
  var record: [String: Any] = ["id": name]
  do {
    try autoreleasepool {
      let preprocessing = test["preprocess"] as? [String: Any] ?? [:]
      if let values = preprocessing["scale"] as? [NSNumber] {
        try maped.preprocess(scale: values.map(\.floatValue))
      } else if let value = preprocessing["scale"] as? NSNumber {
        try maped.preprocess(scale: value.floatValue)
      } else {
        try maped.preprocess()
      }
      let origin = test["diffraction_origin"] as? [String: Any] ?? [:]
      let sigma = (origin["sigma"] as? NSNumber)?.doubleValue
      if let pairs = origin["origins"] as? [[Int]] {
        try maped.diffraction_origin(origins: pairs, sigma: sigma)
      } else if let pair = origin["origins"] as? [Int] {
        try maped.diffraction_origin(origins: (pair[0], pair[1]), sigma: sigma)
      } else if origin.isEmpty {
        try maped.diffraction_origin()
      } else {
        try maped.diffraction_origin(sigma: sigma)
      }
      let diffraction = test["diffraction_align"] as? [String: Any] ?? [:]
      if diffraction.isEmpty {
        try maped.diffraction_align()
      } else {
        try maped.diffraction_align(
          edge_blend: scalar(diffraction, "edge_blend", 16),
          padding: (diffraction["padding"] as? NSNumber)?.intValue,
          pad_val: padValue(diffraction["pad_val"], fallback: "min"),
          upsample_factor: integer(diffraction, "upsample_factor", 100),
          weight_scale: scalar(diffraction, "weight_scale", 0.125))
      }
      let real = test["real_space_align"] as? [String: Any] ?? [:]
      if real.isEmpty {
        try maped.real_space_align()
      } else {
        try maped.real_space_align(
          num_images: (real["num_images"] as? NSNumber)?.intValue,
          num_iter: integer(real, "num_iter", 3), edge_blend: scalar(real, "edge_blend", 1),
          padding: (real["padding"] as? NSNumber)?.intValue,
          pad_val: padValue(real["pad_val"], fallback: "median"),
          upsample_factor: integer(real, "upsample_factor", 100),
          max_shift: (real["max_shift"] as? NSNumber)?.doubleValue,
          shift_method: real["shift_method"] as? String ?? "bilinear",
          edge_filter: real["edge_filter"] as? Bool ?? true,
          edge_sigma: scalar(real, "edge_sigma", 2),
          hanning_filter: real["hanning_filter"] as? Bool ?? false)
      }
      record["origins"] = maped.diffraction_origins
      record["diffraction_shifts"] = maped.diffraction_shifts!.values()
      record["real_space_shifts"] = maped.real_space_shifts!.values()
      record["parameters"] = maped.parameters
      var patterns = Data()
      let pixels = maped.shape[2] * maped.shape[3]
      for row in Set(positions.map { $0[0] }).sorted() {
        try autoreleasepool {
          let region = try maped.merged_region(row..<(row + 1))
          for position in positions where position[0] == row {
            patterns.append(
              Data(
                bytes: region.buffer.contents().advanced(by: position[1] * pixels * 4),
                count: pixels * 4))
          }
        }
      }
      try patterns.write(to: output.appendingPathComponent(name + ".f32"))
      record["source_bytes_unchanged"] = maped.sources.map(\.residentBytes) == sourceBytes
      record["status"] = "passed"
    }
  } catch {
    record["status"] = "error"
    record["error"] = String(describing: error)
  }
  record["seconds"] = Date.timeIntervalSinceReferenceDate - started
  results.append(record)
  let report: [String: Any] = [
    "shape": maped.shape, "cases": results, "peak_metal_bytes": maped.peak_metal_bytes,
    "source_read_passes": maped.sources.map(\.sourceReadPasses),
  ]
  try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    .write(to: output.appendingPathComponent("native.json"))
  print("\(name): \(record["status"]!)")
  fflush(stdout)
}

var rejected: [[String: Any]] = []
for test in manifest["rejected_merge_cases"] as? [[String: Any]] ?? [] {
  let name = test["id"] as! String
  let options = test["options"] as! [String: Any]
  do {
    _ = try maped.merge_datasets(
      real_space_padding: integer(options, "real_space_padding", 0),
      real_space_edge_blend: scalar(options, "real_space_edge_blend", 1),
      diffraction_padding: integer(options, "diffraction_padding", 0),
      diffraction_edge_blend: scalar(options, "diffraction_edge_blend", 0),
      shift_method: options["shift_method"] as? String ?? "bilinear",
      dtype: options["dtype"] as? String,
      save_to: output.appendingPathComponent(name + "_master.h5"),
      scale_output: options["scale_output"] as? Bool ?? false, verbose: false)
    rejected.append(["id": name, "rejected": false])
  } catch {
    let message = String(describing: error)
    rejected.append([
      "id": name, "rejected": message.contains("Native resident merging supports"),
      "error": message,
    ])
  }
}
try JSONSerialization.data(withJSONObject: rejected, options: [.prettyPrinted, .sortedKeys])
  .write(to: output.appendingPathComponent("validation.json"))

if results.contains(where: { $0["status"] as? String != "passed" })
  || rejected.contains(where: { $0["rejected"] as? Bool != true })
{
  exit(1)
}
