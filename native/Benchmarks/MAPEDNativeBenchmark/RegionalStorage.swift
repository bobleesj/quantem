import Foundation
import Metal
import Metal4DSTEMStreamingIO
import MetalCountResources
import QuantEMMAPED

/// Experiment only: retain each independently calibrated scan region using
/// existing precision and packed-source operations. No new file format or API.
func benchmarkRegionalStorage(_ maped: MAPEDNative) throws -> [String: Any] {
  let device = maped.operations.device
  let shape = maped.shape
  let rowsPerRegion = max(1, 4096 / shape[1])
  let auditPath = ProcessInfo.processInfo.environment["MAPED_REGIONAL_REFERENCE_REPORT"]
  var reference: MetalPrecision?
  var verification: MTLComputePipelineState?
  if let auditPath {
    let document =
      try JSONSerialization.jsonObject(
        with: Data(
          contentsOf:
            URL(fileURLWithPath: auditPath))) as! [String: Any]
    reference = try MetalPrecision(device: device)
    try reference!.useSavedReport(document["precision"] as! [String: Any])
    let library = try device.makeLibrary(
      source: MetalCountResources.source("resident_utilities"), options: nil)
    verification = try device.makeComputePipelineState(
      function: library.makeFunction(name: "count_verify")!)
  }
  var residents: [MetalPackedSource] = []
  defer { for resident in residents { resident.releaseResidentStorage() } }
  var reports: [[String: Any]] = []
  var mergeSeconds = 0.0
  var conversionSeconds = 0.0
  var packingSeconds = 0.0
  var auditSeconds = 0.0
  var peak = device.currentAllocatedSize
  let started = Date.timeIntervalSinceReferenceDate
  for first in stride(from: 0, to: shape[0], by: rowsPerRegion) {
    try autoreleasepool {
      let stop = min(shape[0], first + rowsPerRegion)
      var phase = Date.timeIntervalSinceReferenceDate
      let values = try maped.merged_region(first..<stop)
      mergeSeconds += Date.timeIntervalSinceReferenceDate - phase
      phase = Date.timeIntervalSinceReferenceDate
      let precision = try MetalPrecision(device: device)
      let localShape = [stop - first, shape[1], shape[2], shape[3]]
      let count = localShape.reduce(1, *)
      try precision.includeRange(values.buffer, count: count)
      try precision.calibrate(shape: localShape)
      let codes = try precision.convert(values.buffer, count: count)
      var report = try precision.finish()
      report["first_scan_row"] = first
      report["stop_scan_row"] = stop
      reports.append(report)
      conversionSeconds += Date.timeIntervalSinceReferenceDate - phase
      phase = Date.timeIntervalSinceReferenceDate
      let packed = try MetalPackedSource(shape: localShape, precision: precision)
      try packed.append(codes, frames: localShape[0] * localShape[1])
      residents.append(packed)
      packingSeconds += Date.timeIntervalSinceReferenceDate - phase
      peak = max(peak, maped.peak_metal_bytes, device.currentAllocatedSize)
      if let reference, let verification {
        phase = Date.timeIntervalSinceReferenceDate
        let globalCodes = try reference.convert(values.buffer, count: count)
        let expected = try precision.restore(codes, count: count)
        let actual = try packed.read(0..<(localShape[0] * localShape[1]))
        if let directory = ProcessInfo.processInfo.environment["MAPED_REGIONAL_DP_DIRECTORY"],
          shape[0] == 512, shape[1] == 512, [0, 248, 504].contains(first) {
          let folder = URL(fileURLWithPath: directory)
          try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
          let frame = 4 * shape[1] + 256
          let pixels = shape[2] * shape[3]
          let selectedCodes = device.makeBuffer(length: pixels * 2, options: .storageModeShared)!
          let copy = maped.operations.queue.makeCommandBuffer()!
          let blit = copy.makeBlitCommandEncoder()!
          blit.copy(from: globalCodes, sourceOffset: frame * pixels * 2,
            to: selectedCodes, destinationOffset: 0, size: pixels * 2)
          blit.endEncoding()
          copy.commit()
          copy.waitUntilCompleted()
          guard copy.status == .completed else { fatalError("DP export copy failed.") }
          let globalDP = try reference.restore(selectedCodes, count: pixels)
          let prefix = "row-\(first + 4)-col-256"
          for (name, buffer, offset) in [
            ("float32", values.buffer, frame * pixels * 4),
            ("global", globalDP, 0), ("regional", actual, frame * pixels * 4)] {
            try Data(bytes: buffer.contents().advanced(by: offset), count: pixels * 4)
              .write(to: folder.appendingPathComponent("\(prefix)-\(name).f32"))
          }
        }
        let error = device.makeBuffer(length: 4, options: .storageModeShared)!
        error.contents().storeBytes(of: UInt32(0), as: UInt32.self)
        let command = maped.operations.queue.makeCommandBuffer()!
        let encoder = command.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(verification)
        encoder.setBuffer(expected, offset: 0, index: 0)
        encoder.setBuffer(actual, offset: 0, index: 1)
        encoder.setBuffer(error, offset: 0, index: 2)
        var bytes = UInt32(count * 4)
        encoder.setBytes(&bytes, length: 4, index: 3)
        encoder.dispatchThreads(
          MTLSize(width: count * 4, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed, error.contents().load(as: UInt32.self) == 0
        else { fatalError("Regional packed intensity restoration is not exact.") }
        auditSeconds += Date.timeIntervalSinceReferenceDate - phase
      }
    }
  }
  let processingWall = Date.timeIntervalSinceReferenceDate - started
  let residentBytes = residents.reduce(0) { $0 + $1.residentBytes }
  // Only GPU-produced scalar reports are combined on the host, never arrays.
  let count = reports.reduce(0.0) { $0 + ($1["values"] as! NSNumber).doubleValue }
  let squaredError = reports.reduce(0.0) {
    $0 + pow($1["rmse"] as! Double, 2) * ($1["values"] as! NSNumber).doubleValue
  }
  let sourceReleased = Date.timeIntervalSinceReferenceDate
  maped.close()
  let releaseSeconds = Date.timeIntervalSinceReferenceDate - sourceReleased
  var readSeconds: [Double] = []
  for frame in [0, 4095, 4096, shape[0] * shape[1] / 2, shape[0] * shape[1] - 1] {
    let index = frame / (rowsPerRegion * shape[1])
    let local = frame - index * rowsPerRegion * shape[1]
    for _ in 0..<5 {
      try autoreleasepool {
        let phase = Date.timeIntervalSinceReferenceDate
        _ = try residents[index].read(local..<(local + 1))
        readSeconds.append(Date.timeIntervalSinceReferenceDate - phase)
      }
    }
  }
  var result: [String: Any] = [
    "merge_passes": 1, "region_count": residents.count, "region_frames": rowsPerRegion * shape[1],
    "processing_wall_seconds": processingWall, "audit_seconds": auditSeconds,
    "merge_seconds": mergeSeconds, "range_convert_report_seconds": conversionSeconds,
    "packing_seconds": packingSeconds, "source_release_seconds": releaseSeconds,
    "packed_resident_bytes": residentBytes, "peak_metal_bytes_without_audit": peak,
    "rmse": sqrt(squaredError / count),
    "max_abs_error": reports.map { $0["max_abs_error"] as! Double }.max()!,
    "positive_to_zero": reports.reduce(UInt64(0)) {
      $0 + ($1["positive_to_zero"] as! NSNumber).uint64Value
    },
    "overflow": reports.reduce(UInt64(0)) { $0 + ($1["overflow"] as! NSNumber).uint64Value },
    "values": UInt64(count), "selected_dp_seconds": readSeconds,
    "region_reports": reports, "saved_to_disk": false, "ui_rendering_measured": false,
    "packed_restore_audited": verification != nil,
  ]
  if let reference { result["global_reference_precision"] = try reference.finish() }
  return result
}
