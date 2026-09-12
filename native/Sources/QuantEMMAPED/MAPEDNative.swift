import Foundation
import Metal4DSTEMStreamingIO
import MetalScientificNumerics

/// The native MAPED scientific sequence, using QuantEM.GPU infrastructure.
/// Calls are serialized by the owner. Images and shifts remain GPU-resident.
public final class MAPEDNative {
  public let operations: MetalImageOperations
  public private(set) var sources: [any MetalResidentCounts]
  public let shape: [Int]
  public private(set) var dp_mean: [GPUImage] = []
  public private(set) var im_bf: [GPUImage] = []
  public private(set) var diffraction_origins: [[Int]] = []
  public private(set) var diffraction_shifts: GPUImage?
  public private(set) var real_space_shifts: GPUImage?
  public private(set) var peak_metal_bytes = 0
  public private(set) var timings: [String: Double] = [:]
  public private(set) var merged: MetalPackedSource?
  /// Effective scientific settings, grouped by the existing stage names.
  public private(set) var parameters: [String: [String: Any]] = [:]
  public private(set) var scales: [Float] = []
  private struct MergeWeights {
    let scan: [Float]
    let detector: [Float]
    let scanWeights: [GPUImage]
    let detectorWeights: [GPUImage]
    let uncovered: GPUImage
  }
  private var mergeWeights: MergeWeights?
  private let ownsSources: Bool

  private init(
    sources: [any MetalResidentCounts], operations: MetalImageOperations, ownsSources: Bool
  )
    throws
  {
    guard let first = sources.first,
      sources.allSatisfy({
        $0.shape == first.shape && !$0.isReleased && $0.readyFrames == $0.shape[0] * $0.shape[1]
          && $0.device.registryID == operations.device.registryID
      })
    else {
      throw Self.invalid("Load at least one acquisition; all tilts must share a 4D shape.")
    }
    guard first.shape[1] <= 4096 else {
      throw Self.invalid(
        "Native row-aligned merging supports at most 4096 scan columns; select a narrower acquisition."
      )
    }
    self.sources = sources
    self.operations = operations
    shape = first.shape
    self.ownsSources = ownsSources
    recordPeak()
  }
  /// Load every tilt once into encoded residency, with median hot-pixel correction.
  public static func from_files(_ files: [URL], index_directory: URL? = nil) throws -> MAPEDNative {
    let started = Date.timeIntervalSinceReferenceDate
    let operations = try MetalImageOperations()
    let root =
      index_directory
      ?? FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("QuantEM/Index", isDirectory: true)
    let sources = try operations.loadEncoded(files: files, indexDirectory: root)
    let result = try MAPEDNative(sources: sources, operations: operations, ownsSources: true)
    result.timings["load"] = Date.timeIntervalSinceReferenceDate - started
    return result
  }
  /// Borrow encoded or bit-packed count acquisitions without changing their lifetime.
  public static func from_resident(
    _ sources: [any MetalResidentCounts], operations: MetalImageOperations
  ) throws -> MAPEDNative {
    try MAPEDNative(
      sources: sources.map { try $0.correctedHotPixels() }, operations: operations,
      ownsSources: false)
  }
  /// Reuse complete-detector and complete-scan means calculated during loading.
  @discardableResult public func preprocess(scale: Float) throws -> MAPEDNative {
    try preprocess(scale: Array(repeating: scale, count: sources.count))
  }
  @discardableResult public func preprocess(scale: [Float]? = nil) throws -> MAPEDNative {
    guard !sources.isEmpty, sources.allSatisfy({ !$0.isReleased }) else {
      throw Self.invalid(
        "Load resident inputs before preprocessing; owned inputs are released after saving.")
    }
    if let scale, scale.count != sources.count || scale.contains(where: { !$0.isFinite || $0 == 0 })
    {
      throw Self.invalid("scale needs a finite nonzero value or one such value per tilt.")
    }
    // As in MAPEDTorch, scale is a summary-display control; it does not change counts.
    let means = try sources.map { try $0.countMeans() }
    dp_mean = means.map { GPUImage(buffer: $0.diffraction, rows: shape[2], columns: shape[3]) }
    im_bf = means.map { GPUImage(buffer: $0.brightField, rows: shape[0], columns: shape[1]) }
    scales = scale ?? Array(repeating: 1, count: sources.count)
    parameters["preprocess"] = ["scale": scales]
    return self
  }
  @discardableResult public func diffraction_origin(origins: (Int, Int), sigma: Double? = nil)
    throws -> MAPEDNative
  {
    try diffraction_origin(
      origins: Array(repeating: [origins.0, origins.1], count: sources.count), sigma: sigma)
  }
  @discardableResult public func diffraction_origin(origins: [[Int]]? = nil, sigma: Double? = nil)
    throws -> MAPEDNative
  {
    guard !dp_mean.isEmpty else {
      throw Self.invalid("Run preprocess() before diffraction_origin().")
    }
    if let sigma {
      guard sigma.isFinite, sigma <= 0 || 2 * sigma < Double(min(shape[2], shape[3])) else {
        throw Self.invalid(
          "sigma must be finite with reflection padding smaller than the detector; got \(sigma).")
      }
    }
    if let origins {
      guard origins.count == sources.count, origins.allSatisfy({ $0.count == 2 }) else {
        throw Self.invalid("origins needs one (row, column) pair per tilt.")
      }
      diffraction_origins = origins
    } else {
      diffraction_origins = try dp_mean.map {
        try operations.origin(operations.gaussian($0, sigma: sigma ?? 0))
      }
    }
    parameters["diffraction_origin"] = [
      "origins": origins as Any? ?? NSNull(), "sigma": sigma as Any? ?? NSNull(),
    ]
    return self
  }
  /// Align weighted mean diffraction patterns and center the resulting shifts.
  /// `padding` and `pad_val` are presentation-only in the reference implementation.
  @discardableResult public func diffraction_align(
    edge_blend: Double = 16, padding: Int? = nil, pad_val: MAPEDPadValue = "min",
    upsample_factor: Int = 100, weight_scale: Double = 0.125
  ) throws -> MAPEDNative {
    guard diffraction_origins.count == sources.count else {
      throw Self.invalid("Run diffraction_origin() before diffraction_align().")
    }
    guard !sources.isEmpty, edge_blend.isFinite, weight_scale.isFinite else {
      throw Self.invalid("Use loaded inputs and finite diffraction edge_blend and weight_scale.")
    }
    let started = Date.timeIntervalSinceReferenceDate
    let shifts = try operations.image(rows: sources.count, columns: 2)
    var reference = try operations.fourier(
      operations.window(dp_mean[0], kind: 1, edge_blend: edge_blend))
    for index in 1..<sources.count {
      try autoreleasepool {
        let spectrum = try operations.fourier(
          operations.window(dp_mean[index], kind: 1, edge_blend: edge_blend))
        let shift = try operations.correlation(
          reference, spectrum, upsample_factor: upsample_factor)
        try operations.addShift(shifts, shift, index: index)
        reference = try operations.blendSpectrum(reference, spectrum, count: index, shift: shift)
      }
    }
    try operations.centerShifts(shifts)
    diffraction_shifts = shifts
    mergeWeights = nil
    merged?.releaseResidentStorage()
    merged = nil
    parameters["diffraction_align"] = [
      "edge_blend": edge_blend, "padding": padding as Any? ?? NSNull(),
      "pad_val": pad_val.metadata, "upsample_factor": upsample_factor, "weight_scale": weight_scale,
    ]
    timings["diffraction_align"] = Date.timeIntervalSinceReferenceDate - started
    recordPeak()
    return self
  }
  /// Iteratively align mean bright-field images against their current average.
  @discardableResult public func real_space_align(
    num_images: Int? = nil, num_iter: Int = 3, edge_blend: Double = 1,
    padding: Int? = nil, pad_val: MAPEDPadValue = "median", upsample_factor: Int = 100,
    max_shift: Double? = nil, shift_method: String = "bilinear", edge_filter: Bool = true,
    edge_sigma: Double = 2, hanning_filter: Bool = false
  ) throws -> MAPEDNative {
    guard !im_bf.isEmpty else { throw Self.invalid("Run preprocess() before real_space_align().") }
    let n = min(num_images ?? sources.count, sources.count)
    guard n > 0, num_iter > 0, !edge_filter || (edge_sigma.isFinite && edge_sigma > 0),
      edge_blend.isFinite, max_shift?.isFinite != false
    else {
      throw Self.invalid(
        "Use positive num_images and num_iter, finite padding controls, and positive edge_sigma when edge_filter is enabled."
      )
    }
    let paddingValue = ceil(max_shift ?? edge_blend) + 4
    let paddedRows = Double(shape[0]) + 2 * paddingValue
    let paddedColumns = Double(shape[1]) + 2 * paddingValue
    guard paddingValue >= 0,
      paddedRows * paddedColumns * 8 <= Double(operations.device.maxBufferLength)
    else {
      throw Self.invalid(
        "Correlation padding exceeds the device limit; reduce max_shift or edge_blend.")
    }
    let started = Date.timeIntervalSinceReferenceDate
    let pad = Int(paddingValue)
    let ones = try operations.image(rows: shape[0], columns: shape[1], value: 1)
    let window = try operations.window(ones, kind: hanning_filter ? 2 : 0, padding: pad)
    let base = try im_bf.prefix(n).map { image in
      try operations.window(
        edge_filter ? operations.gradientMagnitude(image, sigma: edge_sigma) : image, padding: pad)
    }
    let shifts = try operations.image(rows: sources.count, columns: 2)
    for _ in 0..<num_iter {
      try autoreleasepool {
        let spectra = try base.enumerated().map { index, image in
          try operations.fourier(
            operations.centered(
              operations.shifted(image, shifts: shifts, index: index), window: window))
        }
        let reference = try operations.mean(spectra)
        for index in 1..<n {
          let shift = try operations.correlation(
            reference, spectra[index], upsample_factor: upsample_factor)
          try operations.addShift(shifts, shift, index: index)
        }
        recordPeak()
      }
    }
    try operations.centerShifts(shifts, count: n)
    real_space_shifts = shifts
    mergeWeights = nil
    merged?.releaseResidentStorage()
    merged = nil
    parameters["real_space_align"] = [
      "num_images": num_images as Any? ?? NSNull(), "num_iter": num_iter,
      "edge_blend": edge_blend, "padding": padding as Any? ?? NSNull(),
      "pad_val": pad_val.metadata, "upsample_factor": upsample_factor,
      "max_shift": max_shift as Any? ?? NSNull(), "shift_method": shift_method,
      "edge_filter": edge_filter, "edge_sigma": edge_sigma, "hanning_filter": hanning_filter,
    ]
    timings["real_space_align"] = Date.timeIntervalSinceReferenceDate - started
    recordPeak()
    return self
  }
  /// Produce one float32 region using the established resident merge weights.
  /// This is the numerical sampling boundary for independent native parity tests.
  public func merged_region(_ rows: Range<Int>) throws -> GPUImage {
    try mergedRegion(rows, workspace: nil)
  }
  private func mergedRegion(_ rows: Range<Int>, workspace: (GPUImage, GPUImage)?) throws -> GPUImage
  {
    guard !sources.isEmpty, sources.allSatisfy({ !$0.isReleased }) else {
      throw Self.invalid(
        "Resident inputs are no longer available; read the saved result or load inputs again.")
    }
    guard let scan = real_space_shifts, let detector = diffraction_shifts else {
      throw Self.invalid("Run diffraction_align() and real_space_align() before merging.")
    }
    guard !rows.isEmpty, rows.lowerBound >= 0, rows.upperBound <= shape[0],
      rows.count * shape[1] <= 4096
    else {
      throw Self.invalid(
        "Select nonempty scan rows containing at most 4096 frames inside the acquisition.")
    }
    // Small shift arrays are control metadata; cache the GPU weights until
    // either alignment changes, including changes to an exposed shift buffer.
    let scanValues = scan.values()
    let detectorValues = detector.values()
    if mergeWeights?.scan != scanValues || mergeWeights?.detector != detectorValues {
      let mask = try operations.interiorWindow(rows: shape[0], columns: shape[1])
      let ones = try operations.image(rows: shape[2], columns: shape[3], value: 1)
      let scanWeights = try sources.indices.map {
        try operations.shiftedScanMask(mask, shifts: scan, index: $0)
      }
      let detectorWeights = try sources.indices.map {
        try operations.shifted(ones, shifts: detector, index: $0)
      }
      mergeWeights = MergeWeights(
        scan: scanValues, detector: detectorValues, scanWeights: scanWeights,
        detectorWeights: detectorWeights, uncovered: try operations.uncoveredWeight(detectorWeights)
      )
    }
    let weights = mergeWeights!
    let numerator: GPUImage
    let denominator: GPUImage
    if let workspace {
      // Views expose only the active tail region; scientific reductions never
      // include stale values outside these row/column dimensions.
      numerator = GPUImage(
        buffer: workspace.0.buffer, rows: rows.count * shape[1], columns: shape[2] * shape[3])
      denominator = GPUImage(
        buffer: workspace.1.buffer, rows: rows.count * shape[1], columns: shape[2] * shape[3])
      try operations.fill(numerator)
      try operations.fill(denominator)
    } else {
      numerator = try operations.image(rows: rows.count * shape[1], columns: shape[2] * shape[3])
      denominator = try operations.image(rows: rows.count * shape[1], columns: shape[2] * shape[3])
    }
    for index in sources.indices {
      try operations.accumulateTranslated(
        source: sources[index], outputRows: rows,
        scanShifts: scan, detectorShifts: detector, index: index,
        scanWeight: weights.scanWeights[index], detectorWeight: weights.detectorWeights[index],
        numerator: numerator, denominator: denominator)
      recordPeak()
    }
    return try operations.finishWeighted(
      numerator, denominator: denominator, uncovered: weights.uncovered)
  }
  /// Merge bounded float32 regions, save globally scaled uint16, then reopen
  /// the complete result in packed GPU memory. Scientific keyword names and
  /// the supported resident subset match MAPEDTorch.merge_datasets.
  @discardableResult public func merge_datasets(
    real_space_padding: Int = 0, real_space_edge_blend: Double = 1,
    diffraction_padding: Int = 0, diffraction_edge_blend: Double = 0,
    diffraction_pad_val: MAPEDPadValue = "min", shift_method: String = "bilinear",
    dtype: String? = nil, save_to: URL, scale_output: Bool = false,
    verbose: Bool = true
  ) throws -> MetalPackedSource {
    guard real_space_padding == 0, real_space_edge_blend == 1,
      diffraction_padding == 0, diffraction_edge_blend == 0,
      shift_method.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "bilinear",
      dtype == nil || dtype == "scaled_uint16", !scale_output
    else {
      throw Self.invalid(
        "Native resident merging supports bilinear shifts, zero padding, scan edge blend 1, detector edge blend 0, and scaled_uint16 storage."
      )
    }
    guard !sources.isEmpty, sources.allSatisfy({ !$0.isReleased }) else {
      throw Self.invalid("Load resident inputs before merging another result.")
    }
    let precision = try MetalPrecision(device: operations.device)
    let writer = try MetalHDF5Writer(path: save_to, shape: shape, runtime: precision)
    merged?.releaseResidentStorage()
    merged = nil
    // Fixed 4096-frame ceiling matches the established resident workflow.
    let rowsPerRegion = max(1, 4096 / shape[1])
    let regions = stride(from: 0, to: shape[0], by: rowsPerRegion).map {
      $0..<min(shape[0], $0 + rowsPerRegion)
    }
    var workspace: (GPUImage, GPUImage)? = (
      try operations.image(
        rows: min(shape[0], rowsPerRegion) * shape[1], columns: shape[2] * shape[3]),
      try operations.image(
        rows: min(shape[0], rowsPerRegion) * shape[1], columns: shape[2] * shape[3])
    )
    var started = Date.timeIntervalSinceReferenceDate
    for rows in regions {
      try autoreleasepool {
        let values = try mergedRegion(rows, workspace: workspace)
        try precision.includeRange(values.buffer, count: values.rows * values.columns)
        recordPeak()
      }
    }
    try precision.calibrate(shape: shape)
    timings["merge_range"] = Date.timeIntervalSinceReferenceDate - started
    started = Date.timeIntervalSinceReferenceDate
    var generation = 0.0
    var conversion = 0.0
    for rows in regions {
      try autoreleasepool {
        var phase = Date.timeIntervalSinceReferenceDate
        let values = try mergedRegion(rows, workspace: workspace)
        generation += Date.timeIntervalSinceReferenceDate - phase
        phase = Date.timeIntervalSinceReferenceDate
        let codes = try precision.convert(values.buffer, count: values.rows * values.columns)
        conversion += Date.timeIntervalSinceReferenceDate - phase
        try writer.append(codes, frames: rows.count * shape[1])
        recordPeak()
        peak_metal_bytes = max(peak_metal_bytes, writer.peakAllocatedBytes)
      }
    }
    let report = try precision.finish()
    parameters["merge_datasets"] = [
      "real_space_padding": real_space_padding, "real_space_edge_blend": real_space_edge_blend,
      "diffraction_padding": diffraction_padding, "diffraction_edge_blend": diffraction_edge_blend,
      "diffraction_pad_val": diffraction_pad_val.metadata, "shift_method": shift_method,
      "dtype": dtype as Any? ?? NSNull(), "scale_output": scale_output,
    ]
    func json(_ value: Any) throws -> String {
      String(
        data: try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]),
        encoding: .utf8)!
    }
    let summary: [String: Any] = [
      "version": 1, "source_count": sources.count,
      "hot_pixel_correction": [
        "methods": sources.map { $0.hotPixelCorrection },
        "applied_to_every_source": sources.allSatisfy { $0.hotPixelCorrection == "median" },
        "pixel_counts": sources.map { $0.hotPixelIndices.count },
      ],
      "mean_bright_field": [
        "operation": "arithmetic_mean", "reduction_axes": ["detector_row", "detector_column"],
        "divisor": shape[2] * shape[3], "output_shape": Array(shape[0..<2]),
        "detector_selection": "complete_detector", "alignment_role": "real_space",
        "invalid_pixel_policy": "stored detector-mask pixels use their local 3x3 median",
      ],
      "mean_diffraction_pattern": [
        "operation": "arithmetic_mean", "reduction_axes": ["scan_row", "scan_column"],
        "divisor": shape[0] * shape[1], "output_shape": Array(shape[2..<4]),
        "invalid_pixel_policy": "stored detector-mask pixels use their local 3x3 median",
      ],
      "intensity_normalization": "none",
    ]
    func pairs(_ values: GPUImage?) -> [[Float]] {
      let flat = values!.values()
      return stride(from: 0, to: flat.count, by: 2).map { [flat[$0], flat[$0 + 1]] }
    }
    let mergeMetadata: [String: Any] = [
      "version": 1, "backend": "metal",
      "source_representation": Set(sources.map { $0.representation.rawValue }).count == 1
        ? sources[0].representation.rawValue : "mixed",
      "source_representations": sources.map { $0.representation.rawValue },
      "region_frames": rowsPerRegion * shape[1],
      "released_sources_before_reopen": ownsSources,
      "real_space_shifts_row_column": pairs(real_space_shifts),
      "diffraction_shifts_row_column": pairs(diffraction_shifts),
      "parameters": parameters,
    ]
    try writer.finish(metadata: [
      "quantem_precision_v1": json(report), "quantem_maped_summary_v1": json(summary),
      "quantem_maped_merge_v1": json(mergeMetadata),
    ])
    workspace = nil
    timings["merge_write"] = Date.timeIntervalSinceReferenceDate - started
    timings["merge_generation_write_pass"] = generation
    timings["precision_conversion"] = conversion
    timings["hdf5_compression"] = writer.compressionSeconds
    timings["hdf5_write"] = writer.writeSeconds
    if ownsSources {
      for source in sources { source.releaseResidentStorage() }
      sources.removeAll()
    }
    started = Date.timeIntervalSinceReferenceDate
    let index = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("QuantEM/Index")
    let result = try MetalPackedSource.load(
      path: save_to, device: operations.device, indexDirectory: index)
    merged = result
    recordPeak()
    peak_metal_bytes = max(peak_metal_bytes, result.peakAllocatedBytes)
    timings["reopen_packed"] = Date.timeIntervalSinceReferenceDate - started
    if verbose {
      print(
        String(
          format: "Merged %d×%d scan | scaled_uint16 | %.3f GiB on GPU | RMSE %.7g", shape[0],
          shape[1], Double(result.residentBytes) / pow(2, 30), report["rmse"] as! Double))
    }
    return result
  }
  public func close() {
    if ownsSources { for source in sources { source.releaseResidentStorage() } }
    sources.removeAll()
    mergeWeights = nil
    dp_mean.removeAll()
    im_bf.removeAll()
    diffraction_shifts = nil
    real_space_shifts = nil
    merged?.releaseResidentStorage()
    merged = nil
  }
  private func recordPeak() {
    peak_metal_bytes = max(
      peak_metal_bytes, operations.allocatedBytes(),
      sources.compactMap { ($0 as? MetalEncodedSource)?.peakAllocatedBytes }.max() ?? 0)
  }
  private static func invalid(_ message: String) -> Metal4DSTEMStreamingIOError {
    .invalidRequest(message)
  }
}
