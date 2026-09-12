import CNativeHDF5
import Metal
import Metal4DSTEMStreamingIO
import MetalScientificNumerics
import Native4DSTEMIO
import QuantEMMAPED
import XCTest

final class MAPEDNativeTests: XCTestCase {
  func testMergeExportReusesWorkspaceWithPartialFinalRegion() throws {
    let ops = try MetalImageOperations()
    let shape = [65, 64, 64, 64]
    let source = try MetalEncodedSource(shape: shape, device: ops.device)
    defer { source.releaseResidentStorage() }
    for frames in [4096, 64] {
      let raw = ops.device.makeBuffer(length: frames * 4096 * 2, options: .storageModeShared)!
      let command = ops.queue.makeCommandBuffer()!
      let fill = command.makeBlitCommandEncoder()!
      fill.fill(buffer: raw, range: 0..<raw.length, value: 1)
      fill.endEncoding()
      command.commit()
      command.waitUntilCompleted()
      try source.append(raw, frames: frames, verify: true)
    }
    let maped = try MAPEDNative.from_resident([source], operations: ops)
    defer { maped.close() }
    try maped.preprocess()
    try maped.diffraction_origin(sigma: 1)
    try maped.diffraction_align(edge_blend: 2, upsample_factor: 1)
    try maped.real_space_align(num_iter: 1, upsample_factor: 1)
    let expected = try maped.merged_region(64..<65).values()
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let result = try maped.merge_datasets(
      save_to: folder.appendingPathComponent("tail_master.h5"), verbose: false)
    XCTAssertEqual(
      (result.metadata["values"] as? NSNumber)?.uint64Value, UInt64(shape.reduce(1, *)))
    let scale = result.metadata["scale"] as! Double
    let offset = result.metadata["offset"] as! Double
    let rounded = expected.map {
      Float(((Double($0) - offset) / scale).rounded(.toNearestOrEven) * scale + offset)
    }
    let restored = try result.read((64 * 64)..<(65 * 64))
    let values = Array(
      UnsafeBufferPointer(
        start: restored.contents().assumingMemoryBound(to: Float.self), count: expected.count))
    XCTAssertEqual(values.map(\.bitPattern), rounded.map(\.bitPattern))
    XCTAssertFalse(source.isReleased)
  }

  func testBorrowedMaskedUInt32CountsUseMedianWithoutChangingStorage() throws {
    let fixture = try numpyFixture()
    let ops = try MetalImageOperations()
    let raw = (fixture["raw"] as! [Int]).map(UInt32.init)
    let source = try MaskedCountFixture(
      shape: fixture["shape"] as! [Int],
      values: raw, bad: fixture["bad"] as! [Int], device: ops.device)
    let corrected = try source.correctedHotPixels()
    let read = try corrected.read(0..<source.readyFrames)
    XCTAssertEqual(
      Array(
        UnsafeBufferPointer(
          start: read.contents().assumingMemoryBound(to: UInt32.self), count: raw.count)),
      (fixture["corrected"] as! [Int]).map(UInt32.init))
    let maped = try MAPEDNative.from_resident([source], operations: ops)
    try maped.preprocess()
    XCTAssertEqual(maped.dp_mean[0].values(), (fixture["dp_mean"] as! [Double]).map(Float.init))
    XCTAssertEqual(maped.im_bf[0].values(), (fixture["im_bf"] as! [Double]).map(Float.init))
    maped.close()
    corrected.releaseResidentStorage()
    XCTAssertFalse(source.isReleased)
    let original = try source.read(0..<1)
    for pixel in source.hotPixelIndices {
      XCTAssertEqual(original.contents().load(fromByteOffset: pixel * 4, as: UInt32.self), 0)
    }
    let large = UInt32.max
    let overflow = try MaskedCountFixture(
      shape: [1, 1, 3, 3],
      values: [large - 1, large, large - 1, large, 0, large - 1, large, large - 1, large], bad: [4],
      device: ops.device)
    let median = try overflow.correctedHotPixels().read(0..<1)
    XCTAssertEqual(median.contents().load(fromByteOffset: 16, as: UInt32.self), large - 1)
    let empty = try MaskedCountFixture(
      shape: [1, 1, 2, 2], values: [9, 9, 9, 9], bad: [0, 1, 2, 3], device: ops.device)
    let zero = try empty.correctedHotPixels().read(0..<1)
    XCTAssertEqual(
      Array(
        UnsafeBufferPointer(start: zero.contents().assumingMemoryBound(to: UInt32.self), count: 4)),
      [0, 0, 0, 0])
  }

  func testPreparedSamplingMatchesReferenceAtBoundariesAndAfterShiftChanges() throws {
    let previous = ProcessInfo.processInfo.environment["QUANTEM_GPU_SAMPLING_REFERENCE"]
    defer {
      if let previous {
        setenv("QUANTEM_GPU_SAMPLING_REFERENCE", previous, 1)
      } else {
        unsetenv("QUANTEM_GPU_SAMPLING_REFERENCE")
      }
    }
    setenv("QUANTEM_GPU_SAMPLING_REFERENCE", "1", 1)
    let reference = try MetalImageOperations()
    unsetenv("QUANTEM_GPU_SAMPLING_REFERENCE")
    let candidate = try MetalImageOperations()
    for shape in [[4, 9, 11, 13], [2, 5, 3, 5], [8, 8, 64, 64]] {
      let frames = shape[0] * shape[1]
      let pixels = shape[2] * shape[3]
      for itemBytes in [1, 2] {
        let source = try MetalEncodedSource(
          shape: shape, itemBytes: itemBytes, device: candidate.device)
        defer { source.releaseResidentStorage() }
        let raw = candidate.device.makeBuffer(
          length: frames * pixels * itemBytes, options: .storageModeShared)!
        for i in 0..<(frames * pixels) {
          if itemBytes == 1 {
            raw.contents().storeBytes(of: UInt8((i * 37) % 256), toByteOffset: i, as: UInt8.self)
          } else {
            raw.contents().storeBytes(
              of: UInt16((i * 37) % 65536), toByteOffset: i * 2, as: UInt16.self)
          }
        }
        try source.append(raw, frames: frames, verify: true)
        let scanWeight = try candidate.image(
          values: (0..<frames).map { Float($0 % 13) / 13 }, rows: shape[0], columns: shape[1])
        let detectorWeight = try candidate.image(
          values: (0..<pixels).map { Float($0 % 17) / 17 }, rows: shape[2], columns: shape[3])
        for shifts: [Float] in [
          [0, 0, 0, 0], [0.3, -1.7, -0.42, 1.23], [-1, 1, 1, -1], [0, 0, 0, 0],
        ] {
          let scan = try candidate.image(values: Array(shifts[0..<2]), rows: 1, columns: 2)
          let detector = try candidate.image(values: Array(shifts[2..<4]), rows: 1, columns: 2)
          var outputs: [[UInt32]] = []
          for operations in [reference, candidate] {
            let numerator = try operations.image(rows: frames, columns: pixels)
            let denominator = try operations.image(rows: frames, columns: pixels)
            try operations.accumulateTranslated(
              source: source, outputRows: 0..<shape[0],
              scanShifts: scan, detectorShifts: detector, index: 0,
              scanWeight: scanWeight, detectorWeight: detectorWeight,
              numerator: numerator, denominator: denominator)
            outputs.append((numerator.values() + denominator.values()).map(\.bitPattern))
          }
          XCTAssertEqual(
            outputs[0], outputs[1], "shape=\(shape), bytes=\(itemBytes), shifts=\(shifts)")
        }
      }
    }
  }

  func testPackedAndEncodedResidentsPreserveCountsAlignmentAndOwnership() throws {
    let ops = try MetalImageOperations()
    let shape = [8, 8, 64, 64]
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    var encoded: [MetalEncodedSource] = []
    var packed: [MetalCompactH5ResidentSource] = []
    for tilt in 0..<3 {
      let values = (0..<(64 * 4096)).map {
        UInt16(($0 * 37 + ($0 / 4096) * 19 + tilt * 113) % 65536)
      }
      let raw = ops.device.makeBuffer(length: values.count * 2, options: .storageModeShared)!
      values.withUnsafeBytes { _ = memcpy(raw.contents(), $0.baseAddress!, $0.count) }
      let source = try MetalEncodedSource(shape: shape, device: ops.device)
      try source.append(raw, frames: 64, verify: true)
      encoded.append(source)
      let file = folder.appendingPathComponent("tilt-\(tilt)_master.h5")
      let writer = try MetalHDF5Writer(
        path: file, shape: shape, runtime: MetalPrecision(device: ops.device))
      try writer.append(raw, frames: 64)
      try writer.finish(metadata: [:])
      let catalog = try Native4DSTEMCatalogBuilder(
        cacheDirectory: folder.appendingPathComponent("index")
      ).prepare(input: file)
      let indexed = try Native4DSTEMIndexedSource.open(dataset: catalog.datasets[0])
      let resident = try MetalCompactH5Loader.load(source: indexed, device: ops.device)
      packed.append(resident)
      let decoded = try resident.read(7..<23)
      XCTAssertEqual(
        Array(
          UnsafeBufferPointer(
            start: decoded.contents().assumingMemoryBound(to: UInt32.self), count: 16 * 4096)),
        values[(7 * 4096)..<(23 * 4096)].map(UInt32.init))
    }
    defer {
      encoded.forEach { $0.releaseResidentStorage() }
      packed.forEach { $0.releaseResidentStorage() }
    }
    let reference = try MAPEDNative.from_resident(encoded, operations: ops)
    let candidate = try MAPEDNative.from_resident(packed, operations: ops)
    for workflow in [reference, candidate] {
      try workflow.preprocess()
      try workflow.diffraction_origin(sigma: 1)
      try workflow.diffraction_align(edge_blend: 2, upsample_factor: 3)
      try workflow.real_space_align(num_iter: 1, upsample_factor: 3)
    }
    XCTAssertEqual(reference.dp_mean.map { $0.values() }, candidate.dp_mean.map { $0.values() })
    XCTAssertEqual(reference.im_bf.map { $0.values() }, candidate.im_bf.map { $0.values() })
    XCTAssertEqual(reference.diffraction_origins, candidate.diffraction_origins)
    XCTAssertEqual(reference.diffraction_shifts!.values(), candidate.diffraction_shifts!.values())
    XCTAssertEqual(reference.real_space_shifts!.values(), candidate.real_space_shifts!.values())
    XCTAssertEqual(
      try reference.merged_region(0..<8).values(), try candidate.merged_region(0..<8).values())
    candidate.close()
    reference.close()
    XCTAssertTrue(packed.allSatisfy { !$0.isReleased })
    XCTAssertTrue(encoded.allSatisfy { !$0.isReleased })
    XCTAssertThrowsError(try packed[0].read(-1..<1))
    let savedRead = try packed[0].read(0..<1)
    packed[0].releaseResidentStorage()
    XCTAssertThrowsError(try packed[0].read(0..<1))
    XCTAssertEqual(savedRead.contents().load(as: UInt32.self), 0)
  }

  func testEncodedCountsAndMedianMatchNumpy() throws {
    let fixture = try numpyFixture()
    let ops = try MetalImageOperations()
    let raw = (fixture["raw"] as! [Int]).map(UInt16.init)
    let expected = (fixture["corrected"] as! [Int]).map(UInt16.init)
    let source = try MetalEncodedSource(
      shape: fixture["shape"] as! [Int], hotPixelIndices: fixture["bad"] as! [Int],
      device: ops.device)
    for frames in [0..<5, 5..<17] {
      let buffer = ops.device.makeBuffer(
        length: frames.count * 35 * 2, options: .storageModeShared)!
      Array(raw[(frames.lowerBound * 35)..<(frames.upperBound * 35)]).withUnsafeBytes {
        _ = memcpy(buffer.contents(), $0.baseAddress!, $0.count)
      }
      try source.append(buffer, frames: frames.count, verify: true)
    }
    let restored = try source.read(3..<15)
    XCTAssertEqual(
      Array(
        UnsafeBufferPointer(
          start: restored.contents().assumingMemoryBound(to: UInt16.self), count: 12 * 35)),
      Array(expected[(3 * 35)..<(15 * 35)]))
    let dp = Array(
      UnsafeBufferPointer(
        start: source.meanDiffraction.contents().assumingMemoryBound(to: Float.self), count: 35))
    let bf = Array(
      UnsafeBufferPointer(
        start: source.meanBrightField.contents().assumingMemoryBound(to: Float.self), count: 17))
    XCTAssertEqual(dp, (fixture["dp_mean"] as! [Double]).map(Float.init))
    XCTAssertEqual(bf, (fixture["im_bf"] as! [Double]).map(Float.init))
    let maped = try MAPEDNative.from_resident([source], operations: ops)
    try maped.preprocess()
    maped.close()
    XCTAssertFalse(source.isReleased, "Closing MAPED must preserve borrowed acquisitions.")
    source.releaseResidentStorage()
  }
  func testGaussianAndSubpixelCorrelationMatchNumpy() throws {
    let fixture = try numpyFixture()
    let ops = try MetalImageOperations()
    let values = (fixture["image"] as! [Double]).map(Float.init)
    let expected = (fixture["gaussian"] as! [Double]).map(Float.init)
    let image = try ops.image(values: values, rows: 17, columns: 19)
    let actual = try ops.gaussian(image, sigma: 1.25).values()
    for (a, b) in zip(actual, expected) { XCTAssertEqual(a, b, accuracy: 1e-6) }
    let translated = try ops.image(
      values: (fixture["translated"] as! [Double]).map(Float.init), rows: 17, columns: 19)
    let spectrum = try ops.fourier(image)
    let target = try ops.fourier(translated)
    for factor in [1, 2, 3, 100] {
      let shift = try ops.correlation(spectrum, target, upsample_factor: factor).values()
      XCTAssertEqual(shift[0], -3, accuracy: 0.001)
      XCTAssertEqual(shift[1], 4, accuracy: 0.001)
    }
  }
  func testAdvancedNumericsMatchFrozenTorchMPS() throws {
    let path = Bundle.module.url(
      forResource: "torch_mps_parameters", withExtension: "json", subdirectory: "Fixtures")!
    let fixture = try JSONSerialization.jsonObject(with: Data(contentsOf: path)) as! [String: Any]
    let ops = try MetalImageOperations()
    func image(_ rows: Int, _ columns: Int) throws -> GPUImage {
      try ops.image(
        values: (0..<(rows * columns)).map { Float(($0 * 37) % 251 - 125) * 0.03125 },
        rows: rows, columns: columns)
    }
    func check(_ actual: GPUImage, _ observation: [String: Any]) {
      let values = actual.values()
      let indices = observation["indices"] as! [Int]
      if actual.isComplex {
        for (index, expected) in zip(indices, observation["values"] as! [[Double]]) {
          XCTAssertEqual(values[2 * index], Float(expected[0]))
          XCTAssertEqual(values[2 * index + 1], Float(expected[1]))
        }
      } else {
        for (index, expected) in zip(indices, observation["values"] as! [Double]) {
          XCTAssertEqual(values[index], Float(expected))
        }
      }
    }
    let raw = try image(512, 512)
    for (sigma, observation) in fixture["gradient"] as! [String: [String: Any]] {
      check(try ops.gradientMagnitude(raw, sigma: Double(sigma)!), observation)
    }
    check(try ops.fourier(image(520, 520)), fixture["fft"] as! [String: Any])
    for (edge, observation) in fixture["windows"] as! [String: [String: Any]] {
      check(
        try ops.window(
          ops.image(rows: 192, columns: 192, value: 1),
          kind: 1, edge_blend: Double(edge)!), observation)
    }
    check(
      try ops.centered(raw, window: ops.image(rows: 512, columns: 512, value: 1)),
      fixture["centered"] as! [String: Any])
  }
  private func numpyFixture() throws -> [String: Any] {
    let path = Bundle.module.url(
      forResource: "numpy", withExtension: "json", subdirectory: "Fixtures")!
    return try JSONSerialization.jsonObject(with: Data(contentsOf: path)) as! [String: Any]
  }
  func testScaledStorageAndHDF5Reopen() throws {
    let ops = try MetalImageOperations()
    let precision = try MetalPrecision(device: ops.device)
    XCTAssertThrowsError(
      try precision.useSavedReport([
        "storage": "scaled_uint16", "scale": 1.0, "offset": 0.0,
      ]))
    let shape = [17, 19, 64, 64]
    let count = shape.reduce(1, *)
    let values = (0..<count).map { Float(($0 * 37) % 1024) * 0.125 - 10.75 }
    let image = try ops.image(values: values, rows: 17 * 19, columns: 64 * 64)
    try precision.includeRange(image.buffer, count: count)
    try precision.calibrate(shape: shape)
    let codes = try precision.convert(image.buffer, count: count)
    let report = try precision.finish()
    let scale = report["scale"] as! Double
    let offset = report["offset"] as! Double
    let expectedCodes = values.map {
      UInt16(((Double($0) - offset) / scale).rounded(.toNearestOrEven))
    }
    let actualCodes = Array(
      UnsafeBufferPointer(
        start: codes.contents().assumingMemoryBound(to: UInt16.self), count: count))
    let codeMismatch = zip(actualCodes, expectedCodes).enumerated().filter {
      $0.element.0 != $0.element.1
    }
    XCTAssertEqual(codeMismatch.count, 0, "First code differences: \(codeMismatch.prefix(5))")
    let restored = try precision.restore(codes, count: count)
    let expected = expectedCodes.map { Float(Double($0) * scale + offset) }
    let direct = Array(
      UnsafeBufferPointer(
        start: restored.contents().assumingMemoryBound(to: Float.self), count: count))
    let restoreMismatch = zip(direct, expected).enumerated().filter { $0.element.0 != $0.element.1 }
    XCTAssertEqual(
      restoreMismatch.count, 0, "First restoration differences: \(restoreMismatch.prefix(5))")
    let squared = zip(values, expected).reduce(0.0) { $0 + pow(Double($1.0) - Double($1.1), 2) }
    XCTAssertEqual(report["rmse"] as! Double, sqrt(squared / Double(count)), accuracy: 1e-8)
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(
      "native-precision-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let path = folder.appendingPathComponent("scaled_master.h5")
    let writer = try MetalHDF5Writer(path: path, shape: shape, runtime: precision)
    var first = 0
    for frames in [7, 64, 5, 128, 119] {
      let bytes = frames * 4096 * 2
      let piece = ops.device.makeBuffer(length: bytes, options: .storageModeShared)!
      let command = ops.queue.makeCommandBuffer()!
      let blit = command.makeBlitCommandEncoder()!
      blit.copy(
        from: codes, sourceOffset: first * 4096 * 2, to: piece, destinationOffset: 0, size: bytes)
      blit.endEncoding()
      command.commit()
      command.waitUntilCompleted()
      try writer.append(piece, frames: frames)
      first += frames
    }
    XCTAssertFalse(FileManager.default.fileExists(atPath: path.path))
    let json = String(data: try JSONSerialization.data(withJSONObject: report), encoding: .utf8)!
    try writer.finish(metadata: ["quantem_precision_v1": json])
    let reopened = try MetalPackedSource.load(
      path: path, device: ops.device, indexDirectory: folder.appendingPathComponent("index"))
    let output = try reopened.read(0..<(17 * 19))
    let reopenedValues = Array(
      UnsafeBufferPointer(
        start: output.contents().assumingMemoryBound(to: Float.self), count: count))
    XCTAssertEqual(zip(reopenedValues, expected).filter { $0 != $1 }.count, 0)
    XCTAssertEqual(reopened.metadata["scale"] as? Double, scale)
    reopened.releaseResidentStorage()
    XCTAssertEqual(reopened.residentBytes, 0)
    let cancelledPath = folder.appendingPathComponent("cancelled_master.h5")
    let cancelled = try MetalHDF5Writer(path: cancelledPath, shape: shape, runtime: precision)
    try cancelled.append(codes, frames: 7)
    cancelled.cancel()
    XCTAssertThrowsError(try cancelled.finish(metadata: [:]))
    XCTAssertFalse(FileManager.default.fileExists(atPath: cancelledPath.path))
    XCTAssertFalse(
      try FileManager.default.contentsOfDirectory(atPath: folder.path)
        .contains { $0.hasPrefix(".cancelled_master.h5.") })
  }
  func testFourierRoundTripAndTranslation() throws {
    let ops = try MetalImageOperations()
    let rows = 17
    let columns = 19
    var values = [Float](repeating: 0, count: rows * columns)
    values[7 * columns + 8] = 3
    values[5 * columns + 6] = 7
    let image = try ops.image(values: values, rows: rows, columns: columns)
    let spectrum = try ops.fourier(image)
    let restored = try ops.fourier(spectrum, inverse: true).values()
    for (actual, expected) in zip(restored, values) {
      XCTAssertEqual(actual, expected, accuracy: 2e-5)
    }
    let shift = try ops.correlation(spectrum, spectrum).values()
    XCTAssertEqual(shift[0], 0, accuracy: 1e-4)
    XCTAssertEqual(shift[1], 0, accuracy: 1e-4)
  }
  func testNormalizedGridBoundaryMatchesNumpy() throws {
    let path = Bundle.module.url(
      forResource: "normalized_grid", withExtension: "json", subdirectory: "Fixtures")!
    let fixture = try JSONSerialization.jsonObject(with: Data(contentsOf: path)) as! [String: Any]
    let shape = fixture["shape"] as! [Int]
    let ops = try MetalImageOperations()
    let image = try ops.image(rows: shape[0], columns: shape[1], value: 1)
    let shift = try ops.image(
      values: (fixture["shift"] as! [Double]).map(Float.init), rows: 1, columns: 2)
    let translated = try ops.shifted(image, shifts: shift, index: 0).values()
    for (index, expected) in zip(fixture["indices"] as! [Int], fixture["values"] as! [Double]) {
      XCTAssertEqual(translated[index], Float(expected), accuracy: 2e-7, "Boundary pixel \(index)")
    }
  }
  func testParameterChangesAndSavedProvenance() throws {
    let ops = try MetalImageOperations()
    let shape = [8, 8, 64, 64]
    var sources: [MetalEncodedSource] = []
    for index in 0..<2 {
      let values = (0..<shape.reduce(1, *)).map { UInt16(($0 * 37 + index * 13) % 251) }
      let raw = ops.device.makeBuffer(length: values.count * 2, options: .storageModeShared)!
      values.withUnsafeBytes { _ = memcpy(raw.contents(), $0.baseAddress!, $0.count) }
      let source = try MetalEncodedSource(shape: shape, device: ops.device)
      try source.append(raw, frames: 64, verify: true)
      sources.append(source)
    }
    defer { for source in sources { source.releaseResidentStorage() } }
    let maped = try MAPEDNative.from_resident(sources, operations: ops)
    defer { maped.close() }
    XCTAssertThrowsError(try maped.preprocess(scale: 0))
    XCTAssertThrowsError(try maped.preprocess(scale: [1]))
    try maped.preprocess(scale: 2)
    XCTAssertEqual(maped.scales, [2, 2])
    try maped.diffraction_origin(origins: (31, 32))
    XCTAssertEqual(maped.diffraction_origins, [[31, 32], [31, 32]])
    try maped.diffraction_align(edge_blend: 2, upsample_factor: 3)
    try maped.real_space_align(num_iter: 1, edge_filter: false, edge_sigma: 0)
    XCTAssertThrowsError(try maped.real_space_align(num_images: 0))
    XCTAssertThrowsError(try maped.real_space_align(num_iter: 0))
    XCTAssertThrowsError(try maped.real_space_align(edge_sigma: 0))
    XCTAssertThrowsError(try maped.real_space_align(max_shift: Double.greatestFiniteMagnitude))
    XCTAssertThrowsError(try maped.diffraction_origin(origins: [[31, 32]]))
    XCTAssertThrowsError(try maped.merged_region(0..<0))
    let original = try maped.merged_region(0..<1).values()
    try maped.real_space_align(num_images: 1, num_iter: 1, edge_filter: false, edge_sigma: 0)
    try maped.real_space_align(num_iter: 1, edge_filter: false, edge_sigma: 0)
    XCTAssertEqual(try maped.merged_region(0..<1).values(), original)

    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let first = try maped.merge_datasets(
      shift_method: " BILINEAR ", save_to: directory.appendingPathComponent("first_master.h5"),
      verbose: false)
    let firstBuffer = try first.read(0..<1)
    let expected = Array(
      UnsafeBufferPointer(
        start: firstBuffer.contents().assumingMemoryBound(to: Float.self), count: 4096))
    let savedText = qh5_read_root_attribute(
      directory.appendingPathComponent("first_master.h5").path, "quantem_maped_merge_v1")!
    let saved =
      try JSONSerialization.jsonObject(
        with: Data(String(cString: savedText).utf8)) as! [String: Any]
    qh5_free_error(savedText)
    let savedParameters = saved["parameters"] as! [String: [String: Any]]
    XCTAssertEqual(savedParameters["preprocess"]?["scale"] as? [Float], [2, 2])
    XCTAssertEqual(
      savedParameters["diffraction_origin"]?["origins"] as? [[Int]], [[31, 32], [31, 32]])
    XCTAssertEqual(savedParameters["real_space_align"]?["edge_filter"] as? Bool, false)
    XCTAssertTrue(sources.allSatisfy { !$0.isReleased })
    XCTAssertThrowsError(
      try maped.merge_datasets(
        save_to: directory.appendingPathComponent("first_master.h5"), verbose: false))
    XCTAssertFalse(first.isReleased)
    XCTAssertTrue(maped.merged === first)
    let second = try maped.merge_datasets(
      diffraction_pad_val: 0.5, dtype: "scaled_uint16",
      save_to: directory.appendingPathComponent("second_master.h5"), verbose: false)
    XCTAssertTrue(first.isReleased)
    let buffer = try second.read(0..<1)
    XCTAssertEqual(
      Array(
        UnsafeBufferPointer(
          start: buffer.contents().assumingMemoryBound(to: Float.self), count: 4096)),
      expected)
    XCTAssertEqual(maped.parameters["real_space_align"]?["edge_filter"] as? Bool, false)
    XCTAssertEqual(maped.parameters["real_space_align"]?["edge_sigma"] as? Double, 0)
    try maped.real_space_align(num_iter: 1, edge_filter: false, edge_sigma: 0)
    XCTAssertTrue(second.isReleased)
    XCTAssertNil(maped.merged)
    maped.close()
    XCTAssertThrowsError(try maped.merged_region(0..<1))
    XCTAssertTrue(sources.allSatisfy { !$0.isReleased })
  }
}

/// Small independent GPU fixture for an existing resident's exclusion-mask contract.
private final class MaskedCountFixture: MetalResidentCounts {
  let shape: [Int]
  let itemBytes = 4
  let hotPixelIndices: [Int]
  let hotPixelCorrection = "exclude"
  let device: MTLDevice
  let representation: Metal4DSTEMResidentRepresentation = .packed
  private var storage: MTLBuffer?
  var readyFrames: Int { shape[0] * shape[1] }
  var isReleased: Bool { storage == nil }
  var residentBytes: Int { storage?.length ?? 0 }
  init(shape: [Int], values: [UInt32], bad: [Int], device: MTLDevice) throws {
    self.shape = shape
    self.hotPixelIndices = bad
    self.device = device
    let buffer = device.makeBuffer(length: values.count * 4, options: .storageModeShared)!
    let pixels = shape[2] * shape[3]
    for (i, value) in values.enumerated() {
      buffer.contents().storeBytes(
        of: bad.contains(i % pixels) ? UInt32(0) : value,
        toByteOffset: i * 4, as: UInt32.self)
    }
    storage = buffer
  }
  func encodeRead(_ frames: Range<Int>, into result: MTLBuffer, command: MTLCommandBuffer) throws {
    guard let storage, frames.lowerBound >= 0, frames.upperBound <= readyFrames else {
      throw Metal4DSTEMStreamingIOError.invalidRequest("Read live fixture frames.")
    }
    let bytes = shape[2] * shape[3] * 4
    let copy = command.makeBlitCommandEncoder()!
    copy.copy(
      from: storage, sourceOffset: frames.lowerBound * bytes, to: result, destinationOffset: 0,
      size: frames.count * bytes)
    copy.endEncoding()
  }
  func countMeans() throws -> (diffraction: MTLBuffer, brightField: MTLBuffer) {
    throw Metal4DSTEMStreamingIOError.invalidRequest("Means must use the corrected GPU view.")
  }
  func releaseResidentStorage() { storage = nil }
}
