import Metal4DSTEMStreamingIO
import MetalScientificNumerics
import QuantEMMAPED
import XCTest

final class MAPEDNativeTests: XCTestCase {
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
    try writer.append(codes, frames: 17 * 19)
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
}
