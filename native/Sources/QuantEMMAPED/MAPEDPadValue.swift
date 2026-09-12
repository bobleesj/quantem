/// The existing MAPED padding choice: a named statistic or a numerical value.
/// Padding remains presentation-only in native alignment and inactive in the
/// supported zero-padding resident merge; the choice is retained in provenance.
public enum MAPEDPadValue: ExpressibleByStringLiteral, ExpressibleByFloatLiteral,
  ExpressibleByIntegerLiteral
{
  case statistic(String)
  case value(Double)

  public init(stringLiteral value: String) { self = .statistic(value) }
  public init(floatLiteral value: Double) { self = .value(value) }
  public init(integerLiteral value: Int) { self = .value(Double(value)) }

  var metadata: Any {
    switch self {
    case .statistic(let value): return value
    case .value(let value): return value
    }
  }
}
