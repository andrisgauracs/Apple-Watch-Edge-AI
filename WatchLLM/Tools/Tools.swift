import Foundation

/// A declarative web lookup. The *code* here is generic and ships in the repo;
/// the actual tool definitions live in an optional, git-ignored `Tools.json`
/// bundle resource. With no such file present `ToolBox.specs` is empty, no
/// network code ever runs, and the app is fully offline.
struct WebToolSpec: Codable, Sendable {
    /// Shown in the UI when the tool fires.
    let name: String
    /// Lowercased substrings; if any appears in the query, the tool runs.
    let triggers: [String]
    let url: String
    /// When set, `{startDate}` and `{endDate}` in the URL are replaced with an
    /// ISO date window ending today.
    let windowDays: Int?
    /// Trims long text values to at most this many characters, cut at a
    /// sentence boundary. A 90M model drowns in a full encyclopaedia intro.
    let maxChars: Int?
    let headers: [String: String]?
    /// Named values to pull out of the response. Each value is a list of keys to
    /// walk; numeric components index arrays.
    /// e.g. "subs": ["items", "0", "statistics", "subscriberCount"]
    let fields: [String: [String]]?
    /// Single-field shorthand, exposed to the template as `{value}`.
    let jsonPath: [String]?
    /// Values computed from fetched ones, evaluated in order so later entries
    /// can use earlier ones. Expressions are strictly left-to-right (no operator
    /// precedence): "subMin / total * 100".
    /// Small models cannot do arithmetic on six-digit numbers, so any figure the
    /// answer must be correct about has to be worked out here.
    let derived: [DerivedField]?
    /// Optional OAuth refresh-token exchange, for APIs that reject plain API
    /// keys (the YouTube Analytics API is one).
    let auth: OAuthRefresh?
    /// `{name}` placeholders are replaced with the fetched or derived values.
    let factTemplate: String
}

struct DerivedField: Codable, Sendable {
    let name: String
    let expr: String
}

struct OAuthRefresh: Codable, Sendable {
    let tokenURL: String
    let clientId: String
    /// Absent for public clients (Google's iOS/Android types have no secret).
    /// Desktop-app clients do have one and it must be sent.
    let clientSecret: String?
    let refreshToken: String
}

enum ToolBox {
    /// Loaded once at launch. Empty unless a `Tools.json` resource is bundled.
    static let specs: [WebToolSpec] = {
        guard let url = Bundle.main.url(forResource: "Tools", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return [] }
        do {
            return try JSONDecoder().decode([WebToolSpec].self, from: data)
        } catch {
            return []
        }
    }()

    static var isEnabled: Bool { !specs.isEmpty }

    static func match(_ query: String) -> WebToolSpec? {
        let q = query.lowercased()
        return specs.first { $0.triggers.contains { q.contains($0.lowercased()) } }
    }

    enum ToolError: LocalizedError {
        case badURL, badResponse(Int), pathMissing, authFailed
        var errorDescription: String? {
            switch self {
            case .badURL:      return "tool URL is malformed"
            case .badResponse(let code): return "tool request failed (HTTP \(code))"
            case .pathMissing: return "tool response did not contain the expected field"
            case .authFailed:  return "could not refresh the OAuth token"
            }
        }
    }

    /// Walks `path` through a decoded JSON tree.
    private static func resolve(_ root: Any, _ path: [String]) -> Any? {
        var node: Any? = root
        for key in path {
            if key == "*" {
                if let dict = node as? [String: Any] { node = dict.values.first }
                else if let arr = node as? [Any] { node = arr.first }
                else { return nil }
            } else if let idx = Int(key), let arr = node as? [Any], arr.indices.contains(idx) {
                node = arr[idx]
            } else if let dict = node as? [String: Any] {
                node = dict[key]
            } else {
                return nil
            }
            if node == nil { return nil }
        }
        return node
    }

    /// Evaluates a strictly left-to-right expression over named numbers and
    /// literals, e.g. "subMin / total * 100". No operator precedence.
    private static func evaluate(_ expr: String, _ nums: [String: Double]) -> Double? {
        var tokens: [String] = []
        var cur = ""
        for ch in expr {
            if "+-*/".contains(ch) {
                tokens.append(cur.trimmingCharacters(in: .whitespaces)); cur = ""
                tokens.append(String(ch))
            } else {
                cur.append(ch)
            }
        }
        tokens.append(cur.trimmingCharacters(in: .whitespaces))
        tokens.removeAll { $0.isEmpty }
        guard let first = tokens.first, var acc = nums[first] ?? Double(first) else { return nil }
        var i = 1
        while i + 1 < tokens.count {
            let op = tokens[i]
            let name = tokens[i + 1]
            guard let v = nums[name] ?? Double(name) else { return nil }
            switch op {
            case "/": if v == 0 { return nil }; acc /= v
            case "*": acc *= v
            case "+": acc += v
            case "-": acc -= v
            default:  return nil
            }
            i += 2
        }
        return acc
    }

    /// Exchanges a long-lived refresh token for a short-lived access token.
    private static func accessToken(_ auth: OAuthRefresh) async throws -> String {
        guard let url = URL(string: auth.tokenURL) else { throw ToolError.badURL }
        var req = URLRequest(url: url, timeoutInterval: 8)
        req.httpMethod = "POST"
        req.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        var body = URLComponents()
        var items: [URLQueryItem] = [
            .init(name: "client_id",     value: auth.clientId),
            .init(name: "refresh_token", value: auth.refreshToken),
            .init(name: "grant_type",    value: "refresh_token"),
        ]
        if let secret = auth.clientSecret, !secret.isEmpty {
            items.append(.init(name: "client_secret", value: secret))
        }
        body.queryItems = items
        req.httpBody = body.percentEncodedQuery?.data(using: .utf8)
        let (data, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let token = obj["access_token"] as? String else {
            throw ToolError.authFailed
        }
        return token
    }

    /// Performs the lookup and returns a single sentence of fact to prepend to
    /// the prompt. Runs *before* generation starts — the model itself never
    /// touches the network.
    static func run(_ spec: WebToolSpec, query: String = "") async throws -> String {
        var urlString = spec.url
        if let days = spec.windowDays {
            let f = DateFormatter()
            f.dateFormat = "yyyy-MM-dd"
            f.timeZone = TimeZone(identifier: "UTC")
            let end = Date()
            let start = end.addingTimeInterval(-Double(days) * 86_400)
            urlString = urlString
                .replacingOccurrences(of: "{startDate}", with: f.string(from: start))
                .replacingOccurrences(of: "{endDate}",   with: f.string(from: end))
        }
        if urlString.contains("{query}") {
            // Strip the trigger phrase so the search sees the topic, not the ask.
            var topic = query.lowercased()
            for t in spec.triggers where topic.hasPrefix(t.lowercased()) {
                topic = String(topic.dropFirst(t.count))
            }
            topic = topic.trimmingCharacters(in: CharacterSet(charactersIn: " ?.!,"))
            let encoded = topic.addingPercentEncoding(
                withAllowedCharacters: .alphanumerics) ?? topic
            urlString = urlString.replacingOccurrences(of: "{query}", with: encoded)
        }
        guard let url = URL(string: urlString) else { throw ToolError.badURL }
        var req = URLRequest(url: url, timeoutInterval: 8)
        spec.headers?.forEach { req.setValue($1, forHTTPHeaderField: $0) }
        if let auth = spec.auth {
            let token = try await accessToken(auth)
            req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        let (data, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse else { throw ToolError.badResponse(0) }
        guard (200..<300).contains(http.statusCode) else {
            throw ToolError.badResponse(http.statusCode)
        }
        let json = try JSONSerialization.jsonObject(with: data)

        var paths = spec.fields ?? [:]
        if let legacy = spec.jsonPath { paths["value"] = legacy }
        guard !paths.isEmpty else { throw ToolError.pathMissing }

        var text: [String: String] = [:]
        var nums: [String: Double] = [:]
        for (name, path) in paths {
            guard let node = resolve(json, path) else { throw ToolError.pathMissing }
            var raw = "\(node)"
            if let limit = spec.maxChars, raw.count > limit {
                let cut = String(raw.prefix(limit))
                // prefer to end on a full sentence
                if let dot = cut.lastIndex(of: ".") {
                    raw = String(cut[...dot])
                } else {
                    raw = cut
                }
            }
            text[name] = formatted(raw)
            if let d = Double(raw) { nums[name] = d }
        }
        for field in spec.derived ?? [] {
            guard let v = evaluate(field.expr, nums) else { continue }
            let name = field.name
            nums[name] = v
            // Ratios read better with one decimal; counts read better grouped.
            text[name] = v < 100 ? String(format: "%.1f", v) : formatted(String(Int(v.rounded())))
        }

        var out = spec.factTemplate
        for (name, value) in text {
            out = out.replacingOccurrences(of: "{\(name)}", with: value)
        }
        return out
    }

    /// Groups digits so a subscriber count reads naturally when spoken back.
    private static func formatted(_ raw: String) -> String {
        guard let n = Int(raw) else { return raw }
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? raw
    }
}
