import Foundation

/// One bundled model. Both entries are plain GGUF files read by llama.cpp.
struct ModelSpec: Identifiable, Hashable {
    let id: String
    let name: String
    let file: String
    let detail: String
    /// SmolLM2 expects a system turn; Falcon-H1 does not use one.
    let systemPrompt: String?
}

enum AppConfig {
    static let models: [ModelSpec] = [
        ModelSpec(id: "falcon",
                  name: "Falcon-H1 90M",
                  file: "Falcon-H1-Tiny-90M-Instruct-Q4_K_M",
                  detail: "hybrid SSM + attention · 57 MB",
                  systemPrompt: nil),
        ModelSpec(id: "smollm",
                  name: "SmolLM2 135M",
                  file: "SmolLM2-135M-Instruct-Q4_K_M",
                  detail: "Llama · 104 MB",
                  systemPrompt: "You are a helpful AI assistant named SmolLM, trained by Hugging Face"),
    ]
    static let defaultModel = models[0]

    /// Attention KV cache length, in tokens.
    static let contextTokens = 1280
    /// ~10 paragraphs of prose.
    static let maxNewTokens  = 1024
    /// Fixed prompt used by the FH1_AUTORUN benchmark path.
    static let benchPrompt   = "What is the sun?"
}

enum EngineError: LocalizedError {
    case notBundled(String)
    case load(String)
    case context

    var errorDescription: String? {
        switch self {
        case .notBundled(let n): return "\(n).gguf is missing from the app bundle"
        case .load(let m):       return m
        case .context:          return "could not allocate the inference context"
        }
    }
}

/// `phys_footprint` — the figure jetsam actually watches on watchOS.
func footprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1e6 : 0
}

/// Trivially thread-safe boolean.
final class Flag: @unchecked Sendable {
    private let lock = NSLock()
    private var v = false
    var value: Bool {
        get { lock.lock(); defer { lock.unlock() }; return v }
        set { lock.lock(); v = newValue; lock.unlock() }
    }
}

/// A token can end mid-UTF-8. Buffer partial scalars so the UI never shows
/// a replacement character.
struct UTF8Stream {
    private var buf: [UInt8] = []

    mutating func push(_ bytes: [UInt8]) -> String {
        buf.append(contentsOf: bytes)
        let split = Self.completePrefixLength(buf)
        guard split > 0 else { return "" }
        let head = Array(buf[0..<split])
        buf.removeFirst(split)
        return String(decoding: head, as: UTF8.self)
    }

    mutating func drain() -> String {
        guard !buf.isEmpty else { return "" }
        defer { buf.removeAll() }
        return String(decoding: buf, as: UTF8.self)
    }

    private static func completePrefixLength(_ b: [UInt8]) -> Int {
        var i = b.count
        var scanned = 0
        while i > 0 && scanned < 4 {
            let c = b[i - 1]
            if c & 0x80 == 0 { return i }
            if c & 0xC0 == 0xC0 {
                let need = c >= 0xF0 ? 4 : (c >= 0xE0 ? 3 : 2)
                return (b.count - (i - 1)) >= need ? b.count : i - 1
            }
            i -= 1; scanned += 1
        }
        return b.count
    }
}
