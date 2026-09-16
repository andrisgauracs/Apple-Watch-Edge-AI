import Foundation

/// The inference engine: upstream llama.cpp, built for watchOS/arm64_32 with a
/// 4-line patch. See docs/LLAMACPP_ON_WATCHOS.md for how the libraries in
/// vendor/llamacpp are produced.
final class LlamaCppEngine: @unchecked Sendable {

    private static var backendReady = false

    private var model: OpaquePointer?
    private var ctx: OpaquePointer?
    private var sampler: UnsafeMutablePointer<llama_sampler>?
    private var vocab: OpaquePointer?   // llama_vocab is opaque in llama.h

    private(set) var spec: ModelSpec?
    private(set) var weightsMB = 0.0
    private(set) var contextMB = 0.0
    private(set) var threads   = 0
    private(set) var eosToken: Int32 = -1
    private(set) var imEndToken: Int32 = -1

    deinit { unload() }

    func load(_ spec: ModelSpec, threads nThreads: Int) throws {
        unload()
        if !Self.backendReady { llama_backend_init(); Self.backendReady = true }

        guard let url = Bundle.main.url(forResource: spec.file, withExtension: "gguf") else {
            throw EngineError.notBundled(spec.file)
        }
        var mp = llama_model_default_params()
        mp.n_gpu_layers = 0
        guard let m = llama_model_load_from_file(url.path, mp) else {
            throw EngineError.load("llama.cpp could not load \(spec.file).gguf")
        }
        var cp = llama_context_default_params()
        cp.n_ctx           = UInt32(AppConfig.contextTokens)
        // We decode one token at a time, so a large batch only inflates the
        // compute buffers. On a 1 GB watch that is the difference between
        // comfortable and jetsam bait.
        cp.n_batch         = 32
        cp.n_ubatch        = 32
        cp.n_threads       = Int32(nThreads)
        cp.n_threads_batch = Int32(nThreads)
        cp.no_perf         = true
        guard let c = llama_init_from_model(m, cp) else {
            llama_model_free(m)
            throw EngineError.context
        }
        model = m; ctx = c
        vocab = llama_model_get_vocab(m)
        self.spec = spec
        self.threads = nThreads
        // llama.cpp doesn't expose a byte count as directly; use the file size,
        // which is what actually gets mapped.
        let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
        weightsMB = Double(size ?? 0) / 1e6
        // KV cache: n_ctx x n_layer x n_head_kv x head_dim x 2 x 2 bytes (f16)
        contextMB = Double(AppConfig.contextTokens) * 24.0 * 2.0 * 64.0 * 2.0 * 2.0 / 1e6
        if let v = vocab {
            eosToken = llama_vocab_eos(v)
            imEndToken = tokenForText("<|im_end|>")
        }
        setSampling(factual: false)
    }

    func unload() {
        if let sampler { llama_sampler_free(sampler) }
        if let ctx { llama_free(ctx) }
        if let model { llama_model_free(model) }
        sampler = nil; ctx = nil; model = nil; vocab = nil; spec = nil
    }

    func reset() {
        guard let ctx else { return }
        llama_memory_clear(llama_get_memory(ctx), true)
    }

    func setSampling(factual: Bool) {
        if let sampler { llama_sampler_free(sampler) }
        let chain = llama_sampler_chain_init(llama_sampler_chain_default_params())
        if factual {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy())
        } else {
            llama_sampler_chain_add(chain, llama_sampler_init_top_k(40))
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95, 1))
            llama_sampler_chain_add(chain, llama_sampler_init_temp(0.7))
            llama_sampler_chain_add(chain, llama_sampler_init_dist(UInt32.random(in: 1...UInt32.max)))
        }
        sampler = chain
    }

    private func tokenForText(_ s: String) -> Int32 {
        guard let vocab else { return -1 }
        var buf = [Int32](repeating: 0, count: 8)
        let n = llama_tokenize(vocab, s, Int32(s.utf8.count), &buf, 8, false, true)
        return n == 1 ? buf[0] : -1
    }

    func tokenizeChat(prompt: String, context: String? = nil) -> [Int32] {
        guard let vocab else { return [] }
        var sysParts: [String] = []
        if let context { sysParts.append(context) }
        if let sys = spec?.systemPrompt { sysParts.append(sys) }
        var text = ""
        if !sysParts.isEmpty {
            text += "<|im_start|>system\n" + sysParts.joined(separator: " ") + "<|im_end|>\n"
        }
        text += "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"

        var buf = [Int32](repeating: 0, count: 2048)
        let n = llama_tokenize(vocab, text, Int32(text.utf8.count), &buf, 2048, true, true)
        guard n > 0 else { return [] }
        return Array(buf[0..<Int(n)])
    }

    func step(_ token: Int32) -> Int32? {
        guard let ctx, let sampler else { return nil }
        var t = token
        let batch = llama_batch_get_one(&t, 1)
        guard llama_decode(ctx, batch) == 0 else { return nil }
        return llama_sampler_sample(sampler, ctx, -1)
    }

    func piece(_ token: Int32) -> [UInt8]? {
        guard let vocab else { return nil }
        var buf = [CChar](repeating: 0, count: 256)
        let n = llama_token_to_piece(vocab, token, &buf, 256, 0, false)
        guard n > 0 else { return nil }
        return buf[0..<Int(n)].map { UInt8(bitPattern: $0) }
    }

    func isStop(_ token: Int32) -> Bool {
        guard let vocab else { return token == eosToken }
        return llama_vocab_is_eog(vocab, token) || token == imEndToken
    }
}
