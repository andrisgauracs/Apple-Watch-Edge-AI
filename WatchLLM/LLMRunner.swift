import Foundation
import Combine
import os

/// Drives llama.cpp on a background queue and publishes UI state on the main actor.
@MainActor
final class LLMRunner: ObservableObject {

    enum State: Equatable {
        case idle, loading, ready, generating, parked, failed(String)
    }

    struct Stats: Equatable {
        var promptTokens      = 0
        var generatedTokens   = 0
        var timeToFirstTokenS = 0.0
        var tokensPerSecond   = 0.0
        var peakFootprintMB   = 0.0
        var weightsMB         = 0.0
        var contextMB         = 0.0
        var threads           = 0
    }

    @Published private(set) var state: State = .idle
    @Published private(set) var output = ""
    @Published private(set) var stats = Stats()
    @Published private(set) var model: ModelSpec = AppConfig.defaultModel
    /// Set when a web tool supplied a fact for this answer; nil otherwise.
    @Published private(set) var toolNote: String?

    private let engine = LlamaCppEngine()
    private let queue  = DispatchQueue(label: "llm.inference", qos: .userInitiated)
    private let cancelFlag = Flag()
    private let parkFlag   = Flag()
    private let log = Logger(subsystem: "com.andrisgauracs.WatchLLM", category: "bench")

    /// Set FH1_AUTORUN=1 in the scheme to generate on launch and log a
    /// benchmark line — used for repeatable measurement runs.
    private var autorun: Bool { ProcessInfo.processInfo.environment["FH1_AUTORUN"] != nil }
    private var benchPrompt: String {
        ProcessInfo.processInfo.environment["FH1_PROMPT"] ?? AppConfig.benchPrompt
    }

    /// Snapshot of an interrupted run. Only ever touched on `queue`.
    private final class Parked: @unchecked Sendable { var run: Run? }
    private let parked = Parked()

    nonisolated fileprivate struct Run {
        /// When a tool supplied the facts, stop after the first sentence. The
        /// model relays a fetched number correctly and then keeps going and
        /// invents arithmetic about it; the first sentence is the reliable part.
        var stopAtSentence = false
        var answer = ""
        var next: Int32
        var remaining: Int
        var generated: Int
        var decoder: UTF8Stream
        var startedAt: CFAbsoluteTime
        var firstTokenAt: CFAbsoluteTime?
    }

    // MARK: - lifecycle

    func load() { load(AppConfig.defaultModel) }

    /// Switches which bundled model is loaded.
    func select(_ spec: ModelSpec) {
        guard spec != model, state != .generating else { return }
        model = spec
        output = ""
        toolNote = nil
        stats = Stats()
        state = .idle
        load(spec)
    }

    private func load(_ spec: ModelSpec) {
        guard state == .idle else { return }
        state = .loading
        // Series 6 is dual-core; more threads than cores just adds contention.
        let threads = max(1, min(2, ProcessInfo.processInfo.activeProcessorCount))
        let engine = self.engine
        queue.async { [weak self] in
            do {
                try engine.load(spec, threads: threads)
                let w = engine.weightsMB, c = engine.contextMB, t = engine.threads
                Task { @MainActor in
                    guard let self else { return }
                    self.stats.weightsMB = w
                    self.stats.contextMB = c
                    self.stats.threads = t
                    self.state = .ready
                    self.log.notice("loaded: weights=\(w, format: .fixed(precision: 1)) MB ctx=\(c, format: .fixed(precision: 1)) MB threads=\(t)")
                    if self.autorun { self.generate(prompt: self.benchPrompt) }
                }
            } catch {
                let msg = error.localizedDescription
                Task { @MainActor in self?.state = .failed(msg) }
            }
        }
    }

    // MARK: - generation

    func generate(prompt: String) {
        guard state == .ready else { return }
        cancelFlag.value = false
        parkFlag.value = false
        output = ""
        toolNote = nil
        stats.generatedTokens = 0
        stats.tokensPerSecond = 0
        stats.timeToFirstTokenS = 0
        stats.peakFootprintMB = 0
        state = .generating

        // A matching web tool runs here, before any decoding starts, and its
        // result is folded into the prompt. Inference itself is always offline.
        if let spec = ToolBox.match(prompt) {
            Task { @MainActor in
                do {
                    let fact = try await ToolBox.run(spec, query: prompt)
                    self.toolNote = spec.name
                    self.log.notice("TOOL \(spec.name, privacy: .public) -> \(fact, privacy: .public)")
                    self.startDecoding(prompt: prompt, context: fact)
                } catch {
                    // Fail closed. A question that needs live data, answered
                    // with no live data, produces confident invented numbers.
                    self.toolNote = "\(spec.name) unavailable"
                    self.output = "Couldn't fetch \(spec.name), so I didn't ask the model — "
                                + "it would have made the numbers up.\n\n\(error.localizedDescription)"
                    self.state = .ready
                    self.log.error("TOOL \(spec.name, privacy: .public) failed: \(error.localizedDescription, privacy: .public)")
                }
            }
        } else {
            startDecoding(prompt: prompt, context: nil)
        }
    }

    private func startDecoding(prompt: String, context: String?) {
        let engine = self.engine
        queue.async { [weak self] in
            guard let self else { return }
            self.parked.run = nil
            engine.reset()
            engine.setSampling(factual: context != nil)

            let toks = engine.tokenizeChat(prompt: prompt, context: context)
            guard !toks.isEmpty else {
                Task { @MainActor in self.state = .failed("prompt produced no tokens") }
                return
            }
            Task { @MainActor in self.stats.promptTokens = toks.count }

            let started = CFAbsoluteTimeGetCurrent()
            var next: Int32?
            for t in toks {
                next = engine.step(t)
                if next == nil { break }
            }
            guard let first = next else {
                Task { @MainActor in self.state = .failed("prompt is longer than the context window") }
                return
            }
            self.runLoop(Run(stopAtSentence: context != nil,
                             next: first, remaining: AppConfig.maxNewTokens, generated: 0,
                             decoder: UTF8Stream(), startedAt: started, firstTokenAt: nil))
        }
    }

    func stop() {
        cancelFlag.value = true
        if state == .parked { state = .ready }
    }

    /// watchOS suspends a backgrounded app within seconds. Instead of racing that,
    /// stop at a token boundary; the KV cache stays resident so resuming is exact.
    func park() {
        guard state == .generating else { return }
        parkFlag.value = true
        state = .parked
    }

    func resumeIfParked() {
        guard state == .parked else { return }
        parkFlag.value = false
        state = .generating
        queue.async { [weak self] in
            guard let self, let run = self.parked.run else {
                Task { @MainActor in self?.state = .ready }
                return
            }
            self.parked.run = nil
            self.runLoop(run)
        }
    }

    /// True once the text ends on a sentence terminator. The digit guard keeps
    /// "25.7" from looking like the end of a sentence.
    nonisolated private static func endsSentence(_ text: String) -> Bool {
        guard text.count >= 20 else { return false }
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let last = t.last, ".!?".contains(last) else { return false }
        let before = t.dropLast().last
        return !(before?.isNumber ?? false)
    }

    /// Runs on `queue` until the run finishes, is cancelled, or is parked.
    nonisolated private func runLoop(_ start: Run) {
        var r = start
        var pending = ""
        var lastFlush = 0.0
        var peak = 0.0
        var emitted = 0

        func flush(force: Bool) {
            let now = CFAbsoluteTimeGetCurrent()
            // SwiftUI re-lays-out the whole transcript on every update, so back
            // off as it grows or the UI competes with inference for two cores.
            let interval = emitted < 1200 ? 0.08 : (emitted < 4000 ? 0.20 : 0.40)
            guard force || now - lastFlush > interval else { return }
            lastFlush = now
            peak = max(peak, footprintMB())

            let chunk = pending; pending = ""
            emitted += chunk.count
            let generated = r.generated
            let ttft = (r.firstTokenAt ?? now) - r.startedAt
            let elapsed = now - (r.firstTokenAt ?? now)
            let rate = elapsed > 0 ? Double(max(0, generated - 1)) / elapsed : 0
            let peakNow = peak

            Task { @MainActor [weak self] in
                guard let self else { return }
                if !chunk.isEmpty { self.output += chunk }
                self.stats.generatedTokens = generated
                self.stats.tokensPerSecond = rate
                self.stats.timeToFirstTokenS = ttft
                self.stats.peakFootprintMB = max(self.stats.peakFootprintMB, peakNow)
            }
        }

        while r.remaining > 0 {
            if cancelFlag.value {
                pending += r.decoder.drain(); flush(force: true)
                Task { @MainActor [weak self] in self?.state = .ready }
                return
            }
            if parkFlag.value {
                flush(force: true)
                parked.run = r
                return
            }

            let tok = r.next
            if engine.isStop(tok) { break }
            if r.firstTokenAt == nil { r.firstTokenAt = CFAbsoluteTimeGetCurrent() }

            if let bytes = engine.piece(tok) {
                let piece = r.decoder.push(bytes)
                pending += piece
                r.answer += piece
            }
            r.generated += 1
            r.remaining -= 1

            if r.stopAtSentence && Self.endsSentence(r.answer) { break }

            guard let next = engine.step(tok) else { break }
            r.next = next
            flush(force: false)
        }

        pending += r.decoder.drain()
        flush(force: true)
        Task { @MainActor [weak self] in
            guard let self else { return }
            if self.state == .generating { self.state = .ready }
            let s = self.stats
            self.log.notice("""
                BENCH tok/s=\(s.tokensPerSecond, format: .fixed(precision: 3)) \
                ttft=\(s.timeToFirstTokenS, format: .fixed(precision: 3))s \
                gen=\(s.generatedTokens) prompt=\(s.promptTokens) \
                peak=\(s.peakFootprintMB, format: .fixed(precision: 1))MB \
                threads=\(s.threads)
                """)
            self.log.notice("OUTPUT \(self.output, privacy: .public)")
        }
    }
}
