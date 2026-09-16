import SwiftUI

struct ContentView: View {
    @ObservedObject var runner: LLMRunner
    @State private var query = ""

    var body: some View {
        NavigationStack {
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 10) {
                        modelLink
                        askControl
                        if !query.isEmpty {
                            Text(query)
                                .font(.system(size: 14, weight: .semibold))
                                .foregroundStyle(.secondary)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        if let note = runner.toolNote {
                            HStack(spacing: 4) {
                                Image(systemName: "network")
                                Text(note)
                            }
                            .font(.system(size: 11))
                            .foregroundStyle(.tint)
                        }
                        if !runner.output.isEmpty {
                            Text(runner.output)
                                .font(.system(size: 15))
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        Color.clear.frame(height: 1).id("tail")
                        if runner.stats.generatedTokens > 0 {
                            statsView
                        }
                    }
                    .padding(.horizontal, 2)
                }
                .onChange(of: runner.output) { _, _ in
                    withAnimation(.linear(duration: 0.1)) { proxy.scrollTo("tail", anchor: .bottom) }
                }
            }
            .navigationTitle(runner.model.name)
            .navigationBarTitleDisplayMode(.inline)
        }
        .task { runner.load() }
    }

    @ViewBuilder private var modelLink: some View {
        NavigationLink {
            ModelPicker(runner: runner)
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "cpu")
                Text(runner.model.name)
                Spacer()
                Image(systemName: "chevron.right").foregroundStyle(.tertiary)
            }
            .font(.system(size: 13))
        }
        .disabled(runner.state == .generating)
    }

    @ViewBuilder private var askControl: some View {
        switch runner.state {
        case .idle, .loading:
            HStack(spacing: 6) {
                ProgressView()
                Text("Loading model…").font(.footnote)
            }
        case .failed(let msg):
            Text(msg).font(.footnote).foregroundStyle(.red)
        case .generating:
            Button("Stop", role: .destructive) { runner.stop() }
        case .parked:
            HStack(spacing: 6) {
                Image(systemName: "pause.circle")
                Text("Paused — raise wrist").font(.footnote)
            }
        case .ready:
            VStack(spacing: 6) {
                // Presents the system input sheet: Dictation, Scribble or keyboard.
                // No microphone permission needed — the system does the capture.
                TextFieldLink(prompt: Text("Ask")) {
                    Label(query.isEmpty ? "Speak" : "Speak again", systemImage: "mic.fill")
                } onSubmit: { spoken in
                    let t = spoken.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !t.isEmpty else { return }
                    query = t
                }
                if !query.isEmpty {
                    Button {
                        runner.generate(prompt: query)
                    } label: {
                        Label("Ask", systemImage: "arrow.up.circle.fill")
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
        }
    }

    @ViewBuilder private var statsView: some View {
        let s = runner.stats
        VStack(alignment: .leading, spacing: 3) {
            Divider()
            row("tokens/sec", s.tokensPerSecond > 0 ? String(format: "%.2f", s.tokensPerSecond) : "–")
            row("first token", s.timeToFirstTokenS > 0 ? String(format: "%.2f s", s.timeToFirstTokenS) : "–")
            row("generated", "\(s.generatedTokens) tok")
            row("prompt", "\(s.promptTokens) tok")
            row("peak memory", s.peakFootprintMB > 0 ? String(format: "%.0f MB", s.peakFootprintMB) : "–")
            row("weights", String(format: "%.0f MB", s.weightsMB))
            row("kv cache", String(format: "%.0f MB", s.contextMB))
            row("threads", "\(s.threads)")
        }
        .font(.system(size: 12, design: .monospaced))
        .foregroundStyle(.secondary)
    }

    private func row(_ k: String, _ v: String) -> some View {
        HStack {
            Text(k)
            Spacer()
            Text(v).foregroundStyle(.primary)
        }
    }
}

private struct ModelPicker: View {
    @ObservedObject var runner: LLMRunner
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        List(AppConfig.models) { spec in
            Button {
                runner.select(spec)
                dismiss()
            } label: {
                HStack(alignment: .firstTextBaseline) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(spec.name).font(.system(size: 14, weight: .semibold))
                        Text(spec.detail).font(.system(size: 11)).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if spec == runner.model {
                        Image(systemName: "checkmark").foregroundStyle(.tint)
                    }
                }
            }
        }
        .navigationTitle("Model")
    }
}
