import SwiftUI

@main
struct FalconWatchApp: App {
    @StateObject private var runner = LLMRunner()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            ContentView(runner: runner)
                .onChange(of: scenePhase) { _, phase in
                    // watchOS suspends a backgrounded app within seconds. Rather than
                    // fight that, park generation at a token boundary; the recurrent
                    // state stays live so it resumes exactly where it stopped.
                    switch phase {
                    case .active:                 runner.resumeIfParked()
                    case .inactive, .background:  runner.park()
                    @unknown default:             break
                    }
                }
        }
    }
}
