import SwiftUI

@main
struct ReviveApp: App {
    @AppStorage("appMode") private var appMode: AppMode = .unset

    var body: some Scene {
        WindowGroup {
            switch appMode {
            case .unset:
                ModeSelectionView(appMode: $appMode)
            case .worker:
                WorkerView()
            case .coordinator:
                CoordinatorView()
            }
        }
    }
}

enum AppMode: String, RawRepresentable {
    case unset = "unset"
    case worker = "worker"
    case coordinator = "coordinator"
}

struct ModeSelectionView: View {
    @Binding var appMode: AppMode

    var body: some View {
        ZStack {
            Color(hex: "#0a0a0a").ignoresSafeArea()
            VStack(spacing: 40) {
                VStack(spacing: 8) {
                    Text("R E V I V E")
                        .font(.system(size: 36, weight: .black, design: .monospaced))
                        .foregroundColor(Color(hex: "#00ff88"))
                        .shadow(color: Color(hex: "#00ff88").opacity(0.5), radius: 20)
                    Text("Phone Swarm Collective Intelligence")
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundColor(.gray)
                }
                .padding(.top, 60)

                VStack(spacing: 16) {
                    ModeCard(
                        title: "⚡ Worker",
                        subtitle: "Run a local model. Join the swarm.",
                        description: "This device will load a GGUF model and serve inference requests to the coordinator.",
                        color: Color(hex: "#00ff88")
                    ) {
                        appMode = .worker
                    }

                    ModeCard(
                        title: "🧠 Coordinator",
                        subtitle: "Orchestrate the swarm. Synthesize answers.",
                        description: "This device discovers all workers, fans out queries, and runs MoA aggregation. Best for iPad.",
                        color: Color(hex: "#4a90d9")
                    ) {
                        appMode = .coordinator
                    }
                }
                .padding(.horizontal, 24)

                Spacer()

                Text("Version 1.0 · HackPrinceton 2025")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(Color(hex: "#333333"))
                    .padding(.bottom, 20)
            }
        }
    }
}

struct ModeCard: View {
    let title: String
    let subtitle: String
    let description: String
    let color: Color
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 8) {
                Text(title)
                    .font(.system(size: 20, weight: .bold, design: .monospaced))
                    .foregroundColor(color)
                Text(subtitle)
                    .font(.system(size: 13, weight: .semibold, design: .monospaced))
                    .foregroundColor(.white)
                Text(description)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.gray)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(hex: "#111111"))
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(color.opacity(0.4), lineWidth: 1)
            )
            .cornerRadius(12)
        }
        .buttonStyle(.plain)
    }
}
