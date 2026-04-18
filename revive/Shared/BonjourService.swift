import Foundation
import Network

// MARK: - Bonjour Advertisement (Worker side)

/// Advertises this device as a REVIVE swarm node on the local network.
class BonjourAdvertiser {
    private var listener: NWListener?
    private let role: AgentRole
    private let model: String
    private let port: UInt16
    private let ramMb: Int

    init(role: AgentRole, model: String, port: UInt16, ramMb: Int) {
        self.role = role
        self.model = model
        self.port = port
        self.ramMb = ramMb
    }

    func start() {
        do {
            let params = NWParameters.tcp
            listener = try NWListener(using: params, on: NWEndpoint.Port(rawValue: port)!)
            listener?.service = NWListener.Service(
                name: "REVIVE-\(role.rawValue)-\(deviceName())",
                type: "_revive._tcp",
                domain: nil,
                txtRecord: NWTXTRecord([
                    "role":  role.rawValue,
                    "model": model,
                    "ram":   "\(ramMb)",
                    "port":  "\(port)"
                ])
            )
            listener?.newConnectionHandler = { conn in conn.cancel() } // handled by HTTPServer
            listener?.start(queue: .global())
            print("[Bonjour] Advertising as \(role.displayName) on port \(port)")
        } catch {
            print("[Bonjour] Failed to start advertiser: \(error)")
        }
    }

    func stop() {
        listener?.cancel()
        listener = nil
    }

    private func deviceName() -> String {
        // Use a stable short name derived from device model
        let model = ProcessInfo.processInfo.environment["SIMULATOR_DEVICE_NAME"] ?? UIDevice.current.name
        return model.components(separatedBy: " ").first ?? "device"
    }
}

// MARK: - Bonjour Browser (Coordinator side)

typealias WorkerDiscoveredCallback = @Sendable (WorkerInfo) -> Void
typealias WorkerLostCallback       = @Sendable (String) -> Void  // by bonjour service name

class BonjourBrowser {
    private var browser: NWBrowser?
    private var resolvers: [String: NWConnection] = [:]

    var onWorkerDiscovered: WorkerDiscoveredCallback?
    var onWorkerLost: WorkerLostCallback?

    func start() {
        browser = NWBrowser(for: .bonjour(type: "_revive._tcp", domain: nil), using: .tcp)

        browser?.browseResultsChangedHandler = { [weak self] results, changes in
            guard let self else { return }
            for change in changes {
                switch change {
                case .added(let result):
                    self.resolve(result: result)
                case .removed(let result):
                    if case .service(let name, _, _, _) = result.endpoint {
                        self.onWorkerLost?(name)
                    }
                default:
                    break
                }
            }
        }

        browser?.start(queue: .global())
        print("[Bonjour] Browsing for _revive._tcp services")
    }

    func stop() {
        browser?.cancel()
        browser = nil
    }

    // MARK: - Resolve endpoint to IP:port + metadata

    private func resolve(result: NWBrowser.Result) {
        guard case .service(let name, let type, let domain, _) = result.endpoint else { return }

        // Extract TXT record metadata
        var role: AgentRole = .drafter
        var model = "unknown"
        var ramMb = 2048
        var port: UInt16 = 8080

        if case .bonjour(let txtRecord) = result.metadata {
            if let roleStr = txtRecord.dictionary["role"], let r = AgentRole(rawValue: roleStr) {
                role = r
            }
            if let m = txtRecord.dictionary["model"] { model = m }
            if let r = txtRecord.dictionary["ram"], let ri = Int(r) { ramMb = ri }
            if let p = txtRecord.dictionary["port"], let pi = UInt16(p) { port = pi }
        }

        // Connect briefly to resolve the actual IP address
        let endpoint = NWEndpoint.service(name: name, type: type, domain: domain, interface: nil)
        let conn = NWConnection(to: endpoint, using: .tcp)

        conn.stateUpdateHandler = { [weak self] state in
            switch state {
            case .ready:
                if let path = conn.currentPath,
                   let remoteEndpoint = path.remoteEndpoint,
                   case .hostPort(let host, _) = remoteEndpoint {
                    let hostStr: String
                    if case .name(let n, _) = host {
                        hostStr = n
                    } else {
                        hostStr = "\(host)"
                    }
                    let worker = WorkerInfo(
                        name: name,
                        role: role,
                        model: model,
                        host: hostStr,
                        port: Int(port),
                        ramMb: ramMb
                    )
                    self?.onWorkerDiscovered?(worker)
                }
                conn.cancel()
            case .failed, .cancelled:
                conn.cancel()
            default:
                break
            }
        }
        conn.start(queue: .global())
        resolvers[name] = conn
    }
}
