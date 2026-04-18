import Foundation
import Network

/// Embeds a small HTTP server on port 8080 so any browser on the LAN can view
/// the live swarm dashboard at http://<iPad-IP>:8080
final class CoordinatorWebServer {
    private var listener: NWListener?
    private let swarmManager: SwarmManager
    private let port: UInt16 = 8080

    // Holds the query/mode state for the REST API
    private(set) var currentMode: SwarmMode = .swarm
    var onQuery: ((String, SwarmMode) async -> Void)?

    init(swarmManager: SwarmManager) {
        self.swarmManager = swarmManager
    }

    func start() {
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        guard let listener = try? NWListener(using: params, on: NWEndpoint.Port(rawValue: port)!) else {
            print("[WebServer] Failed to create listener on port \(port)")
            return
        }
        self.listener = listener
        listener.newConnectionHandler = { [weak self] conn in
            self?.handleConnection(conn)
        }
        listener.start(queue: .global(qos: .utility))
        print("[WebServer] Dashboard available at http://localhost:\(port)")
    }

    func stop() {
        listener?.cancel()
    }

    // MARK: - Connection handling

    private func handleConnection(_ conn: NWConnection) {
        conn.start(queue: .global(qos: .utility))
        receive(conn)
    }

    private func receive(_ conn: NWConnection) {
        conn.receive(minimumIncompleteLength: 1, maximumLength: 16384) { [weak self] data, _, isComplete, error in
            guard let self, let data, !data.isEmpty else {
                if isComplete { conn.cancel() }
                return
            }
            self.processRequest(data: data, conn: conn)
        }
    }

    private func processRequest(data: Data, conn: NWConnection) {
        guard let raw = String(data: data, encoding: .utf8) else {
            conn.cancel(); return
        }

        let lines = raw.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else { conn.cancel(); return }
        let parts = requestLine.components(separatedBy: " ")
        guard parts.count >= 2 else { conn.cancel(); return }

        let method = parts[0]
        let path   = parts[1].components(separatedBy: "?")[0]  // strip query string

        // Separate headers and body
        let bodyStart = raw.range(of: "\r\n\r\n").map { raw.index($0.upperBound, offsetBy: 0) }
        let bodyString = bodyStart.map { String(raw[$0...]) } ?? ""

        switch (method, path) {
        case ("GET", "/"), ("GET", "/index.html"):
            serveFile(name: "index.html", mimeType: "text/html", conn: conn)
        case ("GET", "/style.css"):
            serveFile(name: "style.css", mimeType: "text/css", conn: conn)
        case ("GET", "/dashboard.js"):
            serveFile(name: "dashboard.js", mimeType: "application/javascript", conn: conn)
        case ("GET", "/manifest.json"):
            serveManifest(conn: conn)
        case ("GET", "/api/state"):
            serveState(conn: conn)
        case ("POST", "/api/query"):
            handleQueryPost(body: bodyString, conn: conn)
        case ("POST", "/api/mode"):
            handleModePost(body: bodyString, conn: conn)
        default:
            send404(conn: conn)
        }
    }

    // MARK: - Route handlers

    private func serveFile(name: String, mimeType: String, conn: NWConnection) {
        // Look for the file next to the binary (copied by Xcode build phase)
        let bundle = Bundle.main
        let nameWithoutExt = (name as NSString).deletingPathExtension
        let ext = (name as NSString).pathExtension

        if let url = bundle.url(forResource: nameWithoutExt, withExtension: ext),
           let data = try? Data(contentsOf: url) {
            sendResponse(status: "200 OK", mimeType: mimeType, body: data, conn: conn)
        } else {
            send404(conn: conn)
        }
    }

    private func serveManifest(conn: NWConnection) {
        let manifest = """
        {
          "name": "REVIVE Swarm",
          "short_name": "REVIVE",
          "display": "standalone",
          "background_color": "#0a0a0a",
          "theme_color": "#00ff88",
          "icons": []
        }
        """
        sendResponse(status: "200 OK", mimeType: "application/json",
                     body: manifest.data(using: .utf8)!, conn: conn)
    }

    @MainActor
    private func buildStatePayload() -> Data {
        struct WorkerDTO: Encodable {
            let id: String
            let role: String
            let model: String
            let host: String
            let port: Int
            let status: String
            let tps: Double
            let weight: Double
            let color: String
        }

        struct StateDTO: Encodable {
            let workers: [WorkerDTO]
            let messages: [MessageDTO]
            let mode: String
            let isQuerying: Bool
            let totalQueries: Int
        }

        struct MessageDTO: Encodable {
            let role: String
            let content: String
            let color: String?
            let model: String?
            let tps: Double?
        }

        let workers = swarmManager.workers.map { w in
            WorkerDTO(
                id: w.id.uuidString,
                role: w.role.rawValue,
                model: w.model,
                host: w.host,
                port: w.port,
                status: w.status.rawValue,
                tps: w.lastMetrics.tokensPerSecond,
                weight: w.weight,
                color: w.role.color
            )
        }

        // We don't store messages centrally here — the dashboard polls and
        // the iOS UI drives messages. For the web dashboard we expose workers
        // and status; chat is shown in the iOS app.
        let dto = StateDTO(
            workers: workers,
            messages: [],
            mode: currentMode == .swarm ? "swarm" : "speed",
            isQuerying: swarmManager.isQuerying,
            totalQueries: 0
        )

        return (try? JSONEncoder().encode(dto)) ?? Data()
    }

    private func serveState(conn: NWConnection) {
        Task { @MainActor in
            let body = self.buildStatePayload()
            self.sendResponse(status: "200 OK", mimeType: "application/json", body: body, conn: conn)
        }
    }

    private func handleQueryPost(body: String, conn: NWConnection) {
        struct QueryBody: Decodable { let query: String; let mode: String? }
        if let data = body.data(using: .utf8),
           let qb = try? JSONDecoder().decode(QueryBody.self, from: data) {
            let mode: SwarmMode = qb.mode == "speed" ? .speed : .swarm
            Task {
                await self.onQuery?(qb.query, mode)
            }
        }
        sendOK(conn: conn)
    }

    private func handleModePost(body: String, conn: NWConnection) {
        struct ModeBody: Decodable { let mode: String }
        if let data = body.data(using: .utf8),
           let mb = try? JSONDecoder().decode(ModeBody.self, from: data) {
            currentMode = mb.mode == "speed" ? .speed : .swarm
        }
        sendOK(conn: conn)
    }

    // MARK: - HTTP helpers

    private func sendResponse(status: String, mimeType: String, body: Data, conn: NWConnection) {
        let header = """
        HTTP/1.1 \(status)\r
        Content-Type: \(mimeType); charset=utf-8\r
        Content-Length: \(body.count)\r
        Access-Control-Allow-Origin: *\r
        Connection: close\r
        \r

        """.data(using: .utf8)!

        var response = header
        response.append(body)

        conn.send(content: response, completion: .contentProcessed { _ in conn.cancel() })
    }

    private func send404(conn: NWConnection) {
        sendResponse(status: "404 Not Found", mimeType: "text/plain",
                     body: "Not found".data(using: .utf8)!, conn: conn)
    }

    private func sendOK(conn: NWConnection) {
        sendResponse(status: "200 OK", mimeType: "application/json",
                     body: #"{"ok":true}"#.data(using: .utf8)!, conn: conn)
    }
}
