import Foundation
import Network

/// A minimal HTTP/1.1 server backed by Network.framework.
/// Handles POST /v1/chat/completions and GET /health.
actor HTTPServer {
    private var listener: NWListener?
    private let port: UInt16
    private let handler: @Sendable (CompletionRequest) async -> (String, ResponseMetrics)

    init(port: UInt16, handler: @escaping @Sendable (CompletionRequest) async -> (String, ResponseMetrics)) {
        self.port = port
        self.handler = handler
    }

    func start() throws {
        let params = NWParameters.tcp
        params.acceptLocalOnly = false

        listener = try NWListener(using: params, on: NWEndpoint.Port(rawValue: port)!)
        listener?.newConnectionHandler = { [weak self] conn in
            guard let self else { return }
            Task { await self.handleConnection(conn) }
        }
        listener?.start(queue: .global(qos: .userInitiated))
        print("[HTTPServer] Listening on port \(port)")
    }

    func stop() {
        listener?.cancel()
        listener = nil
    }

    // MARK: - Connection handling

    private func handleConnection(_ conn: NWConnection) {
        conn.start(queue: .global(qos: .userInitiated))
        receiveRequest(conn: conn)
    }

    private func receiveRequest(conn: NWConnection) {
        conn.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, isComplete, error in
            guard let self, let data, !data.isEmpty else {
                conn.cancel()
                return
            }

            Task {
                await self.processHTTPData(data, conn: conn)
            }
        }
    }

    private func processHTTPData(_ data: Data, conn: NWConnection) async {
        guard let raw = String(data: data, encoding: .utf8) else {
            send(conn: conn, status: 400, body: "{\"error\":\"bad request\"}")
            return
        }

        // Split headers from body
        let parts = raw.components(separatedBy: "\r\n\r\n")
        let headerSection = parts.first ?? ""
        let body = parts.dropFirst().joined(separator: "\r\n\r\n")

        let requestLine = headerSection.components(separatedBy: "\r\n").first ?? ""
        let method = requestLine.components(separatedBy: " ").first ?? ""
        let path = requestLine.components(separatedBy: " ").dropFirst().first ?? ""

        if method == "GET" && path == "/health" {
            send(conn: conn, status: 200, body: "{\"status\":\"ok\"}")
            return
        }

        if method == "POST" && path == "/v1/chat/completions" {
            guard let bodyData = body.data(using: .utf8),
                  let request = try? JSONDecoder().decode(CompletionRequest.self, from: bodyData) else {
                send(conn: conn, status: 400, body: "{\"error\":\"invalid JSON\"}")
                return
            }

            let (content, metrics) = await handler(request)
            let response = buildCompletionResponse(content: content, metrics: metrics)
            send(conn: conn, status: 200, body: response)
            return
        }

        send(conn: conn, status: 404, body: "{\"error\":\"not found\"}")
    }

    // MARK: - Response helpers

    private func buildCompletionResponse(content: String, metrics: ResponseMetrics) -> String {
        let metricsJSON = (try? String(data: JSONEncoder().encode(metrics), encoding: .utf8)) ?? "{}"
        let escaped = content
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\r", with: "\\r")
        return """
        {
          "choices": [{"message": {"role": "assistant", "content": "\(escaped)"}}],
          "metrics": \(metricsJSON)
        }
        """
    }

    private func send(conn: NWConnection, status: Int, body: String) {
        let bodyData = body.data(using: .utf8) ?? Data()
        let headers = """
        HTTP/1.1 \(status) \(statusText(status))\r
        Content-Type: application/json\r
        Content-Length: \(bodyData.count)\r
        Access-Control-Allow-Origin: *\r
        Connection: close\r
        \r

        """
        var responseData = headers.data(using: .utf8)!
        responseData.append(bodyData)
        conn.send(content: responseData, completion: .contentProcessed { _ in
            conn.cancel()
        })
    }

    private func statusText(_ code: Int) -> String {
        switch code {
        case 200: return "OK"
        case 400: return "Bad Request"
        case 404: return "Not Found"
        default:  return "Error"
        }
    }
}
