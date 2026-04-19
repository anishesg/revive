import Foundation
import Network
import llama

/// HTTP server implementing the pipeline-worker wire protocol spoken by the
/// Python coordinator in true-distribution/pipeline/protocol.py.
///
/// Endpoints:
///   POST /forward    — binary frame: [4B length][UTF-8 JSON header][raw tensor bytes]
///                      First-stage: tensor is int32 token ids (shape [T]).
///                      Mid/last:     tensor is fp16 hidden state [1,T,H].
///                      Response: same frame format. Non-last returns hidden;
///                      last returns empty tensor + token_id in header.
///   POST /reset      — JSON {"seq_id": "..."}  → drops KV for that seq
///   GET  /info       — JSON stage metadata (matches Python worker)
///   GET  /health     — JSON {"status":"ok"}
final class PipelineWorkerServer: @unchecked Sendable {
    private var listener: NWListener?
    private let port: UInt16
    private let stage: PipelineStage
    private let queue = DispatchQueue(label: "revive.pipeline-server", qos: .userInitiated)
    /// Called on any listener state change. View model watches this to surface
    /// issues like "waiting (missing local network permission)" to the user.
    var onListenerState: ((String) -> Void)?
    /// Fired after each /forward completes. `tps` is this forward's tok/s,
    /// `tokens` is the count of tokens processed. View model watches this to
    /// show live activity.
    var onForwardComplete: ((_ tokens: Int, _ tps: Double, _ kind: String) -> Void)?

    init(port: UInt16, stage: PipelineStage) {
        self.port = port
        self.stage = stage
    }

    func start() throws {
        let params = NWParameters.tcp
        params.acceptLocalOnly = false
        params.allowLocalEndpointReuse = true
        guard let nwPort = NWEndpoint.Port(rawValue: port) else {
            throw PipelineError.decodeFailed(rc: -1)
        }
        let lst = try NWListener(using: params, on: nwPort)
        // Advertise via Bonjour — triggers iOS local-network permission prompt
        // if the user hasn't granted it, and gives the coordinator a way to
        // auto-discover us.
        lst.service = NWListener.Service(name: "revive-stage-\(port)", type: "_revive-pp._tcp")
        lst.stateUpdateHandler = { [weak self] state in
            let s: String
            switch state {
            case .setup:     s = "setup"
            case .waiting(let e): s = "waiting (\(e)) — check iOS Settings > Privacy > Local Network"
            case .ready:     s = "ready"
            case .failed(let e): s = "failed: \(e)"
            case .cancelled: s = "cancelled"
            @unknown default: s = "unknown"
            }
            print("[PipelineWorker] listener: \(s)")
            self?.onListenerState?(s)
        }
        lst.newConnectionHandler = { [weak self] conn in
            guard let self else { return }
            print("[PipelineWorker] accepted connection")
            self.handleConnection(conn)
        }
        self.listener = lst
        lst.start(queue: queue)
        print("[PipelineWorker] starting on :\(port)")
    }

    func stop() {
        listener?.cancel()
        listener = nil
    }

    // MARK: - Connection handling

    private func handleConnection(_ conn: NWConnection) {
        conn.start(queue: .global(qos: .userInitiated))
        // Accumulate bytes until we've read the full Content-Length body,
        // then dispatch. Buffer strategy: read chunks, scan for end of
        // headers, then read exactly content-length more.
        readRequest(conn: conn, buffer: Data())
    }

    private func readRequest(conn: NWConnection, buffer: Data) {
        conn.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) { [weak self] chunk, _, isComplete, error in
            guard let self else { return }
            var newBuf = buffer
            if let chunk = chunk, !chunk.isEmpty {
                newBuf.append(chunk)
            }
            // Find header/body boundary
            if let hb = findHeaderBoundary(newBuf) {
                let headerBytes = newBuf.subdata(in: 0..<hb)
                guard let headerStr = String(data: headerBytes, encoding: .utf8),
                      let (method, path, contentLength) = parseRequestLine(headerStr) else {
                    self.send(conn: conn, status: 400, contentType: "application/json",
                              body: Data("{\"error\":\"bad request line\"}".utf8))
                    return
                }
                let haveBody = newBuf.count - (hb + 4)
                if haveBody >= contentLength {
                    let body = newBuf.subdata(in: (hb + 4)..<(hb + 4 + contentLength))
                    Task { await self.dispatch(conn: conn, method: method, path: path, body: body) }
                    return
                }
                // Need more bytes
                if isComplete {
                    self.send(conn: conn, status: 400, contentType: "application/json",
                              body: Data("{\"error\":\"premature eof\"}".utf8))
                    return
                }
                self.readRequest(conn: conn, buffer: newBuf)
                return
            }
            if isComplete {
                self.send(conn: conn, status: 400, contentType: "application/json",
                          body: Data("{\"error\":\"no headers\"}".utf8))
                return
            }
            if error != nil {
                conn.cancel()
                return
            }
            self.readRequest(conn: conn, buffer: newBuf)
        }
    }

    // MARK: - Dispatch

    private func dispatch(conn: NWConnection, method: String, path: String, body: Data) {
        // Stage access needs actor hop; always go through a Task.
        Task { [weak self] in
            guard let self else { return }
            await self.dispatchAsync(conn: conn, method: method, path: path, body: body)
        }
    }

    private func dispatchAsync(conn: NWConnection, method: String, path: String, body: Data) async {
        switch (method, path) {
        case ("GET", "/health"):
            send(conn: conn, status: 200, contentType: "application/json",
                 body: Data("{\"status\":\"ok\"}".utf8))

        case ("GET", "/info"):
            let info = await (
                modelPath: stage.modelPath,
                layerStart: stage.layerStart,
                layerEnd: stage.layerEnd,
                numLayersTotal: stage.numLayersTotal,
                isFirst: stage.isFirst,
                isLast: stage.isLast,
                hiddenSize: stage.hiddenSize
            )
            let json: [String: Any] = [
                "model": info.modelPath,
                "layer_start": info.layerStart,
                "layer_end": info.layerEnd,
                "num_layers_total": info.numLayersTotal,
                "is_first": info.isFirst,
                "is_last": info.isLast,
                "hidden_size": info.hiddenSize,
                "device": "iOS",
                "dtype": "fp16",
                "active_seqs": 0,
            ]
            if let data = try? JSONSerialization.data(withJSONObject: json) {
                send(conn: conn, status: 200, contentType: "application/json", body: data)
            } else {
                send(conn: conn, status: 500, contentType: "application/json", body: Data())
            }

        case ("POST", "/reset"):
            struct ResetReq: Decodable { let seq_id: String }
            if let r = try? JSONDecoder().decode(ResetReq.self, from: body) {
                await stage.reset(seqId: r.seq_id)
                send(conn: conn, status: 200, contentType: "application/json",
                     body: Data("{\"ok\":true}".utf8))
            } else {
                send(conn: conn, status: 400, contentType: "application/json",
                     body: Data("{\"error\":\"bad json\"}".utf8))
            }

        case ("POST", "/forward"):
            await handleForward(conn: conn, body: body)

        default:
            send(conn: conn, status: 404, contentType: "application/json",
                 body: Data("{\"error\":\"not found\"}".utf8))
        }
    }

    // MARK: - /forward

    private func handleForward(conn: NWConnection, body: Data) async {
        // Frame format: [4B BE header_len][JSON header][tensor bytes]
        guard body.count >= 4 else {
            send(conn: conn, status: 400, contentType: "application/octet-stream", body: Data())
            return
        }
        let hlen = Int(body[0]) << 24 | Int(body[1]) << 16 | Int(body[2]) << 8 | Int(body[3])
        guard hlen >= 0 && 4 + hlen <= body.count else {
            send(conn: conn, status: 400, contentType: "application/octet-stream", body: Data())
            return
        }
        let headerData = body.subdata(in: 4..<(4 + hlen))
        let tensorData = body.subdata(in: (4 + hlen)..<body.count)
        guard let hdr = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any],
              let seqId = hdr["seq_id"] as? String,
              let dtype = hdr["dtype"] as? String,
              let positionsAny = hdr["positions"] as? [Int] else {
            send(conn: conn, status: 400, contentType: "application/octet-stream", body: Data())
            return
        }

        let positions = positionsAny.map { Int32($0) }
        // One actor hop to read immutable metadata.
        let meta = await (
            isFirst: stage.isFirst,
            isLast: stage.isLast,
            hiddenSize: stage.hiddenSize
        )

        do {
            let result: PipelineStage.ForwardResult
            if meta.isFirst {
                guard dtype == "int32" else { throw PipelineError.shapeMismatch(expected: 4, got: 0) }
                var tokens = [llama_token]()
                tokens.reserveCapacity(positions.count)
                tensorData.withUnsafeBytes { raw in
                    let p = raw.bindMemory(to: Int32.self)
                    for i in 0..<positions.count { tokens.append(p[i]) }
                }
                result = try await stage.forward(seqId: seqId, input: .init(
                    tokens: tokens, hidden: nil, positions: positions, overrideSeqPosition: true))
            } else {
                let n = positions.count * Int(meta.hiddenSize)
                var floats = Data(count: n * MemoryLayout<Float>.stride)
                if dtype == "float16" {
                    guard tensorData.count == n * 2 else {
                        throw PipelineError.shapeMismatch(expected: n * 2, got: tensorData.count)
                    }
                    tensorData.withUnsafeBytes { src in
                        floats.withUnsafeMutableBytes { dst in
                            let s = src.bindMemory(to: UInt16.self)
                            let d = dst.bindMemory(to: Float.self)
                            for i in 0..<n { d[i] = float16BitsToFloat(s[i]) }
                        }
                    }
                } else if dtype == "float32" {
                    guard tensorData.count == n * 4 else {
                        throw PipelineError.shapeMismatch(expected: n * 4, got: tensorData.count)
                    }
                    floats = tensorData
                } else {
                    throw PipelineError.shapeMismatch(expected: 0, got: 0)
                }
                result = try await stage.forward(seqId: seqId, input: .init(
                    tokens: nil, hidden: floats, positions: positions, overrideSeqPosition: true))
            }

            let responseHeader: [String: Any]
            var tensorOut = Data()
            if meta.isLast {
                let logits = result.logits ?? []
                let tok = argmax(logits)
                responseHeader = [
                    "seq_id": seqId,
                    "shape": [] as [Int],
                    "dtype": "int32",
                    "token_id": tok,
                    "eos": false,
                    "latency_ms": result.latencyMs,
                    "tokens_per_second": Double(result.nTokens) * 1000.0 / max(result.latencyMs, 0.001),
                ]
            } else {
                let hiddenF32 = result.hidden ?? Data()
                let nEl = hiddenF32.count / MemoryLayout<Float>.stride
                var fp16 = Data(count: nEl * 2)
                hiddenF32.withUnsafeBytes { src in
                    fp16.withUnsafeMutableBytes { dst in
                        let s = src.bindMemory(to: Float.self)
                        let d = dst.bindMemory(to: UInt16.self)
                        for i in 0..<nEl { d[i] = floatToFloat16Bits(s[i]) }
                    }
                }
                tensorOut = fp16
                responseHeader = [
                    "seq_id": seqId,
                    "shape": [1, Int(result.nTokens), Int(meta.hiddenSize)],
                    "dtype": "float16",
                    "token_id": NSNull(),
                    "eos": false,
                    "latency_ms": result.latencyMs,
                    "tokens_per_second": Double(result.nTokens) * 1000.0 / max(result.latencyMs, 0.001),
                ]
            }

            guard let hbytes = try? JSONSerialization.data(withJSONObject: responseHeader) else {
                send(conn: conn, status: 500, contentType: "application/octet-stream", body: Data())
                return
            }
            var frame = Data()
            var l = UInt32(hbytes.count).bigEndian
            withUnsafeBytes(of: &l) { raw in frame.append(raw.bindMemory(to: UInt8.self)) }
            frame.append(hbytes)
            frame.append(tensorOut)
            send(conn: conn, status: 200, contentType: "application/octet-stream", body: frame)

            // Tick UI counter so the phone visibly reacts to forwards.
            let tps = Double(result.nTokens) * 1000.0 / max(result.latencyMs, 0.001)
            let kind = meta.isLast ? "last" : (meta.isFirst ? "first" : "mid")
            onForwardComplete?(Int(result.nTokens), tps, kind)
        } catch {
            send(conn: conn, status: 500, contentType: "application/json",
                 body: Data("{\"error\":\"forward failed: \(error)\"}".utf8))
        }
    }

    // MARK: - HTTP response

    private func send(conn: NWConnection, status: Int, contentType: String, body: Data) {
        let statusText: String = {
            switch status { case 200: return "OK"; case 400: return "Bad Request"; case 404: return "Not Found"; default: return "Error" }
        }()
        let headers = """
        HTTP/1.1 \(status) \(statusText)\r
        Content-Type: \(contentType)\r
        Content-Length: \(body.count)\r
        Connection: close\r
        \r

        """
        var resp = Data(headers.utf8)
        resp.append(body)
        conn.send(content: resp, completion: .contentProcessed { _ in conn.cancel() })
    }
}

// MARK: - Helpers

private func argmax(_ v: [Float]) -> Int32 {
    var best: Int32 = 0
    var bestVal: Float = -.infinity
    for i in 0..<v.count { if v[i] > bestVal { bestVal = v[i]; best = Int32(i) } }
    return best
}

private func findHeaderBoundary(_ data: Data) -> Int? {
    // \r\n\r\n
    let needle: [UInt8] = [13, 10, 13, 10]
    if data.count < 4 { return nil }
    for i in 0...(data.count - 4) {
        if data[i] == needle[0] && data[i+1] == needle[1] && data[i+2] == needle[2] && data[i+3] == needle[3] {
            return i
        }
    }
    return nil
}

private func parseRequestLine(_ headers: String) -> (String, String, Int)? {
    let lines = headers.components(separatedBy: "\r\n")
    guard let first = lines.first else { return nil }
    let parts = first.components(separatedBy: " ")
    guard parts.count >= 2 else { return nil }
    var cl = 0
    for line in lines.dropFirst() {
        let lc = line.lowercased()
        if lc.hasPrefix("content-length:") {
            let v = line.split(separator: ":", maxSplits: 1).dropFirst().first.map {
                $0.trimmingCharacters(in: .whitespaces)
            } ?? ""
            cl = Int(v) ?? 0
        }
    }
    return (parts[0], parts[1], cl)
}

private func float16BitsToFloat(_ h: UInt16) -> Float {
    // IEEE 754 half → float. Uses software conversion (Swift has Float16 on
    // iOS 14+, but this is portable and bit-accurate).
    let s = UInt32(h & 0x8000) << 16
    let e = Int32((h >> 10) & 0x1f)
    let m = UInt32(h & 0x3ff)
    if e == 0 {
        if m == 0 {
            return Float(bitPattern: s)
        } else {
            var mm = m
            var ee = Int32(-14)
            while (mm & 0x400) == 0 { mm <<= 1; ee -= 1 }
            mm &= 0x3ff
            let bits = s | UInt32(Int32(127) + ee) << 23 | (mm << 13)
            return Float(bitPattern: bits)
        }
    } else if e == 31 {
        return Float(bitPattern: s | 0x7f800000 | (m << 13))
    } else {
        return Float(bitPattern: s | UInt32(e - 15 + 127) << 23 | (m << 13))
    }
}

private func floatToFloat16Bits(_ f: Float) -> UInt16 {
    let bits = f.bitPattern
    let s = UInt16((bits >> 16) & 0x8000)
    var e = Int32((bits >> 23) & 0xff) - 127 + 15
    var m = bits & 0x007fffff
    if e >= 31 {                                  // overflow → inf
        return s | 0x7c00
    } else if e <= 0 {                            // subnormal / underflow
        if e < -10 { return s }                   // too small
        m = (m | 0x00800000) >> UInt32(1 - e)
        // round
        if (m & 0x00001000) != 0 { m += 0x00002000 }
        return s | UInt16(m >> 13)
    } else {
        if (m & 0x00001000) != 0 {
            m += 0x00002000
            if (m & 0x00800000) != 0 { m = 0; e += 1 }
        }
        if e >= 31 { return s | 0x7c00 }
        return s | UInt16(e << 10) | UInt16(m >> 13)
    }
}
