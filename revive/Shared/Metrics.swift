import Foundation
import UIKit

/// Collects real-time device telemetry.
struct DeviceMetrics {

    static var thermalStateString: String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal:  return "nominal"
        case .fair:     return "fair"
        case .serious:  return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    static var batteryPercent: Int {
        UIDevice.current.isBatteryMonitoringEnabled = true
        let level = UIDevice.current.batteryLevel
        if level < 0 { return -1 }
        return Int(level * 100)
    }

    static var memoryUsedMb: Int {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        if result == KERN_SUCCESS {
            return Int(info.resident_size) / (1024 * 1024)
        }
        return 0
    }

    static var availableMemoryMb: Int {
        let total = ProcessInfo.processInfo.physicalMemory
        let used = memoryUsedMb * 1024 * 1024
        return Int(total - UInt64(used)) / (1024 * 1024)
    }

    static func snapshot(tokensGenerated: Int, tokensPerSecond: Double,
                         timeToFirstTokenMs: Int, totalTimeMs: Int) -> ResponseMetrics {
        ResponseMetrics(
            tokensGenerated: tokensGenerated,
            tokensPerSecond: tokensPerSecond,
            timeToFirstTokenMs: timeToFirstTokenMs,
            totalTimeMs: totalTimeMs,
            thermalState: thermalStateString,
            batteryPercent: batteryPercent,
            memoryUsedMb: memoryUsedMb
        )
    }
}
