package com.revive.worker

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager

object DeviceMetrics {
    fun batteryPercent(context: Context): Int {
        val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        return bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
    }

    fun thermalState(): String {
        return try {
            val temp = java.io.File("/sys/class/thermal/thermal_zone0/temp")
            if (temp.exists()) {
                val celsius = temp.readText().trim().toInt() / 1000.0
                when {
                    celsius < 45 -> "nominal"
                    celsius < 55 -> "fair"
                    celsius < 65 -> "serious"
                    else -> "critical"
                }
            } else "unknown"
        } catch (e: Exception) {
            "unknown"
        }
    }

    fun memoryUsedMb(): Int {
        val runtime = Runtime.getRuntime()
        val used = runtime.totalMemory() - runtime.freeMemory()
        return (used / (1024 * 1024)).toInt()
    }

    fun totalRamMb(context: Context): Int {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        val memInfo = android.app.ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        return (memInfo.totalMem / (1024 * 1024)).toInt()
    }

    fun snapshot(
        context: Context,
        tokensGenerated: Int = 0,
        tokensPerSecond: Double = 0.0,
        timeToFirstTokenMs: Int = 0,
        totalTimeMs: Int = 0,
    ): Map<String, Any> = mapOf(
        "tokens_generated" to tokensGenerated,
        "tokens_per_second" to tokensPerSecond,
        "time_to_first_token_ms" to timeToFirstTokenMs,
        "total_time_ms" to totalTimeMs,
        "thermal_state" to thermalState(),
        "battery_percent" to batteryPercent(context),
        "memory_used_mb" to memoryUsedMb(),
    )
}
