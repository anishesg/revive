package com.revive.worker

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat

/**
 * Foreground service that keeps the worker running when the app is backgrounded.
 * Manages the LLM engine, HTTP server, and mDNS advertisement.
 */
class WorkerService : Service() {
    companion object {
        private const val TAG = "WorkerService"
        private const val CHANNEL_ID = "revive_worker"
        private const val NOTIFICATION_ID = 1

        const val EXTRA_MODEL_PATH = "model_path"
        const val EXTRA_ROLE = "role"
        const val EXTRA_PORT = "port"
    }

    private var engine: LlamaEngine? = null
    private var httpServer: ReviveHttpServer? = null
    private var nsdAdvertiser: NsdAdvertiser? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val modelPath = intent?.getStringExtra(EXTRA_MODEL_PATH) ?: return START_NOT_STICKY
        val role = intent.getStringExtra(EXTRA_ROLE) ?: "drafter"
        val port = intent.getIntExtra(EXTRA_PORT, 8080)

        startForeground(NOTIFICATION_ID, buildNotification(role))

        Thread {
            try {
                val llama = LlamaEngine()
                if (llama.load(modelPath)) {
                    engine = llama
                    Log.i(TAG, "Model loaded: $modelPath")

                    httpServer = ReviveHttpServer(this, llama, role, port).also { it.start() }

                    val ramMb = DeviceMetrics.totalRamMb(this)
                    nsdAdvertiser = NsdAdvertiser(this).also {
                        it.advertise(role, llama.getModelName(), port, ramMb)
                    }

                    Log.i(TAG, "Worker running: role=$role port=$port")
                } else {
                    Log.e(TAG, "Failed to load model")
                    stopSelf()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Worker start failed", e)
                stopSelf()
            }
        }.start()

        return START_STICKY
    }

    override fun onDestroy() {
        nsdAdvertiser?.stop()
        httpServer?.stop()
        engine?.release()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID, "REVIVE Worker", NotificationManager.IMPORTANCE_LOW,
            ).apply { description = "Keeps the REVIVE inference worker running" }
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
    }

    private fun buildNotification(role: String): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("REVIVE Worker")
            .setContentText("Running as $role")
            .setSmallIcon(android.R.drawable.ic_menu_manage)
            .setOngoing(true)
            .build()
    }
}
