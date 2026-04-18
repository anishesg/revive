package com.revive.worker

import android.content.Context
import android.net.nsd.NsdManager
import android.net.nsd.NsdServiceInfo
import android.util.Log

/**
 * Advertises this Android device as a REVIVE worker via mDNS/NSD.
 * Uses Android's built-in Network Service Discovery API.
 */
class NsdAdvertiser(private val context: Context) {
    companion object {
        private const val TAG = "NsdAdvertiser"
        private const val SERVICE_TYPE = "_revive._tcp."
    }

    private var nsdManager: NsdManager? = null
    private var registrationListener: NsdManager.RegistrationListener? = null
    private var isRegistered = false

    fun advertise(role: String, model: String, port: Int, ramMb: Int) {
        val nsd = context.getSystemService(Context.NSD_SERVICE) as NsdManager
        nsdManager = nsd

        val serviceInfo = NsdServiceInfo().apply {
            serviceName = "REVIVE-$role-${android.os.Build.MODEL.replace(" ", "")}"
            serviceType = SERVICE_TYPE
            setPort(port)
            setAttribute("role", role)
            setAttribute("model", model)
            setAttribute("ram", ramMb.toString())
            setAttribute("port", port.toString())
            setAttribute("platform", "android")
            setAttribute("caps", "neon")
        }

        registrationListener = object : NsdManager.RegistrationListener {
            override fun onServiceRegistered(info: NsdServiceInfo) {
                isRegistered = true
                Log.i(TAG, "Service registered: ${info.serviceName}")
            }

            override fun onRegistrationFailed(info: NsdServiceInfo, errorCode: Int) {
                Log.e(TAG, "Registration failed: $errorCode")
            }

            override fun onServiceUnregistered(info: NsdServiceInfo) {
                isRegistered = false
                Log.i(TAG, "Service unregistered")
            }

            override fun onUnregistrationFailed(info: NsdServiceInfo, errorCode: Int) {
                Log.e(TAG, "Unregistration failed: $errorCode")
            }
        }

        nsd.registerService(serviceInfo, NsdManager.PROTOCOL_DNS_SD, registrationListener)
        Log.i(TAG, "Advertising $role on port $port")
    }

    fun stop() {
        if (isRegistered) {
            try {
                nsdManager?.unregisterService(registrationListener)
            } catch (e: Exception) {
                Log.e(TAG, "Error unregistering: ${e.message}")
            }
        }
    }
}
