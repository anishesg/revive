#!/usr/bin/env python3
"""
REVIVE mDNS Advertiser for Android (Termux)
Advertises the llama-server as a _revive._tcp service on the local network
so the iPad coordinator can discover it via Bonjour without manual IP config.

Usage:
  python3 advertise.py --role drafter --port 8080 --model qwen3-0.6b-q4_k_m --ram 3072
"""
import argparse
import signal
import socket
import sys
import time

def get_local_ip():
    """Get the device's WiFi IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()

def main():
    parser = argparse.ArgumentParser(description="REVIVE mDNS advertiser")
    parser.add_argument("--role",  default="drafter",             help="Agent role name")
    parser.add_argument("--port",  type=int, default=8080,        help="llama-server port")
    parser.add_argument("--model", default="qwen3-0.6b-q4_k_m",  help="Model name")
    parser.add_argument("--ram",   type=int, default=3072,        help="Device RAM in MB")
    args = parser.parse_args()

    try:
        from zeroconf import Zeroconf, ServiceInfo
    except ImportError:
        print("[advertise] zeroconf not installed. Run: pip install zeroconf")
        sys.exit(1)

    local_ip = get_local_ip()
    print(f"[advertise] Local IP: {local_ip}")

    service_name = f"REVIVE-{args.role.capitalize()}._revive._tcp.local."

    info = ServiceInfo(
        type_="_revive._tcp.local.",
        name=service_name,
        addresses=[socket.inet_aton(local_ip)],
        port=args.port,
        properties={
            b"role":  args.role.encode(),
            b"model": args.model.encode(),
            b"ram":   str(args.ram).encode(),
            b"port":  str(args.port).encode(),
        },
        server=f"android-{args.role}.local.",
    )

    zc = Zeroconf()
    zc.register_service(info)
    print(f"[advertise] Advertising {service_name} on {local_ip}:{args.port}")
    print(f"[advertise] Role: {args.role} | Model: {args.model} | RAM: {args.ram}MB")
    print("[advertise] Press Ctrl-C to stop.")

    def shutdown(sig, frame):
        print("\n[advertise] Unregistering service...")
        zc.unregister_service(info)
        zc.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Keep alive
    while True:
        time.sleep(5)

if __name__ == "__main__":
    main()
