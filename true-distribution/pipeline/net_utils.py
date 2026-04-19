"""Tiny utilities for discovering the host's LAN-visible IP.

On a Mac with multiple interfaces (Wi-Fi, USB ethernet, Thunderbolt bridge,
utun VPNs) `socket.gethostname()` or `gethostbyname(gethostname())` often
returns a loopback or wrong-interface IP. The reliable trick: open a UDP
socket to a non-local address; the kernel picks the interface it would
route through, and `getsockname()` gives us that source IP. No packets
are actually sent."""
from __future__ import annotations
import socket


def get_lan_ip(fallback: str = "127.0.0.1") -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Non-routable public-ish target; no packet leaves the machine.
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass
    # Last resort: try all non-loopback addrs associated with hostname
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                return ip
    except Exception:
        pass
    return fallback
