"""
REVIVE mDNS Discovery Module
Discovers and tracks worker nodes on the local network via Bonjour/mDNS.
Also advertises this node as a _revive._tcp service.
"""
import asyncio
import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Callable

from zeroconf import IPVersion, ServiceBrowser, ServiceInfo, ServiceStateChange, Zeroconf

log = logging.getLogger("revive.discovery")

SERVICE_TYPE = "_revive._tcp.local."


@dataclass
class WorkerNode:
    name: str
    role: str
    model: str
    host: str
    port: int
    ram_mb: int
    platform: str
    caps: str
    status: str = "idle"
    last_seen: float = field(default_factory=time.time)
    last_tps: float = 0.0
    last_thermal: str = "nominal"
    last_battery: int = -1
    last_memory_mb: int = 0

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def weight(self) -> float:
        speed = min(self.last_tps / 40.0, 1.0)
        thermal_penalty = {"nominal": 1.0, "fair": 1.0, "serious": 0.5, "critical": 0.1}.get(
            self.last_thermal, 1.0
        )
        return speed * thermal_penalty

    @property
    def age_seconds(self) -> float:
        return time.time() - self.last_seen


class SwarmDiscovery:
    """Discovers REVIVE workers via mDNS and maintains a live worker registry."""

    def __init__(
        self,
        on_worker_added: Callable[[WorkerNode], None] | None = None,
        on_worker_removed: Callable[[str], None] | None = None,
    ):
        self.workers: dict[str, WorkerNode] = {}
        self._on_added = on_worker_added
        self._on_removed = on_worker_removed
        self._zeroconf: Zeroconf | None = None
        self._browser: ServiceBrowser | None = None

    def start(self) -> None:
        self._zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
        self._browser = ServiceBrowser(
            self._zeroconf, SERVICE_TYPE, handlers=[self._on_state_change]
        )
        log.info("Browsing for %s services", SERVICE_TYPE)

    def stop(self) -> None:
        if self._browser:
            self._browser.cancel()
        if self._zeroconf:
            self._zeroconf.close()

    def get_workers_by_role(self, role: str) -> list[WorkerNode]:
        return [w for w in self.workers.values() if w.role == role and w.status != "offline"]

    def get_all_workers(self) -> list[WorkerNode]:
        return [w for w in self.workers.values() if w.status != "offline"]

    def add_manual_worker(self, host: str, port: int, role: str, model: str, platform: str = "unknown", ram_mb: int = 2048) -> WorkerNode:
        name = f"manual-{role}-{host}:{port}"
        node = WorkerNode(
            name=name, role=role, model=model, host=host, port=port,
            ram_mb=ram_mb, platform=platform, caps="cpu",
        )
        self.workers[name] = node
        log.info("Manual worker added: %s @ %s:%d", role, host, port)
        if self._on_added:
            self._on_added(node)
        return node

    def mark_offline(self, name: str) -> None:
        if name in self.workers:
            self.workers[name].status = "offline"

    def update_metrics(self, name: str, tps: float, thermal: str, battery: int, memory_mb: int) -> None:
        if name in self.workers:
            w = self.workers[name]
            w.last_tps = tps
            w.last_thermal = thermal
            w.last_battery = battery
            w.last_memory_mb = memory_mb
            w.last_seen = time.time()
            w.status = "idle"

    def _on_state_change(self, zeroconf: Zeroconf, service_type: str, name: str, state_change: ServiceStateChange) -> None:
        if state_change == ServiceStateChange.Added:
            info = zeroconf.get_service_info(service_type, name)
            if info:
                self._register_worker(name, info)
        elif state_change == ServiceStateChange.Removed:
            short_name = name.replace(f".{SERVICE_TYPE}", "")
            if short_name in self.workers:
                del self.workers[short_name]
                log.info("Worker left: %s", short_name)
                if self._on_removed:
                    self._on_removed(short_name)

    def _register_worker(self, full_name: str, info: ServiceInfo) -> None:
        short_name = full_name.replace(f".{SERVICE_TYPE}", "")
        addresses = info.parsed_addresses(IPVersion.V4Only)
        if not addresses:
            return

        host = addresses[0]
        props = {k.decode(): v.decode() if isinstance(v, bytes) else str(v) for k, v in info.properties.items()}

        node = WorkerNode(
            name=short_name,
            role=props.get("role", "drafter"),
            model=props.get("model", "unknown"),
            host=host,
            port=int(props.get("port", str(info.port))),
            ram_mb=int(props.get("ram", "2048")),
            platform=props.get("platform", "unknown"),
            caps=props.get("caps", "cpu"),
        )
        self.workers[short_name] = node
        log.info("Worker discovered: %s (%s) @ %s:%d [%s]", short_name, node.role, host, node.port, node.platform)
        if self._on_added:
            self._on_added(node)


class ServiceAdvertiser:
    """Advertises this node as a REVIVE service on the local network."""

    def __init__(self, role: str, model: str, port: int, platform: str = "rpi", ram_mb: int = 4096, caps: str = "neon"):
        self.role = role
        self.model = model
        self.port = port
        self.platform = platform
        self.ram_mb = ram_mb
        self.caps = caps
        self._zeroconf: Zeroconf | None = None
        self._info: ServiceInfo | None = None

    def start(self) -> None:
        local_ip = self._get_local_ip()
        self._zeroconf = Zeroconf(ip_version=IPVersion.V4Only)

        self._info = ServiceInfo(
            type_=SERVICE_TYPE,
            name=f"REVIVE-{self.role}-{socket.gethostname()}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(local_ip)],
            port=self.port,
            properties={
                "role": self.role,
                "model": self.model,
                "ram": str(self.ram_mb),
                "port": str(self.port),
                "platform": self.platform,
                "caps": self.caps,
            },
            server=f"{socket.gethostname()}.local.",
        )
        self._zeroconf.register_service(self._info)
        log.info("Advertising %s on %s:%d", self.role, local_ip, self.port)

    def stop(self) -> None:
        if self._zeroconf and self._info:
            self._zeroconf.unregister_service(self._info)
            self._zeroconf.close()

    @staticmethod
    def _get_local_ip() -> str:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
        finally:
            s.close()
