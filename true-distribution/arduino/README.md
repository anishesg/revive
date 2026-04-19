# REVIVE BMC — Arduino Firmware

The Arduino Uno running this firmware is the authoritative cluster
controller. It holds the partition table, watches worker heartbeats,
and declares workers dead when they miss too many. The host (Mac
coordinator) talks to it over USB serial @ 115200 baud.

## Flashing

1. Install the **Arduino IDE** (or `arduino-cli`).
2. Open `revive_bmc.ino` in the IDE.
3. Plug the Arduino Uno into your Mac via USB.
4. **Tools → Board → Arduino Uno**
5. **Tools → Port → /dev/cu.usbmodem...**  (the one that appears when you plug in)
6. Click **Upload**. Should finish in ~5s.
7. Open **Tools → Serial Monitor** at 115200 baud. You should see
   `READY 1` after the board resets. Close the Serial Monitor — it holds
   an exclusive lock on the port and will block the dashboard from
   connecting.

That's the entire one-time setup. No external components needed.

## Running the dashboard against the real Arduino

Two ways:

```bash
# Auto-detect (scans USB serial ports for anything that looks like an Arduino)
SERIAL=auto scripts/launch_cluster.sh

# Or point at a specific device
SERIAL=/dev/tty.usbmodem14101 scripts/launch_cluster.sh
```

The dashboard header will show **BMC: ARDUINO** in amber when it's
talking to real hardware, or **BMC: SIMULATOR** when it's using the
Python stand-in.

## What the built-in LED (pin 13) means

- **Slow pulse (~1 Hz)** — cluster healthy, nothing generating
- **Fast blink (~5 Hz)** — cluster healthy, inference in progress
- **Medium blink (~2 Hz)** — cluster degraded (some worker dead)
- **Panic blink (~5 Hz)** — no workers registered, cluster down

## Verifying the protocol by hand

With the Serial Monitor open at 115200 baud (line-ending: Newline), type:

```
HELLO                 → READY 1
MODEL 28              → ACK MODEL 28 / STATE down
REG w1 150 4096       → ACK REG w1 / STATE healthy / PARTITION w1:0:28
REG w2 100 2048       → ACK REG w2 / PARTITION w1:0:17 w2:17:28
HB w1 250 42          → (silent — heartbeat resets watchdog)
FAIL w1               → DEAD w1 / STATE degraded / PARTITION w2:0:28
HB w1 10 40           → ALIVE w1 / STATE healthy / PARTITION w1:0:17 w2:17:28
```

If the Arduino responds correctly to this by hand, the dashboard will
work against it out of the box.

## Swapping sim ↔ hardware

The `bmc_sim.py` and `revive_bmc.ino` implement the same logic. The
controller in `pipeline/controller.py` doesn't know which one it's
talking to — it sees an identical line protocol. You can run half a
demo on the simulator, hot-swap in the Arduino, and neither the
workers nor the dashboard need to be restarted (though the dashboard
itself does need a `--serial-device` argument at start, so in practice
you restart the dashboard process, not the workers).
