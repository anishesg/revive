// REVIVE BMC — Baseboard Management Controller firmware for Arduino Uno.
//
// Runs the distributed LLM cluster's control plane. Holds the authoritative
// partition assignment. Detects worker failures via heartbeat timeout. If
// this chip dies, the cluster can't re-partition. ~$10 of silicon managing
// a cluster of old phones.
//
// Protocol is line-oriented ASCII. See pipeline/bmc_protocol.py for spec.
// Flash with Arduino IDE, then connect Mac coordinator via USB serial at
// 115200 baud.

#include <Arduino.h>

#define MAX_WORKERS          6
#define MAX_ID_LEN           9     // 8 chars + null
#define LINE_BUF             128
#define HB_TIMEOUT_MS        6000  // ~2 missed @ 3s cadence
#define MAX_LAYERS           64
#define FW_VERSION           3     // bumped for FAN command
#define COORD_URL_LEN        80    // "http://255.255.255.255:65535/worker" + slack

// Fan output — 8-bit PWM on D3. Drive a MOSFET / NPN / Grove Transistor
// gate from this pin (fan's own power is 5V+GND; the transistor switches
// its ground path). Unmodulated direct-to-5V wiring will just stay on —
// the pin still produces a meaningful signal that the host reads back.
#define FAN_PIN              3
#define FAN_SAFETY_FLOOR     0     // set >0 to force minimum airflow
#define FAN_TICK_MS          1000  // how often to emit STATUS FAN

struct Worker {
  char id[MAX_ID_LEN];
  uint16_t score_x100;   // capability score × 100
  uint16_t ram_mb;
  uint32_t last_hb_ms;
  uint16_t last_tps_x10;
  int8_t   last_temp_c;
  uint8_t  start_layer;
  uint8_t  end_layer;
  bool     alive;
  bool     ever_seen;
};

Worker workers[MAX_WORKERS];
uint8_t num_workers = 0;
uint16_t num_layers = 0;
char line_buf[LINE_BUF];
uint8_t line_pos = 0;
uint32_t last_health_check_ms = 0;
bool infer_active = false;
char cluster_state[12] = "down";  // down, degraded, healthy
char coord_url[COORD_URL_LEN] = ""; // coordinator discovery URL (empty until host pushes)

// Fan control state. The host computes the duty from load/temperature/HA
// signals and pushes it via `FAN <0-255>`. Firmware holds the most recent
// value, floors it at FAN_SAFETY_FLOOR, and writes it to the PWM pin.
// `fan_last_host_ms` tracks staleness so a silent host doesn't leave the
// fan stuck at full-tilt forever.
uint8_t  fan_duty = 0;
uint32_t fan_last_host_ms = 0;
uint32_t fan_last_tick_ms = 0;
const uint32_t FAN_HOST_STALE_MS = 15000; // if host silent this long, fall back to autonomous

// ─── helpers ──────────────────────────────────────────────────────────────

int8_t find_worker(const char* id) {
  for (uint8_t i = 0; i < num_workers; i++) {
    if (strcmp(workers[i].id, id) == 0) return i;
  }
  return -1;
}

void log_info(const char* msg) {
  Serial.print(F("INFO "));
  Serial.println(msg);
}

void set_state(const char* new_state) {
  if (strcmp(cluster_state, new_state) == 0) return;
  strncpy(cluster_state, new_state, sizeof(cluster_state) - 1);
  Serial.print(F("STATE "));
  Serial.println(cluster_state);
}

// Greedy proportional partitioner — same algorithm as Python reference.
// Runs in O(num_workers) time, uses only stack locals.
void repartition() {
  // Count alive
  uint8_t n_alive = 0;
  uint32_t total_score = 0;
  for (uint8_t i = 0; i < num_workers; i++) {
    if (workers[i].alive) {
      n_alive++;
      total_score += workers[i].score_x100;
    }
  }

  if (n_alive == 0 || num_layers == 0) {
    set_state("down");
    return;
  }
  set_state(n_alive < num_workers ? "degraded" : "healthy");

  // Assign contiguous ranges, fast-to-slow doesn't matter; we just walk in
  // registration order.
  uint8_t cur = 0;
  uint8_t remaining_workers = n_alive;
  for (uint8_t i = 0; i < num_workers; i++) {
    if (!workers[i].alive) {
      workers[i].start_layer = 0;
      workers[i].end_layer = 0;
      continue;
    }
    uint8_t end;
    if (remaining_workers == 1) {
      end = num_layers;  // last one gets the remainder
    } else {
      // share = num_layers * score / total_score
      uint32_t share = ((uint32_t)num_layers * workers[i].score_x100) / total_score;
      if (share < 1) share = 1;
      end = cur + share;
      // leave at least 1 layer for each remaining worker
      uint8_t max_end = num_layers - (remaining_workers - 1);
      if (end > max_end) end = max_end;
    }
    workers[i].start_layer = cur;
    workers[i].end_layer = end;
    cur = end;
    remaining_workers--;
  }

  // Emit PARTITION
  Serial.print(F("PARTITION"));
  for (uint8_t i = 0; i < num_workers; i++) {
    if (!workers[i].alive) continue;
    Serial.print(' ');
    Serial.print(workers[i].id);
    Serial.print(':');
    Serial.print(workers[i].start_layer);
    Serial.print(':');
    Serial.print(workers[i].end_layer);
  }
  Serial.println();
}

// ─── command handlers ─────────────────────────────────────────────────────

void handle_reg(char* rest) {
  char* id = strtok(rest, " ");
  char* score_s = strtok(NULL, " ");
  char* ram_s = strtok(NULL, " ");
  if (!id || !score_s || !ram_s) { log_info("bad REG"); return; }

  int8_t idx = find_worker(id);
  bool is_new = (idx < 0);
  if (is_new) {
    if (num_workers >= MAX_WORKERS) { log_info("worker table full"); return; }
    idx = num_workers++;
    strncpy(workers[idx].id, id, MAX_ID_LEN - 1);
    workers[idx].id[MAX_ID_LEN - 1] = 0;
  }
  workers[idx].score_x100 = atoi(score_s);
  workers[idx].ram_mb = atoi(ram_s);
  workers[idx].last_hb_ms = millis();
  workers[idx].alive = true;
  workers[idx].ever_seen = true;

  if (is_new || num_layers > 0) repartition();
  Serial.print(F("ACK REG "));
  Serial.println(id);
}

void handle_unreg(char* rest) {
  char* id = strtok(rest, " ");
  if (!id) return;
  int8_t idx = find_worker(id);
  if (idx < 0) return;
  workers[idx].alive = false;
  repartition();
  Serial.print(F("ACK UNREG "));
  Serial.println(id);
}

void handle_hb(char* rest) {
  char* id = strtok(rest, " ");
  char* tps_s = strtok(NULL, " ");
  char* temp_s = strtok(NULL, " ");
  if (!id) return;
  int8_t idx = find_worker(id);
  if (idx < 0) return;

  bool was_dead = !workers[idx].alive;
  workers[idx].last_hb_ms = millis();
  workers[idx].alive = true;
  if (tps_s)  workers[idx].last_tps_x10 = atoi(tps_s);
  if (temp_s) workers[idx].last_temp_c  = atoi(temp_s);

  if (was_dead) {
    Serial.print(F("ALIVE "));
    Serial.println(id);
    repartition();
  }
}

void handle_model(char* rest) {
  num_layers = atoi(rest);
  if (num_layers > MAX_LAYERS) num_layers = MAX_LAYERS;
  if (num_workers > 0) repartition();
  Serial.print(F("ACK MODEL "));
  Serial.println(num_layers);
}

void handle_infer(char* rest) {
  infer_active = (strncmp(rest, "START", 5) == 0);
  Serial.print(F("ACK INFER "));
  Serial.println(infer_active ? "START" : "END");
}

void handle_fail(char* rest) {
  char* id = strtok(rest, " ");
  if (!id) return;
  int8_t idx = find_worker(id);
  if (idx < 0) return;
  workers[idx].alive = false;
  Serial.print(F("DEAD "));
  Serial.println(id);
  repartition();
}

void handle_query() {
  Serial.print(F("STATE "));
  Serial.println(cluster_state);
  if (num_workers > 0 && num_layers > 0) repartition();
}

void handle_reset() {
  num_workers = 0;
  num_layers = 0;
  infer_active = false;
  set_state("down");
  log_info("cluster state cleared");
}

void handle_coordinator(char* rest) {
  if (!rest) return;
  // Skip any leading space
  while (*rest == ' ') rest++;
  strncpy(coord_url, rest, COORD_URL_LEN - 1);
  coord_url[COORD_URL_LEN - 1] = 0;
  Serial.print(F("ACK COORDINATOR "));
  Serial.println(coord_url);
}

void handle_discover() {
  Serial.print(F("COORDINATOR "));
  Serial.println(coord_url[0] ? coord_url : "unknown");
}

// ─── fan ───────────────────────────────────────────────────────────────────

void apply_fan_duty(uint8_t duty) {
  if (duty < FAN_SAFETY_FLOOR) duty = FAN_SAFETY_FLOOR;
  fan_duty = duty;
  analogWrite(FAN_PIN, fan_duty);
}

void handle_fan(char* rest) {
  if (!rest) return;
  while (*rest == ' ') rest++;
  int v = atoi(rest);
  if (v < 0) v = 0;
  if (v > 255) v = 255;
  apply_fan_duty((uint8_t)v);
  fan_last_host_ms = millis();
  Serial.print(F("ACK FAN "));
  Serial.println(fan_duty);
}

// Autonomous fallback: if the host hasn't sent a FAN command for a while,
// run a simple policy based solely on BMC state so the fan still tracks
// load roughly (infer -> spin; degraded -> boost; down -> off + floor).
uint8_t autonomous_fan_duty() {
  if (strcmp(cluster_state, "down") == 0) return 0;
  if (strcmp(cluster_state, "degraded") == 0) return 200;
  if (infer_active) return 160;
  return 40;  // healthy + idle — gentle airflow
}

void fan_tick() {
  uint32_t now = millis();
  if (now - fan_last_tick_ms < FAN_TICK_MS) return;
  fan_last_tick_ms = now;

  bool host_stale = (fan_last_host_ms == 0) ||
                    (now - fan_last_host_ms > FAN_HOST_STALE_MS);
  if (host_stale) apply_fan_duty(autonomous_fan_duty());

  Serial.print(F("STATUS FAN "));
  Serial.print(fan_duty);
  Serial.print(F(" src="));
  Serial.println(host_stale ? F("auto") : F("host"));
}

// ─── dispatch ──────────────────────────────────────────────────────────────

void process_line(char* line) {
  char* cmd = strtok(line, " ");
  char* rest = strtok(NULL, "");
  if (!cmd) return;
  if      (!strcmp(cmd, "HELLO"))  { Serial.print(F("READY ")); Serial.println(FW_VERSION); }
  else if (!strcmp(cmd, "REG"))    handle_reg(rest ? rest : (char*)"");
  else if (!strcmp(cmd, "UNREG"))  handle_unreg(rest ? rest : (char*)"");
  else if (!strcmp(cmd, "HB"))     handle_hb(rest ? rest : (char*)"");
  else if (!strcmp(cmd, "MODEL"))  handle_model(rest ? rest : (char*)"");
  else if (!strcmp(cmd, "INFER"))  handle_infer(rest ? rest : (char*)"");
  else if (!strcmp(cmd, "FAIL"))   handle_fail(rest ? rest : (char*)"");
  else if (!strcmp(cmd, "QUERY"))        handle_query();
  else if (!strcmp(cmd, "RESET"))        handle_reset();
  else if (!strcmp(cmd, "COORDINATOR"))  handle_coordinator(rest);
  else if (!strcmp(cmd, "DISCOVER"))     handle_discover();
  else if (!strcmp(cmd, "FAN"))          handle_fan(rest);
  else { Serial.print(F("INFO unknown cmd ")); Serial.println(cmd); }
}

// Built-in LED on pin 13: slow blink when idle, fast blink when inference
// active, SOS if cluster is down.
void update_led() {
  static uint32_t last_blink_ms = 0;
  static bool led_on = false;
  uint32_t now = millis();
  uint16_t period = 1000;
  if (strcmp(cluster_state, "down") == 0) period = 200;      // fast blink = bad
  else if (infer_active) period = 100;                        // faster blink = working
  else if (strcmp(cluster_state, "degraded") == 0) period = 500;
  if (now - last_blink_ms >= period) {
    last_blink_ms = now;
    led_on = !led_on;
    digitalWrite(LED_BUILTIN, led_on ? HIGH : LOW);
  }
}

void health_check() {
  uint32_t now = millis();
  if (now - last_health_check_ms < 1000) return;
  last_health_check_ms = now;
  bool any_change = false;
  for (uint8_t i = 0; i < num_workers; i++) {
    if (workers[i].alive && (now - workers[i].last_hb_ms > HB_TIMEOUT_MS)) {
      workers[i].alive = false;
      Serial.print(F("DEAD "));
      Serial.println(workers[i].id);
      any_change = true;
    }
  }
  if (any_change) repartition();
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(FAN_PIN, OUTPUT);
  apply_fan_duty(0);
  set_state("down");
  Serial.print(F("READY "));
  Serial.println(FW_VERSION);
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (line_pos > 0) {
        line_buf[line_pos] = 0;
        process_line(line_buf);
        line_pos = 0;
      }
    } else if (line_pos < LINE_BUF - 1) {
      line_buf[line_pos++] = c;
    }
  }
  health_check();
  update_led();
  fan_tick();
}
