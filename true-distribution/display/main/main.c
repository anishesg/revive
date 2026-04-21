/*
 * REVIVE dashboard for SenseCAP Indicator.
 * Connects to WiFi, polls http://<HOST>:<PORT>/display, and renders the
 * cluster status on the 4" LVGL screen.
 */

#include <string.h>
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_netif.h"
#include "nvs_flash.h"
#include "esp_http_client.h"
#include "cJSON.h"
#include "lvgl.h"

#include "bsp_board.h"
#include "lv_port.h"

/* ── Config — edit before building ─────────────────────────── */
#define REVIVE_WIFI_SSID   "anish kataria\xe2\x80\x99s iPhone"
#define REVIVE_WIFI_PASS   "aaaaaaaa"
#define REVIVE_HOST        "172.20.10.7"
#define REVIVE_PORT        4100
#define POLL_INTERVAL_MS   1000
#define HTTP_BUF_SIZE      4096

static const char *TAG = "revive";

/* ── WiFi ───────────────────────────────────────────────────── */
static EventGroupHandle_t s_wifi_eg;
#define WIFI_CONNECTED_BIT BIT0

static void wifi_event_handler(void *arg, esp_event_base_t base,
                                int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        ESP_LOGI(TAG, "wifi start, connecting...");
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "wifi disconnected, reconnecting...");
        xEventGroupClearBits(s_wifi_eg, WIFI_CONNECTED_BIT);
        vTaskDelay(pdMS_TO_TICKS(1000));
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *e = (ip_event_got_ip_t *)data;
        ESP_LOGI(TAG, "got ip: " IPSTR, IP2STR(&e->ip_info.ip));
        xEventGroupSetBits(s_wifi_eg, WIFI_CONNECTED_BIT);
    }
}

static void wifi_init(void)
{
    s_wifi_eg = xEventGroupCreate();
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);

    esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                        wifi_event_handler, NULL, NULL);
    esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                        wifi_event_handler, NULL, NULL);

    wifi_config_t wc = { 0 };
    strncpy((char *)wc.sta.ssid, REVIVE_WIFI_SSID, sizeof(wc.sta.ssid) - 1);
    strncpy((char *)wc.sta.password, REVIVE_WIFI_PASS, sizeof(wc.sta.password) - 1);
    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_config(WIFI_IF_STA, &wc);
    esp_wifi_start();

    ESP_LOGI(TAG, "waiting for wifi... (non-blocking, tasks start now)");
}

/* ── Shared state ──────────────────────────────────────────── */
typedef struct {
    char state[16];
    float tps;
    bool busy;
    int layers;
    char phase[16];
    char text[512];
    struct {
        char id[12];
        int s, e;
        bool ok;
    } workers[4];
    int nworkers;
    bool have_data;
} ReviveStatus;

static ReviveStatus g_status = {0};
static SemaphoreHandle_t g_status_mutex;

/* ── LVGL widgets ───────────────────────────────────────────── */
static lv_obj_t *lv_state_label;
static lv_obj_t *lv_state_dot;
static lv_obj_t *lv_tps_label;
static lv_obj_t *lv_busy_label;
static lv_obj_t *lv_phase_label;
static lv_obj_t *lv_text_label;
static lv_obj_t *lv_worker_labels[4];
static lv_obj_t *lv_worker_dots[4];

#define CLR_BG      lv_color_hex(0x0d0d0d)
#define CLR_CYAN    lv_color_hex(0x00e5cc)
#define CLR_GREEN   lv_color_hex(0x22c55e)
#define CLR_YELLOW  lv_color_hex(0xfbbf24)
#define CLR_RED     lv_color_hex(0xef4444)
#define CLR_GRAY    lv_color_hex(0x4b5563)
#define CLR_WHITE   lv_color_hex(0xf9fafb)
#define CLR_DIM     lv_color_hex(0x6b7280)

static void ui_create(void)
{
    lv_obj_t *scr = lv_scr_act();
    lv_obj_set_style_bg_color(scr, CLR_BG, 0);
    lv_obj_set_style_bg_opa(scr, LV_OPA_COVER, 0);

    /* Title */
    lv_obj_t *title = lv_label_create(scr);
    lv_label_set_text(title, "REVIVE");
    lv_obj_set_style_text_font(title, &lv_font_montserrat_32, 0);
    lv_obj_set_style_text_color(title, CLR_CYAN, 0);
    lv_obj_align(title, LV_ALIGN_TOP_LEFT, 20, 18);

    lv_obj_t *sub = lv_label_create(scr);
    lv_label_set_text(sub, "Distributed LLM Inference");
    lv_obj_set_style_text_font(sub, &lv_font_montserrat_14, 0);
    lv_obj_set_style_text_color(sub, CLR_DIM, 0);
    lv_obj_align(sub, LV_ALIGN_TOP_LEFT, 22, 58);

    lv_obj_t *line = lv_obj_create(scr);
    lv_obj_set_size(line, 440, 1);
    lv_obj_set_style_bg_color(line, CLR_GRAY, 0);
    lv_obj_set_style_bg_opa(line, LV_OPA_COVER, 0);
    lv_obj_set_style_border_width(line, 0, 0);
    lv_obj_align(line, LV_ALIGN_TOP_MID, 0, 82);

    /* Cluster state */
    lv_state_dot = lv_obj_create(scr);
    lv_obj_set_size(lv_state_dot, 14, 14);
    lv_obj_set_style_radius(lv_state_dot, LV_RADIUS_CIRCLE, 0);
    lv_obj_set_style_border_width(lv_state_dot, 0, 0);
    lv_obj_set_style_bg_color(lv_state_dot, CLR_GRAY, 0);
    lv_obj_align(lv_state_dot, LV_ALIGN_TOP_LEFT, 20, 100);

    lv_state_label = lv_label_create(scr);
    lv_label_set_text(lv_state_label, "CONNECTING...");
    lv_obj_set_style_text_font(lv_state_label, &lv_font_montserrat_20, 0);
    lv_obj_set_style_text_color(lv_state_label, CLR_DIM, 0);
    lv_obj_align(lv_state_label, LV_ALIGN_TOP_LEFT, 44, 96);

    /* Workers section */
    lv_obj_t *wlbl = lv_label_create(scr);
    lv_label_set_text(wlbl, "WORKERS");
    lv_obj_set_style_text_font(wlbl, &lv_font_montserrat_14, 0);
    lv_obj_set_style_text_color(wlbl, CLR_DIM, 0);
    lv_obj_align(wlbl, LV_ALIGN_TOP_LEFT, 20, 132);

    for (int i = 0; i < 4; i++) {
        int y = 152 + i * 26;

        lv_worker_dots[i] = lv_obj_create(scr);
        lv_obj_set_size(lv_worker_dots[i], 10, 10);
        lv_obj_set_style_radius(lv_worker_dots[i], LV_RADIUS_CIRCLE, 0);
        lv_obj_set_style_border_width(lv_worker_dots[i], 0, 0);
        lv_obj_set_style_bg_color(lv_worker_dots[i], CLR_BG, 0);
        lv_obj_align(lv_worker_dots[i], LV_ALIGN_TOP_LEFT, 20, y + 4);

        lv_worker_labels[i] = lv_label_create(scr);
        lv_label_set_text(lv_worker_labels[i], "");
        lv_obj_set_style_text_font(lv_worker_labels[i], &lv_font_montserrat_16, 0);
        lv_obj_set_style_text_color(lv_worker_labels[i], CLR_GRAY, 0);
        lv_obj_align(lv_worker_labels[i], LV_ALIGN_TOP_LEFT, 40, y);
    }

    /* TPS block */
    lv_tps_label = lv_label_create(scr);
    lv_label_set_text(lv_tps_label, "-- tok/s");
    lv_obj_set_style_text_font(lv_tps_label, &lv_font_montserrat_28, 0);
    lv_obj_set_style_text_color(lv_tps_label, CLR_WHITE, 0);
    lv_obj_align(lv_tps_label, LV_ALIGN_TOP_LEFT, 20, 262);

    lv_busy_label = lv_label_create(scr);
    lv_label_set_text(lv_busy_label, "");
    lv_obj_set_style_text_font(lv_busy_label, &lv_font_montserrat_14, 0);
    lv_obj_set_style_text_color(lv_busy_label, CLR_CYAN, 0);
    lv_obj_align(lv_busy_label, LV_ALIGN_TOP_RIGHT, -20, 270);

    /* Divider 2 */
    lv_obj_t *line2 = lv_obj_create(scr);
    lv_obj_set_size(line2, 440, 1);
    lv_obj_set_style_bg_color(line2, CLR_GRAY, 0);
    lv_obj_set_style_bg_opa(line2, LV_OPA_COVER, 0);
    lv_obj_set_style_border_width(line2, 0, 0);
    lv_obj_align(line2, LV_ALIGN_TOP_MID, 0, 302);

    /* Phase label */
    lv_phase_label = lv_label_create(scr);
    lv_label_set_text(lv_phase_label, "IDLE");
    lv_obj_set_style_text_font(lv_phase_label, &lv_font_montserrat_14, 0);
    lv_obj_set_style_text_color(lv_phase_label, CLR_DIM, 0);
    lv_obj_align(lv_phase_label, LV_ALIGN_TOP_LEFT, 20, 314);

    /* Streaming text area */
    lv_text_label = lv_label_create(scr);
    lv_label_set_text(lv_text_label, "");
    lv_obj_set_style_text_font(lv_text_label, &lv_font_montserrat_16, 0);
    lv_obj_set_style_text_color(lv_text_label, CLR_WHITE, 0);
    lv_label_set_long_mode(lv_text_label, LV_LABEL_LONG_WRAP);
    lv_obj_set_width(lv_text_label, 440);
    lv_obj_set_height(lv_text_label, 134);
    lv_obj_set_style_clip_corner(lv_text_label, true, 0);
    lv_obj_set_style_text_line_space(lv_text_label, 2, 0);
    lv_obj_align(lv_text_label, LV_ALIGN_TOP_LEFT, 20, 336);
}

static void ui_update(const ReviveStatus *s)
{
    lv_color_t sc;
    const char *st_str;
    if (!s->have_data) {
        sc = CLR_GRAY; st_str = "CONNECTING...";
    } else if (strcmp(s->state, "healthy") == 0) {
        sc = CLR_GREEN; st_str = "HEALTHY";
    } else if (strcmp(s->state, "degraded") == 0) {
        sc = CLR_YELLOW; st_str = "DEGRADED";
    } else if (strcmp(s->state, "down") == 0) {
        sc = CLR_RED; st_str = "CLUSTER DOWN";
    } else {
        sc = CLR_GRAY; st_str = "WAITING";
    }
    lv_obj_set_style_bg_color(lv_state_dot, sc, 0);
    lv_label_set_text(lv_state_label, st_str);
    lv_obj_set_style_text_color(lv_state_label, sc, 0);

    for (int i = 0; i < 4; i++) {
        if (i < s->nworkers) {
            char buf[48];
            snprintf(buf, sizeof(buf), "%-10s  [%d - %d)",
                     s->workers[i].id, s->workers[i].s, s->workers[i].e);
            lv_label_set_text(lv_worker_labels[i], buf);
            lv_color_t wc = s->workers[i].ok ? CLR_GREEN : CLR_RED;
            lv_obj_set_style_bg_color(lv_worker_dots[i], wc, 0);
            lv_obj_set_style_text_color(lv_worker_labels[i], wc, 0);
        } else {
            lv_label_set_text(lv_worker_labels[i], "");
            lv_obj_set_style_bg_color(lv_worker_dots[i], CLR_BG, 0);
        }
    }

    char tps_buf[24];
    if (s->have_data && s->tps > 0.0f)
        snprintf(tps_buf, sizeof(tps_buf), "%.1f tok/s", s->tps);
    else
        snprintf(tps_buf, sizeof(tps_buf), "--  tok/s");
    lv_label_set_text(lv_tps_label, tps_buf);

    lv_label_set_text(lv_busy_label, s->busy ? "GENERATING" : "");

    /* Phase label */
    const char *plabel = "IDLE";
    lv_color_t pcolor = CLR_DIM;
    if (strcmp(s->phase, "thinking") == 0) {
        plabel = "THINKING"; pcolor = CLR_CYAN;
    } else if (strcmp(s->phase, "refining") == 0) {
        plabel = "REFINING"; pcolor = CLR_CYAN;
    } else if (strcmp(s->phase, "done") == 0) {
        plabel = "FINAL ANSWER"; pcolor = CLR_GREEN;
    }
    lv_label_set_text(lv_phase_label, plabel);
    lv_obj_set_style_text_color(lv_phase_label, pcolor, 0);

    /* Streaming text — show last ~180 chars as new tokens arrive */
    const char *tail = s->text;
    size_t tlen = strlen(s->text);
    if (tlen > 180) tail = s->text + (tlen - 180);
    lv_label_set_text(lv_text_label, tail);
}

/* ── HTTP poll ──────────────────────────────────────────────── */
static char s_http_buf[HTTP_BUF_SIZE];
static int  s_http_len = 0;

static esp_err_t http_event_handler(esp_http_client_event_t *evt)
{
    if (evt->event_id == HTTP_EVENT_ON_DATA) {
        int rem = HTTP_BUF_SIZE - s_http_len - 1;
        if (rem > 0) {
            int copy = (evt->data_len < rem) ? evt->data_len : rem;
            memcpy(s_http_buf + s_http_len, evt->data, copy);
            s_http_len += copy;
        }
    }
    return ESP_OK;
}

static void parse_and_store(const char *json)
{
    cJSON *root = cJSON_Parse(json);
    if (!root) return;

    ReviveStatus tmp = {0};
    tmp.have_data = true;

    cJSON *st = cJSON_GetObjectItem(root, "st");
    if (st && st->valuestring)
        strncpy(tmp.state, st->valuestring, sizeof(tmp.state) - 1);

    cJSON *tps = cJSON_GetObjectItem(root, "tps");
    if (tps) tmp.tps = (float)tps->valuedouble;

    cJSON *busy = cJSON_GetObjectItem(root, "busy");
    if (busy) tmp.busy = cJSON_IsTrue(busy);

    cJSON *layers = cJSON_GetObjectItem(root, "layers");
    if (layers) tmp.layers = layers->valueint;

    cJSON *phase = cJSON_GetObjectItem(root, "phase");
    if (phase && phase->valuestring)
        strncpy(tmp.phase, phase->valuestring, sizeof(tmp.phase) - 1);

    cJSON *text = cJSON_GetObjectItem(root, "text");
    if (text && text->valuestring)
        strncpy(tmp.text, text->valuestring, sizeof(tmp.text) - 1);

    cJSON *ws = cJSON_GetObjectItem(root, "ws");
    if (cJSON_IsArray(ws)) {
        int n = cJSON_GetArraySize(ws);
        if (n > 4) n = 4;
        tmp.nworkers = n;
        for (int i = 0; i < n; i++) {
            cJSON *w = cJSON_GetArrayItem(ws, i);
            cJSON *wid = cJSON_GetObjectItem(w, "id");
            if (wid && wid->valuestring)
                strncpy(tmp.workers[i].id, wid->valuestring,
                        sizeof(tmp.workers[i].id) - 1);
            cJSON *ss = cJSON_GetObjectItem(w, "s");
            if (ss) tmp.workers[i].s = ss->valueint;
            cJSON *ee = cJSON_GetObjectItem(w, "e");
            if (ee) tmp.workers[i].e = ee->valueint;
            cJSON *ok = cJSON_GetObjectItem(w, "ok");
            tmp.workers[i].ok = ok ? cJSON_IsTrue(ok) : true;
        }
    }

    cJSON_Delete(root);

    xSemaphoreTake(g_status_mutex, portMAX_DELAY);
    memcpy(&g_status, &tmp, sizeof(g_status));
    xSemaphoreGive(g_status_mutex);
}

static void poll_task(void *arg)
{
    char url[96];
    snprintf(url, sizeof(url), "http://%s:%d/display", REVIVE_HOST, REVIVE_PORT);
    ESP_LOGI(TAG, "poll_task started, url=%s", url);

    esp_http_client_config_t cfg = {
        .url           = url,
        .event_handler = http_event_handler,
        .timeout_ms    = 3000,
    };
    esp_http_client_handle_t client = esp_http_client_init(&cfg);
    int ok_count = 0;

    while (1) {
        /* Gate on WiFi */
        if (!(xEventGroupGetBits(s_wifi_eg) & WIFI_CONNECTED_BIT)) {
            vTaskDelay(pdMS_TO_TICKS(500));
            continue;
        }
        s_http_len = 0;
        esp_err_t err = esp_http_client_perform(client);
        if (err == ESP_OK) {
            s_http_buf[s_http_len] = '\0';
            parse_and_store(s_http_buf);
            if ((ok_count++ % 10) == 0)
                ESP_LOGI(TAG, "poll ok, len=%d", s_http_len);
        } else {
            ESP_LOGW(TAG, "http err: %s", esp_err_to_name(err));
        }
        vTaskDelay(pdMS_TO_TICKS(POLL_INTERVAL_MS));
    }
}

static void ui_refresh_task(void *arg)
{
    ReviveStatus snap;
    while (1) {
        xSemaphoreTake(g_status_mutex, portMAX_DELAY);
        memcpy(&snap, &g_status, sizeof(snap));
        xSemaphoreGive(g_status_mutex);

        lv_port_sem_take();
        ui_update(&snap);
        lv_port_sem_give();

        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

/* ── app_main ──────────────────────────────────────────────── */
void app_main(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }

    ESP_ERROR_CHECK(bsp_board_init());
    lv_port_init();

    g_status_mutex = xSemaphoreCreateMutex();

    lv_port_sem_take();
    ui_create();
    lv_port_sem_give();

    wifi_init();

    xTaskCreate(poll_task,        "poll",    4096, NULL, 5, NULL);
    xTaskCreate(ui_refresh_task,  "ui",      4096, NULL, 4, NULL);
}
