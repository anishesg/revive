idf_component_register(
    SRCS "main.c" "lv_port.c"
    INCLUDE_DIRS "."
    REQUIRES esp_wifi esp_netif esp_event nvs_flash esp_http_client lwip json lvgl bsp
)
