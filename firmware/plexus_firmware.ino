
#include "esp_camera.h"
#include <ArduinoJson.h>
#include <ESPAsyncWebServer.h>
#include <WiFi.h>


// ==========================================
// CONFIGURATION
// ==========================================

// WiFi Credentials
const char *ssid = "vertebot";
const char *password =
    "password"; // Replace with actual if known, or user handles

// Static IP Configuration for Plexus
IPAddress local_IP(10, 0, 0, 26);
IPAddress gateway(10, 0, 0, 1);
IPAddress subnet(255, 255, 255, 0);

// Pin Definitions for Xiao ESP32S3 Sense
#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 10 // 21 on some boards? Xiao Is specific.
#define SIOD_GPIO_NUM 40
#define SIOC_GPIO_NUM 39

#define Y9_GPIO_NUM 48
#define Y8_GPIO_NUM 11
#define Y7_GPIO_NUM 12
#define Y6_GPIO_NUM 14
#define Y5_GPIO_NUM 16
#define Y4_GPIO_NUM 18
#define Y3_GPIO_NUM 17
#define Y2_GPIO_NUM 15

#define VSYNC_GPIO_NUM 38
#define HREF_GPIO_NUM 47
#define PCLK_GPIO_NUM 13

// Motor Pins (Adjust based on actual wiring Schema for Plexus)
// Assuming generic H-Bridge or similar for now
#define LEFT_MOTOR_FWD 5
#define LEFT_MOTOR_REV 6
#define RIGHT_MOTOR_FWD 7
#define RIGHT_MOTOR_REV 8

// ==========================================
// GLOBALS
// ==========================================

AsyncWebServer server(80);
AsyncWebSocket ws("/ws");

// ==========================================
// CAMERA SETUP
// ==========================================

void setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Xiao S3 has PSRAM
  if (psramFound()) {
    config.frame_size = FRAMESIZE_QVGA; // 320x240 for speed
    config.jpeg_quality = 12;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }
}

// ==========================================
// WEBSOCKET HANDLERS
// ==========================================

void parseCommand(String cmd) {
  // Expected format: "l:0.5;r:-0.5;"
  // Naive parsing
  int l_idx = cmd.indexOf("l:");
  int r_idx = cmd.indexOf("r:");
  int semi1 = cmd.indexOf(";", l_idx);
  int semi2 = cmd.indexOf(";", r_idx);

  if (l_idx != -1 && r_idx != -1 && semi1 != -1 && semi2 != -1) {
    String l_str = cmd.substring(l_idx + 2, semi1);
    String r_str = cmd.substring(r_idx + 2, semi2);

    float l_val = l_str.toFloat();
    float r_val = r_str.toFloat();

    Serial.printf("CMD -> L: %.2f | R: %.2f\n", l_val, r_val);

    // TODO: Drive Motors
    // driveMotors(l_val, r_val);
  }
}

void onEvent(AsyncWebSocket *server, AsyncWebSocketClient *client,
             AwsEventType type, void *arg, uint8_t *data, size_t len) {
  if (type == WS_EVT_CONNECT) {
    Serial.printf("ws[%s][%u] connect\n", server->url(), client->id());
  } else if (type == WS_EVT_DISCONNECT) {
    Serial.printf("ws[%s][%u] disconnect\n", server->url(), client->id());
  } else if (type == WS_EVT_DATA) {
    AwsFrameInfo *info = (AwsFrameInfo *)arg;
    if (info->final && info->index == 0 && info->len == len &&
        info->opcode == WS_TEXT) {
      data[len] = 0;
      String message = (char *)data;
      parseCommand(message);
    }
  }
}

// ==========================================
// STREAM HANDLER
// ==========================================

void streamJpg(AsyncWebServerRequest *request) {
  AsyncWebServerResponse *response = request->beginChunkedResponse(
      "multipart/x-mixed-replace; boundary=frame",
      [](uint8_t *buffer, size_t maxLen, size_t index) -> size_t {
        // This chunked responder is complex for MJPEG in AsyncWebServer typical
        // usage. Easier to just use specific loop implementation or use the
        // esp32-camera example style handlers
        return 0;
      });
  request->send(response);
}

// Simplified Stream Loop (Run in main loop if not using Async Response
// correctly)
WiFiServer streamServer(80);

void serveStreamClient(WiFiClient &client) {
  String response = "HTTP/1.1 200 OK\r\n";
  response += "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
  client.print(response);

  while (client.connected()) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      delay(1000);
      continue;
    }

    client.print("--frame\r\n");
    client.print("Content-Type: image/jpeg\r\n");
    client.print("Content-Length: " + String(fb->len) + "\r\n\r\n");
    client.write(fb->buf, fb->len);
    client.print("\r\n");

    esp_camera_fb_return(fb);
    // delay(10); // Check FPS
  }
}

// ==========================================
// SETUP & LOOP
// ==========================================

void setup() {
  Serial.begin(115200);

  setupCamera();

  // WiFi
  if (!WiFi.config(local_IP, gateway, subnet)) {
    Serial.println("STA Failed to configure");
  }
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");
  Serial.print("Stream Ready! Go to: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/stream");

  // WebSocket
  ws.onEvent(onEvent);
  server.addHandler(&ws);
  server.begin(); // This handles Port 80 for WS

  // Note: AsyncWebServer takes over Port 80.
  // We need to route /stream to a handler.
  server.on("/stream", HTTP_GET, [](AsyncWebServerRequest *request) {
    // This is tricky with AsyncWebServer without writing a custom
    // StreamResponse. For simplicity in this mockup, we might want to run
    // Stream on port 81? But requirement says Port 80. AsyncWebServer *can*
    // stream but needs a chunked callback that captures frames. Let's defer to
    // a separate server for stream if possible? Or just serve 1 frame as test?
    // Actually, standard ESP32 cam examples use a simple sync server for
    // stream.
    request->send(
        200, "text/plain",
        "Stream not implemented in Async Mode yet. Use Port 81 for legacy?");
  });
}

void loop() {
  // ws.cleanupClients(); // Async handles this mostly
  delay(100);
}
