
#include "esp_wifi.h"
#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>


// Forward Declaration
void start_saturation_attack();
bool analyze_sensor_data(int *buffer, int size);

static int sensor_buffer[100];
static const int BUFFER_SIZE = 100;
volatile bool attack_running = false;

// Task to monitor sensor during attack
void sensor_monitor_task(void *pvParameters) {
  int pin = (int)pvParameters;

  // Wait for attack start
  while (!attack_running) {
    vTaskDelay(10 / portTICK_PERIOD_MS);
  }

  // Sample during saturation
  for (int i = 0; i < BUFFER_SIZE; i++) {
    // High speed sampling
    sensor_buffer[i] = analogRead(pin);
    // Busy wait or small delay
    ets_delay_us(2000); // 500Hz
  }

  vTaskDelete(NULL);
}

void start_saturation_attack() {
  WiFiUDP udp;
  // Send to broadcast or host
  udp.beginPacket("255.255.255.255", 8888);

  uint8_t payload[1024];
  memset(payload, 0xAA, 1024);

  attack_running = true;

  // Flood for 2 seconds
  unsigned long start = millis();
  while (millis() - start < 2000) {
    udp.write(payload, 1024);
    // Force immediate send logic if possible in Arduino, usually endPacket
    // flushes But for saturation we want continuous streaming In raw ESP-IDF
    // we'd use esp_wifi_internal_tx Here we loop endPacket which triggers the
    // flush
    udp.endPacket();
    udp.beginPacket("255.255.255.255", 8888);
  }

  attack_running = false;
}

bool run_hardware_verification(int sensor_pin) {
  Serial.println("VERIFY: Starting Secure Reflex Test...");

  // 1. Setup WiFi in Sta mode to enable Radio
  WiFi.begin("Nimbar_Host", "password");
  // We don't need to connect, just enable radio TX

  // 2. Launch Monitor Task on Core 1
  xTaskCreatePinnedToCore(sensor_monitor_task, "SensorMonitor", 4096,
                          (void *)sensor_pin, 1, NULL,
                          1 // Core 1
  );

  // 3. Launch Attack (runs on current Core, likely 1 in Arduino loop, so maybe
  // pin monitor to 0?) Arduino loop usually runs on Core 1 on S3. Let's pin
  // monitor to Core 0 to be distinct from WiFi driver (usually Core 0/1
  // dynamic). Actually WiFi Interrupts are prioritized.

  start_saturation_attack();

  // 4. Analyze
  bool valid = analyze_sensor_data(sensor_buffer, BUFFER_SIZE);
  return valid;
}

bool analyze_sensor_data(int *buffer, int size) {
  int dropouts = 0;
  long sum = 0;

  for (int i = 0; i < size; i++) {
    if (buffer[i] == 0 || buffer[i] == 4095) {
      dropouts++;
    }
    sum += buffer[i];
  }

  float avg = sum / (float)size;

  // If ADC2 is conflicted (Attack successful on Clone), we expect garbage or
  // dropouts. If ADC1 (Genuine), we expect stable readings (assuming sensor is
  // connected).

  if (dropouts > size * 0.1) {
    return false; // Too many dropouts -> Clone utilizing ADC2 likely
  }

  // Variance check could be added
  return true;
}
