#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include "config.h"

#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// === Sensor & Hardware Definitions ===
#define DHTPIN 14
#define DHTTYPE DHT22
#define LDR_PIN 33

DHT dht(DHTPIN, DHTTYPE);

// OLED Display
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);  // -1 = no reset pin

// WiFi credentials
const char *ssid = wifi_name;
const char *password = wifi_pass;

// Backend server
const char *serverBase = IP;

// === I/O Pins ===
int soilPins[3] = {34, 35, 32};
int relayPins[3] = {25, 26, 27};

// === Timing ===
unsigned long previousMillis = 0;
const unsigned long interval = 1000;

void setup()
{
  Serial.begin(115200);

  // Initialize I2C
  Wire.begin();

  // Initialize OLED
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C))
  {
    Serial.println(F("❌ SSD1306 allocation failed"));
    while (true);
  }
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("OLED Initialized");
  display.display();
  delay(1000);

  dht.begin();

  // Setup relay pins
  for (int i = 0; i < 3; i++)
  {
    pinMode(relayPins[i], OUTPUT);
    digitalWrite(relayPins[i], HIGH);
  }

  // Connect to WiFi
  Serial.print("Connecting to WiFi");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ WiFi Connected!");
}

void loop()
{
  float temperature = 0.0;
  float humidity = 0.0;
  float light = 0.0;
  float soilMoisture[3];

  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= interval)
  {
    previousMillis = currentMillis;

    // === Read DHT22 ===
    temperature = dht.readTemperature();
    humidity = dht.readHumidity();

    if (isnan(temperature) || isnan(humidity))
    {
      Serial.println("⚠️  DHT read failed, skipping POST");
      return;
    }

    Serial.printf("🌡️  Temperature: %.2f °C\n", temperature);
    Serial.printf("💧 Humidity: %.2f %%\n", humidity);

    // === LDR Reading ===
    int ldrRaw = analogRead(LDR_PIN);
    if (ldrRaw < 0 || ldrRaw > 4095)
    {
      Serial.println("⚠️ Invalid LDR reading.");
      return;
    }
    light = map(ldrRaw, 0, 4095, 100, 0);
    Serial.printf("🌞 LDR Light: %.2f %%\n", light);

    // === Soil Moisture ===
    for (int i = 0; i < 3; i++)
    {
      int raw = analogRead(soilPins[i]);
      soilMoisture[i] = map(raw, 4095, 0, 0, 100);
    }

    // === Update OLED Display ===
    display.clearDisplay();
    display.setCursor(0, 0);
    display.setTextSize(1);
    display.printf("Temp: %.1f C\n", temperature);
    display.printf("Hum : %.1f %%\n", humidity);
    display.printf("Light: %.1f %%\n", light);
    for (int i = 0; i < 3; i++)
    {
      display.printf("Soil%d: %.1f %%\n", i + 1, soilMoisture[i]);
    }
    display.display();

    // === Prepare JSON Payload ===
    StaticJsonDocument<256> doc;
    doc["temperature"] = temperature;
    doc["humidity"] = humidity;
    doc["light"] = light;

    String payload;
    serializeJson(doc, payload);

    // === Send Sensor Data & Check Thresholds ===
    if (WiFi.status() == WL_CONNECTED)
    {
      HTTPClient http;

      // POST sensor data
      http.begin(String(serverBase) + "/sensor/");
      http.addHeader("Content-Type", "application/json");
      int code = http.POST(payload);
      Serial.print("📡 Sensor POST response: ");
      Serial.println(code);
      Serial.println(http.getString());
      http.end();

      // GET thresholds
      http.begin(String(serverBase) + "/thresholds/");
      int thrCode = http.GET();
      if (thrCode == 200)
      {
        String t = http.getString();
        StaticJsonDocument<128> tdoc;
        deserializeJson(tdoc, t);

        for (int i = 0; i < 3; i++)
        {
          float threshold = tdoc[i];
          float moisture = soilMoisture[i];

          Serial.printf("🌱 Plant %d | Soil: %.2f%% | Threshold: %.2f%%\n", i + 1, moisture, threshold);

          if (moisture < threshold)
          {
            digitalWrite(relayPins[i], LOW); // ON
            Serial.printf("💧 Watering plant %d...\n", i + 1);
          }
          else
          {
            digitalWrite(relayPins[i], HIGH); // OFF
            Serial.printf("✅ Moisture OK for plant %d. Relay OFF.\n", i + 1);
          }
        }
      }
      else
      {
        Serial.printf("❌ Failed to GET thresholds, code: %d\n", thrCode);
      }

      http.end();
    }
  }

  delay(1000); // Short delay before next cycle
}
