#include <Arduino.h>
#include <LittleFS.h>
#include <WiFi.h>

// --- 引用 ESP8266Audio 函式庫 ---
// 即使名稱有 ESP8266，它也完美支援 ESP32
#include "AudioFileSourceLittleFS.h"
#include "AudioGeneratorMP3.h"
#include "AudioOutputI2S.h"

// --- WiFi 設定 ---
const char* ssid = "ESP32_S3";   // 熱點名稱
const char* password = "cilab35324";    // 熱點密碼

// --- TCP 服務器設定 ---
const int SERVER_PORT = 8080;      // TCP 服務器端口
WiFiServer server(SERVER_PORT);    // 建立 TCP 服務器
WiFiClient client;                 // 用於與 Python 通訊的客戶端

// --- 硬體接腳 (ESP32-S3) ---
#define I2S_LRC       4
#define I2S_BCLK      5
#define I2S_DIN       6
#define BUTTON_PIN    7   // BOOT 按鈕

// --- 音訊物件指標 ---
AudioGeneratorMP3 *mp3 = NULL;
AudioFileSourceLittleFS *file = NULL;
AudioOutputI2S *out = NULL;
bool lastState = HIGH;

bool isPlaying = false;

void stopPlaying() {
  if (mp3) {
    mp3->stop();
    delete mp3;
    mp3 = NULL;
  }
  if (file) {
    file->close();
    delete file;
    file = NULL;
  }
  isPlaying = false;
}

void setup() {
    Serial.begin(115200);

    // 1. 啟動 AP 模式
    WiFi.mode(WIFI_AP);
    WiFi.softAP(ssid, password);
    
    Serial.println("\n=== ESP32 伺服器已啟動 ===");
    Serial.print("請電腦連線至: "); Serial.println(ssid);
    Serial.print("伺服器 IP: "); Serial.println(WiFi.softAPIP()); // 通常是 192.168.4.1
    Serial.print("通訊埠 (Port): "); Serial.println(SERVER_PORT);

    // 2. 啟動 TCP 伺服器
    server.begin();

    pinMode(BUTTON_PIN, INPUT_PULLUP);
    
    // 啟動檔案系統
    if (!LittleFS.begin()) {
        Serial.println("LittleFS 初始化失敗");
        return;
    }
    
    // 初始化音訊輸出
    out = new AudioOutputI2S(0, AudioOutputI2S::EXTERNAL_I2S);
    out->SetPinout(I2S_BCLK, I2S_LRC, I2S_DIN);
    out->SetGain(3.95); // 音量設定：0.0 ~ 4.0 (0.5 為 50% 音量)
    out->SetOutputModeMono(true);
}

void loop() {


    // --- 1. 處理音樂播放 ---
    if (isPlaying && mp3) {
        if (mp3->isRunning()) {
            if (!mp3->loop()) { 
                // 如果 loop 回傳 false，代表歌曲播完了
                stopPlaying(); 
            }
        } else {
            stopPlaying();
        }
    }
    
    // --- 2. 檢查是否有新的客戶端連接 ---
    if (!client || !client.connected()) {
        client = server.available();
        if (client) {
            Serial.println("✅ Python 已連接");
        }
    }
    
    // --- 3. 處理按鈕 ---
    bool cur = digitalRead(BUTTON_PIN);
    
    // 偵測「按下」(HIGH -> LOW)
    if (cur == LOW && lastState == HIGH) {
        Serial.println("FIRE");
        sendFireEvent();
        
        // 如果正在播，就先停掉
        if (isPlaying) {
            stopPlaying();
            delay(200); // 稍微停頓一下
        }
        
        // 檢查檔案保險
        if (LittleFS.exists("/shut.mp3")) {
            file = new AudioFileSourceLittleFS("/shut.mp3");
            mp3 = new AudioGeneratorMP3();
            mp3->begin(file, out);
            isPlaying = true;
        }
        
        // 等待按鈕放開 (防止連發)
        // 如果您想要按住連發，可以把下面這段 while 註解掉
        while(digitalRead(BUTTON_PIN) == LOW) {
            // 在等待放開的同時，也要繼續播放音樂，不然聲音會卡住！
            if (isPlaying && mp3 && mp3->isRunning()) mp3->loop();
        }
    }
    lastState = cur;
    if(!isPlaying)
        delay(20);
}

void sendFireEvent() {
    if (client && client.connected()) {
        client.println("FIRE");  // 發送訊息給 Python（帶換行符）
        Serial.println("📡 已發送 FIRE 訊息給 Python");
    } else {
        Serial.println("⚠️ Python 未連接！");
    }
}