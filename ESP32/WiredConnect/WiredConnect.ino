#include <Arduino.h>
#include <LittleFS.h>

// --- 引用 ESP8266Audio 函式庫 ---
// 即使名稱有 ESP8266，它也完美支援 ESP32
#include "AudioFileSourceLittleFS.h"
#include "AudioGeneratorMP3.h"
#include "AudioOutputI2S.h"

// --- 硬體接腳 (ESP32-S3) ---
#define I2S_LRC       4
#define I2S_BCLK      5
#define I2S_DIN       6
#define BUTTON_PIN    42   

// --- 音訊物件指標 ---
AudioGeneratorMP3 *mp3 = NULL;
AudioFileSourceLittleFS *file = NULL;
AudioOutputI2S *out = NULL;
bool lastState = HIGH;

bool isPlaying = false;

// --- 播放結束或停止時的清理函式 ---
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
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  // 1. 啟動檔案系統
  if (!LittleFS.begin()) {
    return;
  }
  
  out = new AudioOutputI2S();
  out->SetPinout(I2S_BCLK, I2S_LRC, I2S_DIN);
  out->SetGain(3.95); // 音量設定：0.0 ~ 4.0 (0.5 為 50% 音量)
  
  // Serial.println("按下按鈕開始播放...");
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
  bool cur = digitalRead(BUTTON_PIN);
  // --- 2. 處理按鈕 ---
  if (cur == LOW && lastState == HIGH) {
      Serial.println("FIRE");
      // 如果正在播，就先停掉
      if (isPlaying) {
        stopPlaying();
        delay(200); // 稍微停頓一下
      }

      // Serial.println("🔫 播放音效...");
      
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
  delay(20);
}