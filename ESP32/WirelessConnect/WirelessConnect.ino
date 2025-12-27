#include <WiFi.h>

// WiFi 設定
const char* ssid = "LYL";          // 請修改為你的 WiFi SSID
const char* password = "29744073";      // 請修改為你的 WiFi 密碼

// TCP 服務器設定
const int SERVER_PORT = 8080;      // TCP 服務器端口
WiFiServer server(SERVER_PORT);    // 建立 TCP 服務器
WiFiClient client;                 // 用於與 Python 通訊的客戶端

// 按鈕設定
const int BTN_PIN = 4;          // 按鈕接的腳位
bool lastState = HIGH;
unsigned long lastFireMs = 0;
const unsigned long debounceMs = 30;

void setup() {
    IPAddress local_IP(192, 168, 1, 200);      // ESP32 的固定 IP
    IPAddress gateway(192, 168, 1, 1);         // 你的路由器閘道
    IPAddress subnet(255, 255, 255, 0);        // 子網路遮罩
    IPAddress primaryDNS(8, 8, 8, 8);          // Google DNS（選用）
    IPAddress secondaryDNS(8, 8, 4, 4);        // Google DNS（選用）
    
    if (!WiFi.config(local_IP, gateway, subnet, primaryDNS, secondaryDNS)) {
        Serial.println("靜態 IP 設定失敗");
    }
    Serial.begin(115200);
    pinMode(BTN_PIN, INPUT_PULLUP);
    
    // 連接 WiFi
    Serial.println();
    Serial.print("連接到 WiFi: ");
    Serial.println(ssid);
    
    WiFi.begin(ssid, password);
    
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    
    Serial.println();
    Serial.println("WiFi 已連接！");
    Serial.print("IP 地址: ");
    Serial.println(WiFi.localIP());
    
    // 啟動 TCP 服務器
    server.begin();
    Serial.print("TCP 服務器已啟動，監聽端口: ");
    Serial.println(SERVER_PORT);
    Serial.println("等待 Python 連接...");
}

void loop() {
    // 檢查是否有新的客戶端連接
    if (!client || !client.connected()) {
        client = server.available();
        if (client) {
            Serial.println("✅ Python 已連接");
        }
    }
    
    // 讀取按鈕狀態
    bool cur = digitalRead(BTN_PIN);

    // 偵測「按下」(HIGH -> LOW)
    if (lastState == HIGH && cur == LOW) {
        unsigned long now = millis();
        if (now - lastFireMs > debounceMs) {
            Serial.println("🔥 按鈕被按下");
            sendFireEvent();
            lastFireMs = now;
        }
    }
    lastState = cur;
    delay(1);
}

void sendFireEvent() {
    if (client && client.connected()) {
        client.println("FIRE");  // 發送訊息給 Python（帶換行符）
        Serial.println("📡 已發送 FIRE 訊息給 Python");
    } else {
        Serial.println("WiFi 未連接！");
    }
}
