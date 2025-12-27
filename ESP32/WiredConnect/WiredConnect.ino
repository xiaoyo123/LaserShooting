const int BTN_PIN = 4;          // 按鈕接的腳位
bool lastState = HIGH;
unsigned long lastFireMs = 0;
const unsigned long debounceMs = 30;

void setup() {
    Serial.begin(115200);
    pinMode(BTN_PIN, INPUT_PULLUP);
}

void loop() {
    bool cur = digitalRead(BTN_PIN);

  // 偵測「按下」(HIGH -> LOW).
    if (lastState == HIGH && cur == LOW) {
        unsigned long now = millis();
        if (now - lastFireMs > debounceMs) {
            Serial.println("FIRE");   // 🔥 關鍵：送給 Python
            lastFireMs = now;
        }
    }
    lastState = cur;
    delay(1);
}