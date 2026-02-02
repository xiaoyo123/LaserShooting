# 找到bounding box中心點跟雷射光點的位置，以每個中心點為中心算出雷射光位置，並存成json檔
import cv2, time, os, json
from collections import deque
import threading
import queue
from ultralytics import YOLO
import numpy as np
import socket
import yaml


model = YOLO("best.pt")

# ====== 讀取 YAML 配置 ======
CONFIG_FILE = "config.yaml"
try:
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"✅ 成功載入配置檔: {CONFIG_FILE}")
except FileNotFoundError:
    print(f"⚠️  找不到配置檔 {CONFIG_FILE},使用預設值")
    config = {}

# ====== 參數 ======
BUFFER_SIZE = 20           # 保留最近20幀
PRE_FRAMES = 3             # 取按下前3幀
POST_FRAMES = 5            # 取按下後5幀（要靠一點延遲等到幀進來）
POST_WAIT_SEC = 0.12       # 等待後續幀進buffer的時間（依FPS調）

USER_ID = "user_001"
MAX_SHOTS = 5
SAVE_DIR = "shots_json"
SAVE_FRAMES_DIR = "shots_frames"  # 儲存處理幀的資料夾
CONF_THRES = 0.6        # YOLO 偵測信心度閾值
DEBOUNCE_SEC = 0.25          # 去抖動:至少隔這麼久才算下一發(避免同一發連續幀重複寫)

# 從 YAML 配置檔讀取相機和顏色參數
CAM_ID = config.get('camera', {}).get('cam_id', 1)
LOWER_RED1 = np.array(config.get('hsv_color', {}).get('lower_red1', [0, 25, 100]))
UPPER_RED1 = np.array(config.get('hsv_color', {}).get('upper_red1', [10, 255, 255]))
LOWER_RED2 = np.array(config.get('hsv_color', {}).get('lower_red2', [170, 25, 100]))
UPPER_RED2 = np.array(config.get('hsv_color', {}).get('upper_red2', [180, 255, 255]))

# print(f"📷 相機 ID: {CAM_ID}")
# print(f"🎨 HSV 範圍 1: {LOWER_RED1} ~ {UPPER_RED1}")
# print(f"🎨 HSV 範圍 2: {LOWER_RED2} ~ {UPPER_RED2}")

# WiFi連接參數
ESP32_IP = "192.168.4.1"    # ESP32 的 IP 地址（請修改為你的 ESP32 IP）
ESP32_PORT = 8080            # ESP32 監聽的端口號

# Unity TCP Server 參數
TCP_HOST = "127.0.0.1"    # TCP Server IP
TCP_PORT = 5000           # TCP Server Port

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_FRAMES_DIR, exist_ok=True)

# ====== 共享資料 ======
frame_buffer = deque(maxlen=BUFFER_SIZE)  # 每筆: (ts, frame)
fire_events = queue.Queue()               # 存 "fire timestamp"
unity_reset_events = queue.Queue()        # 存 Unity 的 reset 訊號
tcp_clients = []                          # 存放連線的 Unity 客戶端
stop_flag = False
round_active = threading.Event()          # 控制是否允許觸發射擊
round_active.set()                        # 初始允許射擊

def atomic_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def send_to_unity(data):
    """發送 JSON 資料給所有連線的 Unity 客戶端"""
    json_str = json.dumps(data, ensure_ascii=False) + "\n"  # 加換行符作為分隔
    json_bytes = json_str.encode('utf-8')
    
    disconnected = []
    for client in tcp_clients:
        try:
            client.sendall(json_bytes)
        except:
            disconnected.append(client)
    
    # 移除斷線的客戶端
    for client in disconnected:
        tcp_clients.remove(client)
        try:
            client.close()
        except:
            pass

# ====== TCP Server 執行緒 ======
def tcp_server_loop():
    global stop_flag
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((TCP_HOST, TCP_PORT))
    server.listen(5)
    server.settimeout(1.0)  # 設置 timeout 以便能檢查 stop_flag
    
    print(f"🌐 TCP Server 啟動於 {TCP_HOST}:{TCP_PORT}")
    
    while not stop_flag:
        try:
            client, addr = server.accept()
            tcp_clients.append(client)
            print(f"✅ Unity 客戶端連線: {addr}")
            
            # 為每個客戶端啟動接收執行緒
            client_thread = threading.Thread(target=handle_unity_client, args=(client, addr), daemon=True)
            client_thread.start()
            
        except socket.timeout:
            continue
        except:
            break
    
    # 關閉所有連線
    for client in tcp_clients:
        try:
            client.close()
        except:
            pass
    server.close()
    print("🌐 TCP Server 已關閉")

def handle_unity_client(client, addr):
    """處理來自 Unity 客戶端的訊息"""
    global stop_flag
    print(f"📡 開始監聽來自 {addr} 的訊息")
    
    try:
        while not stop_flag:
            # 接收 Unity 傳來的資料
            data = client.recv(1024)
            if not data:
                break
                
            message = data.decode('utf-8').strip()
            print(f"📨 收到 Unity 訊息: {message}")
            
            # 檢查是否為 RESET 訊號
            if "RESET" in message.upper() or "reset" in message.lower():
                print("✅ 收到 Unity RESET 訊號")
                unity_reset_events.put(True)
                
    except Exception as e:
        print(f"⚠️  Unity 客戶端 {addr} 連線錯誤: {e}")
    finally:
        if client in tcp_clients:
            tcp_clients.remove(client)
        try:
            client.close()
        except:
            pass
        print(f"❌ Unity 客戶端 {addr} 已斷線")

# ====== 你的偵測：請接你現有的 yolo_detect + select_best_hit_candidate ======
def yolo_detect(frame):
    results = model.predict(source=frame, save=False, verbose=False)  # 不儲存，減少輸出
    return results
def detect_point_in_roi(roi_image, offset_x, offset_y):
    """在指定的 ROI 區域內偵測綠色光點，回傳絕對座標的bbox list: [(x,y,w,h), ...]"""
    point_boxes = []
    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 25, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 25, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # lower_red1 = np.array([0, 0, 200])
    # upper_red1 = np.array([15, 255, 255])
    # lower_red2 = np.array([165, 0, 200])
    # upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) > 30 :  # 閾值
            x, y, w, h = cv2.boundingRect(c)
            point_boxes.append((x + offset_x, y + offset_y, w, h))

    return point_boxes

def bbox_center_xyxy(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def bbox_center_xywh(x, y, w, h):
    return (x + w / 2.0, y + h / 2.0)
def select_best_hit_candidate(frame, yolo_results):
    """
    從所有YOLO框中找出「框內有綠點」的候選，
    並挑選 conf 最高的一個作為本次擊中目標。
    同時計算該目標從左到右是第幾個（最左邊是1）
    回傳 dict 或 None
    """
    h_img, w_img = frame.shape[:2]
    
    # 先收集所有有效的目標框（用於排序編號）
    all_targets = []
    candidates = []
    
    for r in yolo_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < CONF_THRES:
                continue

            # clamp
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w_img, x2); y2 = min(h_img, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # 計算中心點 x 座標用於排序
            center_x = (x1 + x2) / 2.0
            all_targets.append({
                "center_x": center_x,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cls": cls, "conf": conf
            })

    # 按照 center_x 從左到右排序，建立編號
    all_targets.sort(key=lambda t: t["center_x"])
    # target_numbers = {t["center_x"]: idx + 1 for idx, t in enumerate(all_targets)}
    for idx, t in enumerate(all_targets):
        t["No"] = idx

    # 現在檢查每個目標是否有綠點
    for target in all_targets:
        x1, y1, x2, y2 = target["x1"], target["y1"], target["x2"], target["y2"]
        cls = target["cls"]
        conf = target["conf"]
        center_x = target["center_x"]
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # 在這個框內偵測綠點
        point_boxes = detect_point_in_roi(roi, x1, y1)
        if len(point_boxes) == 0:
            continue

        # 多個綠點時：取面積最大的那個（通常比較穩）
        def area(pb): 
            _, _, pw, ph = pb
            return pw * ph
        px, py, pw, ph = max(point_boxes, key=area)
        green_area = pw * ph

        target_cx, target_cy = bbox_center_xyxy(x1, y1, x2, y2)
        laser_cx, laser_cy = bbox_center_xywh(px, py, pw, ph)

        # 計算像素差值
        dx_px = laser_cx - target_cx
        dy_px = laser_cy - target_cy
        
        # 計算標靶的寬高
        target_width = x2 - x1
        target_height = y2 - y1
        
        # 轉換為歸一化座標: 中心為(0,0)，左上(-0.5,-0.5)，右下(0.5,0.5)
        # X軸: 向右為正
        # Y軸: 向下為正 (與圖像座標系統一致)
        dx_normalized = (dx_px / target_width)
        dy_normalized = (dy_px / target_height)  # 向下為正
        
        # 計算點落在哪個橢圓區域
        # 使用橢圓距離公式: sqrt((x/a)^2 + (y/b)^2)
        # 其中 a, b 是橢圓的半軸長度,這裡都是 1.0 (因為已歸一化)
        distance_normalized = np.sqrt(dx_normalized**2 + dy_normalized**2)
        
        # 將標靶分成5個同心橢圓計分: 10, 9, 8, 7, 6
        # 每層橢圓範圍: 0.5 / 5 = 0.1
        # 例如: 點在 (0.3, 0.1) → sqrt(0.3^2 + 0.1^2) = 0.316 → 8分區
        if distance_normalized <= 0.1:
            score = 10  
        elif distance_normalized <= 0.2:
            score = 9   
        elif distance_normalized <= 0.3:
            score = 8   
        elif distance_normalized <= 0.4:
            score = 7   
        else:
            score = 6
        
        candidate = {
            # "No": target_numbers[center_x],  # 從左到右的編號
            "No": target["No"],
            "cls": cls,
            "conf": conf,
            "target_center": (target_cx, target_cy), # 真正中心座標
            "laser_center": (laser_cx, laser_cy),
            "dx": dx_normalized,  # 歸一化座標
            "dy": dy_normalized,  # 歸一化座標
            "dx_px": dx_px,  # 保留像素值供debug
            "dy_px": dy_px,
            "distance": distance_normalized,  # 距離中心的歸一化距離
            "score": score,  # 環狀計分
            "green_area": green_area  # 綠點面積
        }
        
        candidates.append(candidate)
    
    if not candidates: # 完全沒偵測到綠點
        return None
    
    # 從所有候選中選擇綠點面積最大的（最明顯的擊中）
    best = max(candidates, key=lambda c: c["green_area"])
    return best

def detect_from_frames(frames, shot_idx):
    """
    frames: list of (ts, frame)
    回傳 best_payload（命中）或 miss_payload（沒命中）
    同時將所有處理的幀存成圖片
    """
    best = None
    best_ts = None
    best_frame_idx = None

    # 建立此次shot的資料夾
    shot_dir = os.path.join(SAVE_FRAMES_DIR, f"shot{shot_idx:02d}")
    os.makedirs(shot_dir, exist_ok=True)

    for idx, (ts, frame) in enumerate(frames):
        # 儲存原始幀
        frame_path = os.path.join(shot_dir, f"frame_{idx:02d}_{int(ts*1000)}.jpg")
        cv2.imwrite(frame_path, frame)
        
        results = yolo_detect(frame)
        cand = select_best_hit_candidate(frame, results)
        if cand is None:
            continue

        # 例：用 green_area 當排序依據（你也可以加上綠點面積、距離等）
        score = cand["green_area"]
        if (best is None) or (score > best["green_area"]):
            best = cand
            best_ts = ts
            best_frame_idx = idx

    # 如果有最佳幀，額外標記它
    if best_frame_idx is not None:
        best_marker_path = os.path.join(shot_dir, f"BEST_frame_{best_frame_idx:02d}.txt")
        with open(best_marker_path, "w") as f:
            f.write(f"Best frame index: {best_frame_idx}\n")
            f.write(f"Timestamp: {best_ts}\n")
            f.write(f"Green area: {best['green_area']}\n")

    return best, best_ts

# ====== 顯示執行緒：即時顯示偵測結果 ======
def display_loop():
    global stop_flag
    while not stop_flag:
        if len(frame_buffer) == 0:
            time.sleep(0.01)
            continue
        
        # 取最新的幀
        ts, frame = frame_buffer[-1]
        display_frame = frame.copy()
        
        # 執行YOLO偵測
        results = yolo_detect(display_frame)
        
        # 繪製所有偵測框和綠點
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                if conf < CONF_THRES:
                    continue
                
                # 繪製bounding box
                color = (0, 255, 0) if conf >= CONF_THRES else (0, 165, 255)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # 標註信心度
                label = f"Target {conf:.2f}"
                cv2.putText(display_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 繪製5層同心長方形 (計分區域)
                target_cx = (x1 + x2) / 2.0
                target_cy = (y1 + y2) / 2.0
                target_width = x2 - x1
                target_height = y2 - y1
                
                # 5層橢圓: 0.1, 0.2, 0.3, 0.4, 0.5
                # 顏色: 紅->橙->黃->淺藍->深藍
                zone_colors = [
                    (0, 0, 255),      # 10分: 紅色
                    (0, 165, 255),    # 9分: 橙色
                    (0, 255, 255),    # 8分: 黃色
                    (255, 255, 0),    # 7分: 青色
                    (255, 0, 0)       # 6分: 藍色
                ]
                zone_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
                
                for i, ratio in enumerate(zone_ratios):
                    # 計算此層橢圓的半軸長度
                    axes_w = int(target_width * ratio)
                    axes_h = int(target_height * ratio)
                    
                    # 繪製橢圓
                    cv2.ellipse(display_frame, 
                               (int(target_cx), int(target_cy)),  # 中心點
                               (axes_w, axes_h),                   # 半軸長度 (寬, 高)
                               0,                                   # 旋轉角度
                               0, 360,                             # 起始和結束角度
                               zone_colors[i], 1)                  # 顏色和線寬
                
                # 繪製中心十字
                cross_size = 5
                cv2.line(display_frame, 
                        (int(target_cx - cross_size), int(target_cy)),
                        (int(target_cx + cross_size), int(target_cy)),
                        (0, 0, 255), 2)
                
                # 在ROI內偵測綠點
                h_img, w_img = display_frame.shape[:2]
                x1_c = max(0, x1); y1_c = max(0, y1)
                x2_c = min(w_img, x2); y2_c = min(h_img, y2)
                
                if x2_c > x1_c and y2_c > y1_c:
                    roi = frame[y1_c:y2_c, x1_c:x2_c]
                    if roi.size > 0:
                        point_boxes = detect_point_in_roi(roi, x1_c, y1_c)
                        
                        # 繪製綠點
                        for px, py, pw, ph in point_boxes:
                            cv2.rectangle(display_frame, (px, py), (px+pw, py+ph), (0, 255, 255), 2)
                            # 綠點中心
                            # cx, cy = bbox_center_xywh(px, py, pw, ph)
                            # cv2.circle(display_frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        
        # 顯示畫面
        cv2.imshow("Detection", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag = True
            break
    
    cv2.destroyAllWindows()

# ====== 相機擷取執行緒：只負責buffer ======
def camera_loop(cam_id=None):
    global stop_flag
    if cam_id is None:
        cam_id = CAM_ID  # 使用全域配置的相機 ID
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ 無法開啟攝影機")
        stop_flag = True
        return
    
    # 設定相機解析度（可選）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 強制設定顯示視窗大小為 1280x720
    # cv2.namedWindow("Real-time Laser Detection", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Real-time Laser Detection", 1280, 720)
        
    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            continue
        ts = time.time()
        frame_buffer.append((ts, frame))

    cap.release()

# ====== 硬體觸發執行緒：透過 WiFi Socket 讀取 ESP32 按鈕訊息 ======
def trigger_loop_wifi():
    global stop_flag
    sock = None
    last_fire_time = 0  # 用於去抖動
    
    try:
        # 建立 TCP Socket 連接到 ESP32
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)  # 設定連接超時
        print(f"正在連接到 ESP32 ({ESP32_IP}:{ESP32_PORT})...")
        sock.connect((ESP32_IP, ESP32_PORT))
        sock.settimeout(1)  # 連接後設定較短的接收超時
        print(f"✅ 已透過 WiFi 連接到 ESP32: {ESP32_IP}:{ESP32_PORT}")
    except Exception as e:
        print(f"❌ 無法連接到 ESP32: {e}")
        print(f"   請確認:")
        print(f"   1. ESP32 已開機並連接到 WiFi")
        print(f"   2. IP 地址 {ESP32_IP} 正確")
        print(f"   3. ESP32 上的 TCP Server 正在運行於端口 {ESP32_PORT}")
        stop_flag = True
        return
    
    buffer = ""
    while not stop_flag:
        try:
            # 接收數據
            data = sock.recv(1024).decode('utf-8')
            if not data:
                print("⚠️ ESP32 連接已斷開")
                break
            
            buffer += data
            
            # 處理接收到的完整訊息（以換行符分隔）
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()
                
                if line:  # 如果接收到訊息（例如 ESP32 發送 "FIRE"）
                    print(f"📡 收到 ESP32 訊息: {line}")
                    
                    # 只有在 round_active 時才接受射擊訊號
                    if round_active.is_set():
                        # 去抖動：避免在短時間內重複觸發
                        current_time = time.time()
                        if current_time - last_fire_time >= DEBOUNCE_SEC:
                            fire_events.put(current_time)
                            last_fire_time = current_time
                            print(f"🔥 觸發射擊事件")
                        else:
                            print(f"⏭️ 去抖動: 忽略過快的觸發 ({current_time - last_fire_time:.3f}s)")
                    else:
                        print("⚠️  當前回合已結束，等待 Unity reset 中...")
                        
        except socket.timeout:
            # 超時是正常的，繼續接收
            continue
        except Exception as e:
            print(f"❌ 讀取 ESP32 資料錯誤: {e}")
            break
    
    if sock:
        sock.close()
    print("已關閉 ESP32 WiFi 連接")

# ====== 事件處理：一收到fire就抓幀並偵測，輸出JSON ======
def fire_handler_loop():
    global stop_flag
    
    while not stop_flag:
        shot_idx = 0
        print("\n🎯 === 新回合開始 ===")
        print(f"等待射擊訊號... (剩餘 {MAX_SHOTS} 發)")
        
        # 執行五發射擊
        while not stop_flag and shot_idx < MAX_SHOTS:
            fire_ts = fire_events.get()  # block 等待

            # 等待一點時間，讓 POST_FRAMES 幀進buffer（跟FPS有關）
            time.sleep(POST_WAIT_SEC)

            # 把 buffer 複製出來避免被同時修改
            buf = list(frame_buffer)

            # 找到 fire_ts 在 buffer 中的位置（以 timestamp 切）
            # 取 fire_ts 前後幀
            # 作法：找最後一個 ts <= fire_ts 的 index
            idx = None
            for i in range(len(buf)-1, -1, -1):
                if buf[i][0] <= fire_ts:
                    idx = i
                    break
            if idx is None:
                idx = 0

            start = max(0, idx - PRE_FRAMES)
            end = min(len(buf), idx + 1 + POST_FRAMES)
            window = buf[start:end]

            best, best_ts = detect_from_frames(window, shot_idx + 1)

            shot_idx += 1
            ts_now = time.time()

            if best is not None:
                payload = {
                    "shot_idx": shot_idx,
                    "hit": True,
                    "target": {
                        "No": int(best["No"]),
                        "x": float(best["dx"]),
                        "y": float(best["dy"]),
                        "score": int(best["score"])  # 環狀計分: 10, 9, 8, 7, 6
                    }
                }
            else:
                payload = {
                    "shot_idx": shot_idx,
                    "hit": False,
                }

            fname = f"{USER_ID}_shot{shot_idx:02d}_{int(ts_now*1000)}.json"
            out_path = os.path.join(SAVE_DIR, fname)
            atomic_write_json(out_path, payload)
            
            # 發送給 Unity
            send_to_unity(payload)
            
            print(f"✅ 第 {shot_idx}/{MAX_SHOTS} 發 - 寫入 {out_path}  hit={payload['hit']}")

        # 五發射完，停止接受新的射擊訊號
        print("\n🔚 五發射擊完成！")
        round_active.clear()  # 停止接受射擊訊號
        print("⏳ 等待 Unity 發送 RESET 訊號...")
        
        # 等待 Unity 的 reset 訊號
        unity_reset_events.get()  # block 等待
        
        # 收到 reset，清空射擊事件隊列並重新開始
        print("✅ 收到 Unity RESET 訊號，準備下一輪...")
        
        # 清空可能殘留的射擊事件
        while not fire_events.empty():
            fire_events.get()
        
        round_active.set()  # 重新允許射擊
        time.sleep(0.5)  # 短暫延遲避免誤觸
    
    print("🛑 fire_handler_loop 結束")

# ====== 主程式 ======
if __name__ == "__main__":
    t_tcp = threading.Thread(target=tcp_server_loop, daemon=True)  # TCP Server
    t_cam = threading.Thread(target=camera_loop, daemon=True)
    t_dsp = threading.Thread(target=display_loop, daemon=True)  # 顯示執行緒
    t_trg = threading.Thread(target=trigger_loop_wifi, daemon=True)  # WiFi連接ESP32
    t_hnd = threading.Thread(target=fire_handler_loop, daemon=True)

    t_tcp.start()
    t_cam.start()
    t_dsp.start()
    t_trg.start()
    t_hnd.start()

    try:
        while t_hnd.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass

    stop_flag = True
    print("程式結束")
