# Senior_Project-Raspberry_pi

## OpenCV + Picamera2 即時人臉偵測串流

已新增 `face_stream.py`，可在 Raspberry Pi 上執行：

- 使用 Picamera2 擷取即時畫面
- 使用 OpenCV Haar Cascade 做臉部偵測
- 將結果以 MJPEG（HTTP）串流輸出給瀏覽器

### 1) 安裝套件

建議先在 Raspberry Pi 更新並安裝必要套件：

```bash
sudo apt update
sudo apt install -y python3-opencv python3-flask python3-picamera2
```

### 2) 執行程式

```bash
python3 face_stream.py --host 0.0.0.0 --port 8000
```

可選參數：

- `--width`：串流寬度（預設 1280）
- `--height`：串流高度（預設 720）
- `--no-vflip`：關閉垂直翻轉

### 3) 從其他裝置觀看

在同一個網路內，於其他電腦/手機瀏覽器開啟：

```text
http://<RaspberryPi_IP>:8000/
```

例如 Raspberry Pi IP 是 `192.168.1.50`，則網址為：

```text
http://192.168.1.50:8000/
```

### 4) 注意事項

- 若瀏覽器看不到畫面，先確認：
  - Raspberry Pi 與觀看裝置在同一網段
  - 防火牆未阻擋該埠號（如 8000）
- 若要提升速度，可先降低解析度（例如 `--width 640 --height 480`）。
