# 🎵 AirSwipe Music Controller

Gesture-controlled local music player using **Python, OpenCV, MediaPipe, and pygame**. Control your music **hands‑free** with simple air gestures!

---

## 🚀 Features

* ✊ **Fist → Play / Pause toggle**
* 👉 **Swipe Right → Next Track**
* 👈 **Swipe Left → Previous Track**
* 🖐️ **Open Palm → Stop Music**
* 📁 Automatically loads all songs from a selected folder
* 🎯 Fast hand‑tracking using **MediaPipe Hands**
* 🎧 Works with MP3 / WAV formats

---

## 📂 Project Structure

```
Music-controller/
│── main.py
│── requirements.txt
│── README.md
│── music/              # Add your MP3/WAV files here
└── utils/
    ├── gesture_detector.py
    └── player.py
```

---

## 🛠️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Anil-glith/Music-controller.git
cd Music-controller
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If you get errors with MediaPipe:

```bash
pip install mediapipe opencv-python pygame
```

---

## ▶️ Running the Project

```bash
python main.py
```

Make sure your webcam is connected.

---

## ✋ Supported Gestures

| Gesture         | Action             |
| --------------- | ------------------ |
| **Fist**        | Play / Pause music |
| **Swipe Right** | Next Track         |
| **Swipe Left**  | Previous Track     |
| **Open Palm**   | Stop Music         |

---

## 🧠 How It Works

* Uses **OpenCV** to read webcam frames
* **MediaPipe Hands** detects hand landmarks in real-time
* **GestureDetector** interprets gesture patterns
* **Player** handles music playback using pygame
* Gestures convert into commands → Play/Pause/Next/Prev

---

## 📦 Requirements

* Python 3.8+ recommended
* Webcam
* OS: Windows / Linux / MacOS

Libraries:

```
opencv-python
mediapipe
pygame
numpy
```

---

## 🛠️ Customizing

### Change Music Folder

Edit in `main.py`:

```python
MUSIC_FOLDER = "music"  # change path here
```

### Adjust Gesture Sensitivity

Inside `gesture_detector.py`:

```python
SWIPE_THRESHOLD = 80
```

Increase value → Harder to detect swipe.

---

## 📸 Screenshot / Demo

(Add your demo images or GIF here)

---

## 🤝 Contributing

Pull requests are welcome! If you find issues, feel free to open an issue.

---

## 📜 License

This project is open-source under the MIT License.

---

## 👤 Author

**Anil**
GitHub: [https://github.com/Anil-glith](https://github.com/Anil-glith)

---

## ⭐ If you like this project

Please consider giving the repo a **star** on GitHub! 🌟
