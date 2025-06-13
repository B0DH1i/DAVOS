---
title: DAVOS - Emotion Analysis and Productivity Automation System
emoji: 🧠
colorFrom: green
colorTo: blue
sdk: gradio
app_file: gui_main.py
pinned: false
sdk_version: 5.33.0
---

# DAVOS - Emotion Analysis and Productivity Automation System

DAVOS is an advanced AI system that analyzes emotions in real-time using facial expressions and vocal cues to enhance user productivity and emotional well-being. The system proactively intervenes by playing appropriate music or binaural beats based on the detected emotional state.

## 🚀 Key Features

### Multi-Modal Emotion Analysis
- **Face Detection**: Real-time facial detection using DNN model
- **Facial Emotion Recognition (FER)**: Emotion classification using VGG16 architecture
- **Speech Emotion Recognition (SER)**: Voice-based emotion analysis using Whisper model

### Smart Intervention System
- **Proactive Music Selection**: Automatically plays suitable sounds from the Lazanov Music Library
- **Emotion-Based Adaptation**: Adjusts audio output based on detected emotional states (e.g., stress, fatigue)
- **Real-time Processing**: Continuous monitoring and instant response to emotional changes

### Dual Operation Modes
1. **Interactive Mode** (`gui_main.py`):
   - User-friendly Gradio interface
   - Real-time emotion analysis visualization
   - Manual control and testing capabilities

2. **Automated Mode** (`main_live_controller.py`):
   - Background operation
   - Continuous monitoring
   - Automatic intervention system

## 🛠️ Installation and Setup

### Prerequisites
- Python 3.8+
- FFmpeg for audio processing
- Webcam and microphone access

### System Dependencies
Install required system packages:
```bash
sudo apt-get update
sudo apt-get install ffmpeg portaudio19-dev libportaudio2
```

### Python Dependencies
Install required Python packages:
```bash
pip install -r requirements.txt
```

### Running the Application

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DAVOS.git
cd DAVOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
   - For interactive mode:
   ```bash
   python gui_main.py
   ```
   - For automated mode:
   ```bash
   python main_live_controller.py
   ```

## 📁 Project Structure
```
DAVOS/
├── src/                    # Source code
├── data/                   # Data files
├── trained_models/         # Pre-trained models
├── plots/                  # Generated plots
├── logs/                   # Log files
├── gui_main.py            # Interactive mode interface
├── main_live_controller.py # Automated mode controller
├── requirements.txt        # Python dependencies
└── packages.txt           # System dependencies
```

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/DAVOS/issues).

## 📫 Contact
For any questions or suggestions, please open an issue in the repository.