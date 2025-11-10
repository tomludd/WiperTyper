# WiperTyper - Voice to Keyboard

A Python application that transcribes speech to text using the NbAiLab/nb-whisper-large-distil-turbo-beta model and types it out as keyboard input.

## 🚀 Quick Start

**Automated Installation (Recommended):**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\install.ps1
```

See [QUICKSTART.md](QUICKSTART.md) for details.

**Manual Installation:**
See [INSTALL.md](INSTALL.md) for step-by-step guide.

## Features

- **Global Hotkey**: Press `Ctrl+Alt+V` to start/stop recording
- **Local Whisper Model**: Uses Hugging Face Transformers for Norwegian speech recognition
- **Keyboard Simulation**: Types transcribed text automatically
- **Real-time Processing**: Records and transcribes on-the-fly

## Installation

### Quick Install (Automated)
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\install.ps1
```
The script automatically installs everything you need!

### Manual Install
See [INSTALL.md](INSTALL.md) for detailed manual installation instructions.

### Prerequisites

- Python 3.8 or higher (3.11/3.12 recommended for GPU support)
- Windows 10/11
- Microphone or headset
- (Optional) NVIDIA GPU for 5-10x faster performance

### Setup

**Option 1: Automated (Recommended)**
```bash
.\install.ps1
```

**Option 2: Manual**

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the application:**
```bash
python wipertyper.py
```

The Whisper model will be downloaded automatically on first run (~3GB).

## Usage

1. Run `python wipertyper.py`
2. Wait for the model to load
3. Press `Ctrl+Alt+V` to start recording
4. Speak into your microphone
5. Press `Ctrl+Alt+V` again to stop recording
6. The app will transcribe and type the text automatically

## How It Works

1. **Hotkey Detection**: Uses `pynput` to register a global hotkey (Ctrl+Alt+V)
2. **Audio Capture**: Records audio from your default microphone using `pyaudio`
3. **Speech Recognition**: Transcribes using the NbAiLab Whisper model via Transformers
4. **Keyboard Typing**: Simulates keyboard input using `pynput`

## Troubleshooting

### "No speech detected"
- Make sure you're speaking clearly and loudly enough
- Check that the correct microphone is selected (shown when recording starts)
- Try speaking for at least 1-2 seconds

### Model loading takes too long
- The model is ~3GB and downloads on first run
- Subsequent runs will be much faster as the model is cached

### Permission errors
- On Windows: Make sure Python has microphone access in Settings → Privacy → Microphone

### ffmpeg not found
- Install ffmpeg: `winget install ffmpeg`
- Restart your terminal/PowerShell after installation

## Performance Optimization

### 1. GPU Acceleration (5-10x faster) 🚀
**Most Important!** If you have an NVIDIA GPU, install CUDA-enabled PyTorch:

```bash
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

The app automatically detects and uses your GPU. With GPU:
- Larger batch sizes (up to 16 on 8GB+ GPUs)
- Float16 precision for 2x speed boost
- BetterTransformer optimization
- torch.compile for maximum performance

### 2. Voice Activity Detection (VAD) - 30-50% faster
Remove silence from recordings before transcription:

**Option A: webrtcvad (requires C++ Build Tools)**
```bash
pip install webrtcvad
```

**Option B: If webrtcvad fails on Windows:**
The app works fine without VAD! The other optimizations (GPU, torch.compile, BetterTransformer) provide the biggest speed gains anyway.

To install webrtcvad on Windows, you need Visual C++ Build Tools:
1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install "Desktop development with C++"
3. Then: `pip install webrtcvad`

**Note:** VAD is optional - the app automatically detects if it's available and provides excellent performance without it when using GPU.

### 3. Use a Smaller Model (2-3x faster)
Trade accuracy for speed by using a smaller model. Edit line 50 in `wipertyper.py`:

```python
# Options (fastest to slowest):
model="NbAiLab/nb-whisper-small"          # Fastest, good for clear speech
model="NbAiLab/nb-whisper-medium"         # Balanced
model="NbAiLab/nb-whisper-large-distil-turbo-beta"  # Best accuracy (default)
```

### 4. Additional Optimizations
- **PyTorch 2.0+**: Automatic torch.compile acceleration on GPU (Linux/Mac only - requires Triton)
- **BetterTransformer**: Automatic optimization on supported GPUs
- **SSD Storage**: Faster model loading from cache
- **Close Heavy Apps**: Free up GPU/CPU resources
- **Speak Clearly**: Shorter, clearer recordings process faster

### Windows-Specific Notes
- **Triton/torch.compile**: Not available on Windows, but you still get GPU acceleration, BetterTransformer, and optimized batch sizes
- **Python 3.13**: Works with PyTorch nightly or CPU builds. For best GPU support, use Python 3.11 or 3.12

### Performance Summary
| Optimization | Speed Gain | Accuracy Impact | Windows Support |
|--------------|------------|-----------------|-----------------|
| GPU (CUDA) | 5-10x | None | ✅ Yes |
| VAD | 30-50% | None (removes silence only) | ⚠️ Requires C++ Build Tools |
| Smaller Model | 2-3x | Lower for complex speech | ✅ Yes |
| BetterTransformer | 5-10% | None | ✅ Yes (GPU) |
| torch.compile | 10-20% | None | ❌ No (requires Triton) |

## License

MIT License
