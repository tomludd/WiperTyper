# WiperTyper - Installasjonsveiledning

Enkel guide for å installere WiperTyper på en ny maskin.

## Systemkrav

- **Windows 10/11**
- **Python 3.11 eller 3.12** (Python 3.13 fungerer også, men har begrenset GPU-støtte)
- **8GB RAM minimum** (16GB anbefalt)
- **NVIDIA GPU med 4GB+ VRAM** (valgfritt, men anbefalt for 5-10x raskere ytelse)
- **Mikrofon/Headset**

## Trinn-for-trinn installasjon

### 1. Installer Python

**Alternativ A: Microsoft Store (anbefalt)**
```powershell
winget install Python.Python.3.12
```

**Alternativ B: Last ned fra python.org**
- Gå til https://www.python.org/downloads/
- Last ned Python 3.12.x
- ✅ Husk å krysse av "Add Python to PATH" under installasjonen!

Verifiser installasjon:
```powershell
python --version
```

### 2. Installer ffmpeg

Ffmpeg trengs for lydprosessering:

```powershell
winget install ffmpeg
```

**Viktig:** Start en ny PowerShell/Terminal etter installasjon!

### 3. Last ned WiperTyper

**Alternativ A: Med Git**
```powershell
git clone https://github.com/[din-bruker]/wipertyper.git
cd wipertyper
```

**Alternativ B: Last ned ZIP**
- Last ned prosjektet som ZIP
- Pakk ut til ønsket mappe
- Åpne PowerShell i mappen

### 4. Installer Python-pakker

#### For maskiner UTEN NVIDIA GPU (CPU-only):
```powershell
pip install -r requirements.txt
```

#### For maskiner MED NVIDIA GPU (anbefalt):

**Python 3.13:**
```powershell
# Installer PyTorch med CUDA 12.6 (for Python 3.13)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Installer resten av pakkene
pip install transformers pynput pyaudio numpy optimum[onnxruntime] librosa
```

**Python 3.8-3.12:**
```powershell
# Installer PyTorch med CUDA 12.1 (for Python 3.8-3.12)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Installer resten av pakkene
pip install transformers pynput pyaudio numpy optimum[onnxruntime] librosa
```

### 5. Gi mikrofon-tillatelser

Windows må gi Python tilgang til mikrofonen:

1. Trykk `Windows + I` (Åpne Settings)
2. Gå til **Privacy & Security → Microphone**
3. Aktiver **"Microphone access"**
4. Aktiver **"Let desktop apps access your microphone"**

### 6. Kjør WiperTyper

```powershell
python wipertyper.py
```

Første gang du kjører vil den laste ned Whisper-modellen (~3GB). Dette tar noen minutter.

### 7. Test at det fungerer

1. Start programmet: `python wipertyper.py`
2. Vent til du ser "Ready! Waiting for hotkey..."
3. Trykk `Ctrl+Alt+V` for å starte opptak
4. Snakk tydelig i mikrofonen
5. Trykk `Ctrl+Alt+V` igjen for å stoppe
6. Programmet transkriberer og skriver teksten!

## Feilsøking

### "No module named 'pyaudio'"

**Problem:** PyAudio installerte ikke riktig.

**Løsning:**
```powershell
pip install pipwin
pipwin install pyaudio
```

Eller last ned wheel-fil fra: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

### "ffmpeg not found"

**Problem:** ffmpeg er ikke i PATH.

**Løsning:**
1. Installer på nytt: `winget install ffmpeg`
2. Start en **helt ny** PowerShell/Terminal
3. Test: `ffmpeg -version`

### "Audio volume: 0" / Ingen lyd

**Problem:** Mikrofonen er ikke aktiv eller Python har ikke tillatelser.

**Løsning:**
1. Sjekk at headset/mikrofon er på og tilkoblet
2. Sjekk mikrofoninnstillinger i Windows (se steg 5 over)
3. Test mikrofonen i Windows Sound Settings
4. Kjør `python list_microphones.py` for å se tilgjengelige enheter

### "CUDA not available" (selv med NVIDIA GPU)

**Problem:** PyTorch er installert uten CUDA-støtte.

**Løsning for Python 3.13:**
```powershell
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**Løsning for Python 3.8-3.12:**
```powershell
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verifiser:
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```
Skal vise `True`.

### Modellen gir feil tekst

**Problem:** Modell-cache kan være korrupt.

**Løsning:**
Slett cache og last ned på nytt:
```powershell
# Windows
rmdir /s "%USERPROFILE%\.cache\huggingface\hub"

# Deretter kjør programmet igjen
python wipertyper.py
```

## Ytelsesoptimalisering

### GPU er best!
Med NVIDIA GPU får du **5-10x raskere** transkripsjon. Installer CUDA-versjonen av PyTorch (se steg 4).

### Mindre modell for ekstra hastighet
For raskere (men litt mindre nøyaktig) transkripsjon, endre linje 78 i `wipertyper.py`:

```python
# Fra:
model="NbAiLab/nb-whisper-large-distil-turbo-beta"

# Til en av disse:
model="NbAiLab/nb-whisper-medium"  # 2x raskere
model="NbAiLab/nb-whisper-small"   # 3x raskere
```

### Bruk SSD
Installer WiperTyper på en SSD for raskere modell-lasting.

## Oppgradering

For å oppdatere til siste versjon:

```powershell
# Med Git
git pull

# Oppdater pakker
pip install --upgrade -r requirements.txt
```

## Avinstallering

```powershell
# Fjern programmet
cd ..
rmdir /s wipertyper

# Fjern modell-cache (valgfritt, sparer ~3GB)
rmdir /s "%USERPROFILE%\.cache\huggingface"
```

## Support

Ved problemer:
1. Sjekk feilsøkingsseksjonen over
2. Kjør `python test_model.py` for å teste mikrofon og modell
3. Kjør `python list_microphones.py` for å se tilgjengelige mikrofoner
4. Sjekk at alle krav er oppfylt (Python, ffmpeg, mikrofontillatelser)

---

**Lykke til med WiperTyper! 🎤🚀**
