# WiperTyper Installer for Windows
# Automatisk installasjonsskript

Write-Host "================================" -ForegroundColor Cyan
Write-Host "  WiperTyper Installer" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
    
    # Check Python version
    $python313 = $false
    if ($pythonVersion -match "Python 3\.(\d+)") {
        $minorVersion = [int]$matches[1]
        if ($minorVersion -lt 8) {
            Write-Host "⚠ Python 3.8+ required. Please upgrade Python." -ForegroundColor Red
            exit 1
        }
        if ($minorVersion -eq 13) {
            $python313 = $true
            Write-Host "ℹ Python 3.13 detected. Will use CUDA 12.6 for GPU support." -ForegroundColor Cyan
        }
    }
} catch {
    Write-Host "❌ Python not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Installing Python 3.12..." -ForegroundColor Yellow
    try {
        winget install Python.Python.3.12 -e
        Write-Host "✓ Python installed. Please restart this script." -ForegroundColor Green
        exit 0
    } catch {
        Write-Host "❌ Failed to install Python. Please install manually from python.org" -ForegroundColor Red
        exit 1
    }
}

# Check if ffmpeg is installed
Write-Host ""
Write-Host "[2/6] Checking ffmpeg installation..." -ForegroundColor Yellow
try {
    $ffmpegVersion = ffmpeg -version 2>&1 | Select-Object -First 1
    Write-Host "✓ ffmpeg found" -ForegroundColor Green
} catch {
    Write-Host "⚠ ffmpeg not found. Installing..." -ForegroundColor Yellow
    try {
        winget install ffmpeg -e
        Write-Host "✓ ffmpeg installed" -ForegroundColor Green
        Write-Host "⚠ Please restart your terminal after installation completes!" -ForegroundColor Yellow
    } catch {
        Write-Host "❌ Failed to install ffmpeg automatically" -ForegroundColor Red
        Write-Host "  Please run: winget install ffmpeg" -ForegroundColor Yellow
        exit 1
    }
}

# Detect GPU
Write-Host ""
Write-Host "[3/6] Detecting GPU..." -ForegroundColor Yellow
$hasNvidiaGPU = $false
try {
    $gpu = Get-WmiObject Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
    if ($gpu) {
        $hasNvidiaGPU = $true
        Write-Host "✓ NVIDIA GPU detected: $($gpu.Name)" -ForegroundColor Green
        Write-Host "  Will install CUDA-enabled PyTorch for 5-10x faster performance!" -ForegroundColor Green
    } else {
        Write-Host "ℹ No NVIDIA GPU detected. Will use CPU mode." -ForegroundColor Cyan
    }
} catch {
    Write-Host "ℹ Could not detect GPU. Will use CPU mode." -ForegroundColor Cyan
}

# Upgrade pip
Write-Host ""
Write-Host "[4/6] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "✓ pip upgraded" -ForegroundColor Green

# Install PyTorch
Write-Host ""
Write-Host "[5/6] Installing PyTorch..." -ForegroundColor Yellow
if ($hasNvidiaGPU) {
    Write-Host "  Installing CUDA-enabled PyTorch (this may take a few minutes)..." -ForegroundColor Cyan
    
    # Choose CUDA version based on Python version
    $cudaUrl = if ($python313) {
        "https://download.pytorch.org/whl/cu126"  # CUDA 12.6 for Python 3.13
    } else {
        "https://download.pytorch.org/whl/cu121"  # CUDA 12.1 for Python 3.8-3.12
    }
    
    Write-Host "  Using CUDA index: $cudaUrl" -ForegroundColor Cyan
    
    try {
        python -m pip install torch torchvision torchaudio --index-url $cudaUrl --quiet
        Write-Host "✓ PyTorch with CUDA support installed" -ForegroundColor Green
    } catch {
        Write-Host "⚠ Failed to install CUDA PyTorch. Installing CPU version..." -ForegroundColor Yellow
        python -m pip install torch torchvision torchaudio --quiet
        Write-Host "✓ PyTorch (CPU) installed" -ForegroundColor Green
    }
} else {
    Write-Host "  Installing CPU-only PyTorch..." -ForegroundColor Cyan
    python -m pip install torch torchvision torchaudio --quiet
    Write-Host "✓ PyTorch (CPU) installed" -ForegroundColor Green
}

# Install other dependencies
Write-Host ""
Write-Host "[6/6] Installing other dependencies..." -ForegroundColor Yellow
Write-Host "  This includes: transformers, pynput, pyaudio, numpy, optimum, librosa" -ForegroundColor Cyan

# Try to install requirements
try {
    python -m pip install transformers pynput numpy optimum[onnxruntime] librosa --quiet
    Write-Host "✓ Core dependencies installed" -ForegroundColor Green
} catch {
    Write-Host "⚠ Some packages may have failed" -ForegroundColor Yellow
}

# Special handling for pyaudio (often problematic on Windows)
Write-Host ""
Write-Host "  Installing pyaudio..." -ForegroundColor Cyan
try {
    python -m pip install pyaudio --quiet 2>&1 | Out-Null
    Write-Host "✓ pyaudio installed" -ForegroundColor Green
} catch {
    Write-Host "⚠ pyaudio failed with pip. Trying alternative method..." -ForegroundColor Yellow
    try {
        python -m pip install pipwin --quiet
        pipwin install pyaudio --quiet
        Write-Host "✓ pyaudio installed via pipwin" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to install pyaudio automatically" -ForegroundColor Red
        Write-Host "  Please install manually from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio" -ForegroundColor Yellow
    }
}

# Check microphone permissions
Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "⚠ IMPORTANT: Check Microphone Permissions" -ForegroundColor Yellow
Write-Host "  1. Press Windows + I (open Settings)" -ForegroundColor White
Write-Host "  2. Go to: Privacy & Security → Microphone" -ForegroundColor White
Write-Host "  3. Enable 'Microphone access'" -ForegroundColor White
Write-Host "  4. Enable 'Let desktop apps access your microphone'" -ForegroundColor White
Write-Host ""

# Verify installation
Write-Host "Verifying installation..." -ForegroundColor Yellow
Write-Host ""

$allGood = $true

# Check Python imports
Write-Host "Testing Python packages..." -ForegroundColor Cyan
$testScript = @"
import sys
packages = ['torch', 'transformers', 'pynput', 'pyaudio', 'numpy', 'librosa']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError:
        print(f'❌ {pkg}')
        missing.append(pkg)
        
if missing:
    sys.exit(1)
    
# Check CUDA
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA available (GPU: {torch.cuda.get_device_name(0)})')
else:
    print('ℹ CUDA not available (CPU mode)')
"@

$testScript | python 2>&1 | ForEach-Object {
    if ($_ -match "✓") {
        Write-Host $_ -ForegroundColor Green
    } elseif ($_ -match "❌") {
        Write-Host $_ -ForegroundColor Red
        $allGood = $false
    } elseif ($_ -match "ℹ") {
        Write-Host $_ -ForegroundColor Cyan
    } else {
        Write-Host $_
    }
}

Write-Host ""
if ($allGood) {
    Write-Host "================================" -ForegroundColor Green
    Write-Host "  ✓ Installation Successful!" -ForegroundColor Green
    Write-Host "================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "To start WiperTyper, run:" -ForegroundColor White
    Write-Host "  python wipertyper.py" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "On first run, it will download the Whisper model (~3GB)." -ForegroundColor Yellow
    Write-Host "This is normal and only happens once." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Hotkey: Ctrl+Alt+V (start/stop recording)" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "================================" -ForegroundColor Yellow
    Write-Host "  ⚠ Installation completed with warnings" -ForegroundColor Yellow
    Write-Host "================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Some packages may need manual installation." -ForegroundColor Yellow
    Write-Host "Check the errors above and refer to INSTALL.md" -ForegroundColor Yellow
    Write-Host ""
}

# Offer to start the program
Write-Host "Would you like to start WiperTyper now? (y/n)" -ForegroundColor Yellow
$response = Read-Host
if ($response -eq "y" -or $response -eq "Y") {
    Write-Host ""
    Write-Host "Starting WiperTyper..." -ForegroundColor Green
    Write-Host "Press Ctrl+C to exit when done" -ForegroundColor Yellow
    Write-Host ""
    python wipertyper.py
}
