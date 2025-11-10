"""
WiperTyper - Voice to Keyboard using Whisper
Press Ctrl+Alt+V to start/stop recording and type the transcribed text
"""
import os
import sys
import threading
import numpy as np
import pyaudio
import wave
from pynput import keyboard
from pynput.keyboard import Controller, Key
from transformers import pipeline
import torch

# VAD (Voice Activity Detection) for removing silence
try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False

class WiperTyper:
    def __init__(self):
        self.is_recording = False
        self.audio_frames = []
        self.audio = None
        self.stream = None
        self.keyboard_controller = Controller()
        self.should_exit = False  # Flag for clean exit
        
        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        
        print("WiperTyper - Voice to Keyboard")
        print("================================")
        print("Loading Whisper model...")
        
        # Load Whisper model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        print(f"Using device: {device}")
        if device == "cuda":
            print("🚀 GPU detected - transcription will be much faster!")
        else:
            print("💡 Tip: Install CUDA PyTorch for 5-10x faster performance")
        
        # VAD setup
        if VAD_AVAILABLE:
            self.vad = webrtcvad.Vad(1)  # Aggressiveness: 0-3 (1 is less aggressive, keeps more audio)
            print("✓ VAD (Voice Activity Detection) enabled - removes silence for faster processing")
        else:
            self.vad = None
            print("💡 Tip: Install webrtcvad for faster processing (pip install webrtcvad)")
        self.vad = None  # Disable VAD for testing

        # Optimize batch size based on GPU memory
        if device == "cuda":
            try:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_memory_gb >= 8:
                    batch_size = 16  # Large GPU
                elif gpu_memory_gb >= 4:
                    batch_size = 8   # Medium GPU
                else:
                    batch_size = 4   # Small GPU
                print(f"✓ GPU Memory: {gpu_memory_gb:.1f}GB - Using batch_size={batch_size}")
            except:
                batch_size = 8
        else:
            batch_size = 1
        
        try:
            # Suppress deprecation warnings
            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning)
            
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model="NbAiLab/nb-whisper-large-distil-turbo-beta",
                device=device,
                torch_dtype=torch_dtype,
                chunk_length_s=15,  # Shorter chunks for better quality
                batch_size=batch_size,
                model_kwargs={"use_cache": False}  # Disable caching to avoid issues
            )
            
            print("✓ Model loaded successfully")
            print("ℹ Using Norwegian Whisper model - no language override needed")
            
            # Apply BetterTransformer optimization if available (PyTorch 2.0+)
            if device == "cuda" and hasattr(self.pipe.model, 'to_bettertransformer'):
                try:
                    self.pipe.model = self.pipe.model.to_bettertransformer()
                    print("✓ BetterTransformer optimization enabled")
                except Exception as e:
                    print(f"⚠ BetterTransformer not available")
            
            # Use torch.compile for even faster inference (PyTorch 2.0+)
            # Note: Requires Triton which is not available on Windows
            if device == "cuda" and hasattr(torch, 'compile') and sys.platform != 'win32':
                try:
                    # Only compile the encoder for stability
                    self.pipe.model.model.encoder = torch.compile(
                        self.pipe.model.model.encoder,
                        mode="reduce-overhead"
                    )
                    print("✓ Model compiled with torch.compile for maximum speed")
                except Exception as e:
                    print(f"⚠ torch.compile not available")
            elif device == "cuda" and sys.platform == 'win32':
                print("ℹ torch.compile skipped (requires Triton, not available on Windows)")
            
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # List available microphones
        print(f"\nAvailable microphones:")
        input_devices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                input_devices.append((i, info['name']))
                print(f"  [{i}] {info['name']}")
        
        # Get default input device
        try:
            default_input = self.audio.get_default_input_device_info()
            self.input_device_index = default_input['index']
            print(f"\n✓ Using default: [{self.input_device_index}] {default_input['name']}")
        except:
            # If no default, use first available
            if input_devices:
                self.input_device_index = input_devices[0][0]
                print(f"\n⚠ No default mic, using: [{self.input_device_index}] {input_devices[0][1]}")
            else:
                print("\n❌ No microphones found!")
                sys.exit(1)
        
        print("\n💡 To use a different microphone, modify 'input_device_index' in the code")
        print("\nPress Ctrl+Alt+V to start/stop recording")
        print("Press Ctrl+C to exit\n")
        print("Ready! Waiting for hotkey...\n")
    
    def start_recording(self):
        """Start recording audio from microphone"""
        self.is_recording = True
        self.audio_frames = []
        
        try:
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=self.input_device_index,  # Use selected device
                frames_per_buffer=self.CHUNK
            )
            
            device_info = self.audio.get_device_info_by_index(self.input_device_index)
            print(f"🎤 Recording... (Using: {device_info['name']})")
            print("   Press Ctrl+Alt+V again to stop")
            
            # Record in separate thread
            def record():
                while self.is_recording:
                    try:
                        data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                        self.audio_frames.append(data)
                    except Exception as e:
                        print(f"Recording error: {e}")
                        break
            
            self.record_thread = threading.Thread(target=record)
            self.record_thread.start()
            
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
            self.is_recording = False
    
    def stop_recording(self):
        """Stop recording and process audio"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if hasattr(self, 'record_thread'):
            self.record_thread.join()
        
        print("⏹ Stopping recording...")
        
        if not self.audio_frames:
            print("❌ No audio recorded")
            return
        
        # Process audio with VAD to remove silence
        audio_data = b''.join(self.audio_frames)
        if self.vad:
            audio_data = self.remove_silence(audio_data)
            if not audio_data:
                print("❌ No speech detected (only silence)")
                return
        
        # Save audio to temporary WAV file
        temp_wav = "temp_recording.wav"
        try:
            with wave.open(temp_wav, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(audio_data)
            
            duration = len(audio_data) / (self.CHANNELS * 2 * self.RATE)  # 2 bytes per sample
            print(f"Recorded {duration:.2f} seconds")
            
            # Debug: Check audio data
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_volume = np.abs(audio_array).mean()
            print(f"Audio volume: {audio_volume:.0f} (should be > 100 for clear speech)")
            
            # Transcribe
            print("🔄 Transcribing...")
            try:
                result = self.pipe(
                    temp_wav, 
                    return_timestamps=False
                )
                transcription = result['text'].strip()
                print(f"Raw result: {result}")  # Debug: show full result
            except Exception as e:
                if 'ffmpeg' in str(e).lower():
                    print("❌ ffmpeg not found. Please install it:")
                    print("   winget install ffmpeg")
                    print("   Then restart your terminal/PowerShell")
                    return
                raise
            
            if transcription:
                print(f"📝 Transcribed: {transcription}")
                print("⌨ Typing...")
                
                # Type the text
                self.keyboard_controller.type(transcription)
                
                print("✓ Done!\n")
            else:
                print("❌ No speech detected\n")
            
        except Exception as e:
            print(f"❌ Error processing audio: {e}\n")
        finally:
            # Clean up
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except:
                    pass
    
    def remove_silence(self, audio_data):
        """Remove silence from audio using VAD"""
        if not self.vad:
            return audio_data
        
        # VAD works with 10, 20, or 30ms frames
        frame_duration_ms = 30
        frame_size = int(self.RATE * frame_duration_ms / 1000) * 2  # 2 bytes per sample
        
        frames = []
        speech_frames = 0
        total_frames = 0
        
        for i in range(0, len(audio_data), frame_size):
            frame = audio_data[i:i + frame_size]
            if len(frame) < frame_size:
                # Pad last frame if needed
                frame = frame + b'\x00' * (frame_size - len(frame))
            
            total_frames += 1
            try:
                # Check if frame contains speech
                if self.vad.is_speech(frame, self.RATE):
                    frames.append(frame)
                    speech_frames += 1
            except:
                # If VAD fails, keep the frame
                frames.append(frame)
                speech_frames += 1
        
        if total_frames > 0:
            kept_percentage = (speech_frames / total_frames) * 100
            print(f"VAD: Kept {kept_percentage:.0f}% of audio ({speech_frames}/{total_frames} frames)")
        
        return b''.join(frames) if frames else b''
    
    def on_hotkey(self):
        """Handle hotkey press"""
        if self.is_recording:
            # Stop recording in a separate thread to avoid blocking
            threading.Thread(target=self.stop_recording).start()
        else:
            self.start_recording()
    
    def run(self):
        """Start the application"""
        # Register hotkey: Ctrl+Alt+V
        hotkeys = keyboard.GlobalHotKeys({
            '<ctrl>+<alt>+v': self.on_hotkey
        })
        hotkeys.start()
        
        # Keep running until interrupted
        try:
            while not self.should_exit:
                import time
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            hotkeys.stop()
    
    def cleanup(self):
        """Cleanup resources"""
        self.should_exit = True
        if self.is_recording:
            self.is_recording = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        if self.audio:
            self.audio.terminate()

def main():
    app = WiperTyper()
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting...")
    finally:
        app.cleanup()
        print("Goodbye!")

if __name__ == "__main__":
    main()
