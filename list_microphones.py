"""List all available microphones"""
import pyaudio

audio = pyaudio.PyAudio()

print("Available audio devices:")
print("=" * 60)

for i in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(i)
    print(f"\nDevice {i}: {info['name']}")
    print(f"  Max Input Channels: {info['maxInputChannels']}")
    print(f"  Max Output Channels: {info['maxOutputChannels']}")
    print(f"  Default Sample Rate: {info['defaultSampleRate']}")
    if info['maxInputChannels'] > 0:
        print("  ✓ This is a microphone/input device")

print("\n" + "=" * 60)
print(f"\nDefault Input Device: {audio.get_default_input_device_info()['name']}")
print(f"Device Index: {audio.get_default_input_device_info()['index']}")

audio.terminate()
