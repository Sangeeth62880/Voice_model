import sys
import os

print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

modules = [
    ("speech_recognition", "SpeechRecognition"),
    ("torch", "PyTorch"), 
    ("numpy", "NumPy"),
    ("librosa", "Librosa"),
    ("soundfile", "SoundFile"),
    ("pydub", "PyDub")
]

print("\nTesting package imports:")
all_good = True
for module, name in modules:
    try:
        exec(f"import {module}")
        print(f"✓ {name} imported successfully")
    except ImportError as e:
        print(f"✗ {name} import error: {e}")
        all_good = False

if all_good:
    print("\n🎉 All packages imported successfully!")
else:
    print("\n❌ Some packages failed to import")

try:
    import pyaudio
    p = pyaudio.PyAudio()
    print(f"✓ Audio devices available: {p.get_device_count()}")
    p.terminate()
except Exception as e:
    print(f"✗ PyAudio error: {e}")
