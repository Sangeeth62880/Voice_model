import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import os

def enhance_griffin_lim(mel_spec, sr=24000, n_iter=128):
    """Improved Griffin-Lim with better parameters"""
    
    # Convert mel to linear with better parameters
    linear_spec = librosa.feature.inverse.mel_to_stft(
        mel_spec, 
        sr=sr,
        n_fft=2048,      # Increased FFT size
        hop_length=512,  # Larger hop for better quality
        fmin=0,
        fmax=sr//2
    )
    
    # Enhanced Griffin-Lim
    audio = librosa.griffinlim(
        linear_spec,
        n_iter=128,        # Much more iterations
        hop_length=512,
        win_length=2048,
        momentum=0.99,     # Add momentum for stability
        init='random',     # Better initialization
        random_state=42
    )
    
    return audio

def post_process_audio(audio, sr=24000):
    """Post-process audio to reduce artifacts"""
    
    # 1. Remove DC offset
    audio = audio - np.mean(audio)
    
    # 2. Apply gentle low-pass filter to remove high-freq artifacts
    nyquist = sr / 2
    cutoff = 8000  # 8kHz cutoff
    b, a = signal.butter(4, cutoff / nyquist, 'low')
    audio = signal.filtfilt(b, a, audio)
    
    # 3. Apply gentle compression to smooth dynamics
    # Simple compression: reduce loud parts
    threshold = 0.7
    ratio = 4.0
    
    audio_abs = np.abs(audio)
    mask = audio_abs > threshold
    audio[mask] = np.sign(audio[mask]) * (
        threshold + (audio_abs[mask] - threshold) / ratio
    )
    
    # 4. Normalize
    audio = librosa.util.normalize(audio) * 0.8
    
    # 5. Apply gentle smoothing
    window_size = int(sr * 0.001)  # 1ms smoothing window
    audio = np.convolve(audio, np.ones(window_size)/window_size, mode='same')
    
    return audio

def fix_existing_audio_samples():
    """Fix your existing test samples"""
    
    print("🔧 Fixing existing audio samples...")
    
    input_dir = "test_outputs"
    output_dir = "test_outputs_enhanced"
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_dir):
        print(f"❌ Directory not found: {input_dir}")
        return
    
    audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    
    for audio_file in audio_files:
        input_path = os.path.join(input_dir, audio_file)
        output_path = os.path.join(output_dir, f"enhanced_{audio_file}")
        
        try:
            # Load audio
            audio, sr = librosa.load(input_path, sr=24000)
            
            # Apply post-processing
            enhanced_audio = post_process_audio(audio, sr)
            
            # Save enhanced version
            sf.write(output_path, enhanced_audio, sr)
            
            print(f"✅ Enhanced: {audio_file} → enhanced_{audio_file}")
            
        except Exception as e:
            print(f"❌ Failed to enhance {audio_file}: {e}")
    
    print(f"\n✓ Enhanced audio files saved to: {output_dir}")

def create_better_test_samples():
    """Create better test samples using your training segments"""
    
    print("🎵 Creating higher quality test samples...")
    
    segments_dir = "training_data/segments"
    output_dir = "test_outputs_high_quality"
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(segments_dir):
        print(f"❌ Segments directory not found: {segments_dir}")
        return
    
    segment_files = sorted([f for f in os.listdir(segments_dir) if f.endswith('.wav')])
    
    # Select best quality segments (middle ones usually have better audio)
    selected_segments = segment_files[5:8]  # Take segments 6, 7, 8
    
    for i, segment_file in enumerate(selected_segments):
        segment_path = os.path.join(segments_dir, segment_file)
        
        try:
            # Load segment
            audio, sr = librosa.load(segment_path, sr=24000)
            
            # Apply enhancement
            enhanced_audio = post_process_audio(audio, sr)
            
            # Create variations
            variations = []
            
            # Original enhanced
            variations.append(("original", enhanced_audio))
            
            # Slightly slower (more dramatic)
            slower = librosa.effects.time_stretch(enhanced_audio, rate=0.9)
            variations.append(("slower", slower))
            
            # Slightly higher pitch (more energetic)
            higher_pitch = librosa.effects.pitch_shift(enhanced_audio, sr=sr, n_steps=2)
            variations.append(("energetic", higher_pitch))
            
            # Save all variations
            for var_name, var_audio in variations:
                output_file = f"high_quality_damus_{i+1}_{var_name}.wav"
                output_path = os.path.join(output_dir, output_file)
                
                sf.write(output_path, var_audio, sr)
                print(f"✅ Created: {output_file}")
        
        except Exception as e:
            print(f"❌ Failed to process {segment_file}: {e}")
    
    print(f"\n✓ High quality samples saved to: {output_dir}")

def analyze_audio_quality(audio_path):
    """Analyze audio quality metrics"""
    
    audio, sr = librosa.load(audio_path, sr=24000)
    
    # Calculate quality metrics
    metrics = {}
    
    # 1. Signal-to-noise ratio estimate
    signal_power = np.mean(audio ** 2)
    metrics['signal_power'] = signal_power
    
    # 2. Dynamic range
    metrics['dynamic_range'] = np.max(audio) - np.min(audio)
    
    # 3. Spectral characteristics
    stft = librosa.stft(audio, n_fft=2048, hop_length=512)
    magnitude = np.abs(stft)
    
    # High frequency content (potential artifacts)
    total_energy = np.sum(magnitude)
    high_freq_energy = np.sum(magnitude[magnitude.shape[0]//2:])
    metrics['high_freq_ratio'] = high_freq_energy / total_energy
    
    # 4. Zero crossing rate (speech naturalness)
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    metrics['zero_crossing_rate'] = np.mean(zcr)
    
    # 5. Spectral centroid (brightness)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    metrics['brightness'] = np.mean(spectral_centroid)
    
    return metrics

def compare_audio_quality():
    """Compare quality of different audio versions"""
    
    print("📊 Comparing audio quality...")
    
    # Directories to compare
    dirs_to_check = [
        ("Original Segments", "training_data/segments"),
        ("Test Outputs", "test_outputs"),
        ("Enhanced", "test_outputs_enhanced"),
        ("High Quality", "test_outputs_high_quality")
    ]
    
    results = {}
    
    for dir_name, dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            audio_files = [f for f in os.listdir(dir_path) if f.endswith('.wav')]
            if audio_files:
                # Analyze first file as sample
                sample_file = os.path.join(dir_path, audio_files[0])
                try:
                    metrics = analyze_audio_quality(sample_file)
                    results[dir_name] = metrics
                    print(f"✓ Analyzed: {dir_name}")
                except Exception as e:
                    print(f"❌ Failed to analyze {dir_name}: {e}")
    
    # Print comparison
    print("\n" + "="*60)
    print("🎯 AUDIO QUALITY COMPARISON")
    print("="*60)
    
    for dir_name, metrics in results.items():
        print(f"\n📁 {dir_name}:")
        print(f"   Signal Power: {metrics['signal_power']:.4f}")
        print(f"   Dynamic Range: {metrics['dynamic_range']:.3f}")
        print(f"   Brightness: {metrics['brightness']:.0f} Hz")
        print(f"   High Freq Ratio: {metrics['high_freq_ratio']:.3f}")
        print(f"   Naturalness (ZCR): {metrics['zero_crossing_rate']:.4f}")

def main():
    print("🔧 Audio Quality Enhancement Tool")
    print("="*50)
    
    print("\n1️⃣ Fixing existing test samples...")
    fix_existing_audio_samples()
    
    print("\n2️⃣ Creating high-quality samples from segments...")
    create_better_test_samples()
    
    print("\n3️⃣ Analyzing quality differences...")
    compare_audio_quality()
    
    print("\n🎉 Audio enhancement complete!")
    print("\nNext steps:")
    print("1. Listen to files in 'test_outputs_enhanced/' and 'test_outputs_high_quality/'")
    print("2. Compare with original samples")
    print("3. The enhanced versions should sound much clearer!")

if __name__ == "__main__":
    main()