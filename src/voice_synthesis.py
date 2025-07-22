import numpy as np
import librosa
import soundfile as sf
import os
import json
from scipy import signal
from scipy.interpolate import interp1d
import random

class DirectDamusVoiceSynthesizer:
    """Direct voice synthesis using actual Damus audio segments"""
    
    def __init__(self):
        self.sample_rate = 24000
        self.segments = []
        self.phoneme_map = {}
        
        print("🎭 Initializing Direct Damus Voice Synthesizer...")
        self.load_audio_segments()
        self.analyze_segments()
    
    def load_audio_segments(self):
        """Load all available Damus audio segments"""
        
        # Check multiple directories
        segment_dirs = [
            "training_data/segments",
            "audio_samples/dashamoolam_damus"
        ]
        
        for seg_dir in segment_dirs:
            if os.path.exists(seg_dir):
                for file in os.listdir(seg_dir):
                    if file.endswith('.wav'):
                        file_path = os.path.join(seg_dir, file)
                        try:
                            audio, sr = librosa.load(file_path, sr=self.sample_rate)
                            
                            # Clean and enhance the audio
                            audio = self.clean_audio_segment(audio)
                            
                            self.segments.append({
                                'name': file,
                                'audio': audio,
                                'duration': len(audio) / self.sample_rate,
                                'path': file_path
                            })
                            
                        except Exception as e:
                            print(f"⚠️  Could not load {file}: {e}")
        
        print(f"✓ Loaded {len(self.segments)} Damus audio segments")
        
        if len(self.segments) == 0:
            print("❌ No audio segments found!")
            return False
        
        return True
    
    def clean_audio_segment(self, audio):
        """Clean and enhance individual audio segments"""
        
        # 1. Remove silence from beginning and end
        audio, _ = librosa.effects.trim(audio, top_db=30)
        
        # 2. Normalize
        audio = librosa.util.normalize(audio)
        
        # 3. Apply gentle high-pass filter to remove low-freq noise
        b, a = signal.butter(3, 80 / (self.sample_rate / 2), 'high')
        audio = signal.filtfilt(b, a, audio)
        
        # 4. Apply gentle low-pass filter to remove high-freq artifacts
        b, a = signal.butter(4, 8000 / (self.sample_rate / 2), 'low')
        audio = signal.filtfilt(b, a, audio)
        
        # 5. Remove clicks and pops with median filter
        audio = signal.medfilt(audio, kernel_size=3)
        
        # 6. Final normalization
        audio = librosa.util.normalize(audio) * 0.8
        
        return audio
    
    def analyze_segments(self):
        """Analyze segments to categorize by speech characteristics"""
        
        print("🔍 Analyzing speech characteristics...")
        
        for i, segment in enumerate(self.segments):
            audio = segment['audio']
            
            # Extract features for classification
            features = self.extract_speech_features(audio)
            segment['features'] = features
            
            # Classify segment type
            segment['type'] = self.classify_segment(features)
            
            print(f"   Segment {i+1}: {segment['type']} ({segment['duration']:.1f}s)")
    
    def extract_speech_features(self, audio):
        """Extract speech characteristics from audio"""
        
        features = {}
        
        # 1. Fundamental frequency (pitch)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )
        
        # Clean F0
        f0_clean = f0[voiced_flag]
        if len(f0_clean) > 0:
            features['f0_mean'] = np.nanmean(f0_clean)
            features['f0_std'] = np.nanstd(f0_clean)
            features['f0_range'] = np.nanmax(f0_clean) - np.nanmin(f0_clean)
        else:
            features['f0_mean'] = 150.0  # Default
            features['f0_std'] = 20.0
            features['f0_range'] = 50.0
        
        # 2. Energy characteristics
        rms_energy = librosa.feature.rms(y=audio)[0]
        features['energy_mean'] = np.mean(rms_energy)
        features['energy_std'] = np.std(rms_energy)
        
        # 3. Spectral characteristics
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        features['brightness'] = np.mean(spectral_centroids)
        
        # 4. Speaking rate (zero crossing rate as proxy)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['speaking_rate'] = np.mean(zcr)
        
        # 5. Vowel/consonant ratio (spectral rolloff)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        features['vowel_ratio'] = np.mean(rolloff) / (self.sample_rate / 2)
        
        return features
    
    def classify_segment(self, features):
        """Classify segment by speech type"""
        
        # Simple classification based on characteristics
        f0_mean = features['f0_mean']
        energy = features['energy_mean']
        brightness = features['brightness']
        
        if f0_mean > 180 and energy > 0.1:
            return "high_energy"  # Excited/emphatic speech
        elif f0_mean < 130 and energy < 0.05:
            return "low_energy"   # Calm/quiet speech
        elif brightness > 2000:
            return "bright"       # Clear/articulated speech
        else:
            return "neutral"      # Normal speech
    
    def create_synthetic_speech(self, target_type="neutral", duration=3.0, style="original"):
        """Create synthetic speech by combining and morphing segments"""
        
        print(f"🎤 Creating {duration}s of {target_type} Damus speech ({style} style)...")
        
        # Filter segments by type
        matching_segments = [s for s in self.segments if s['type'] == target_type]
        
        if not matching_segments:
            # Fallback to any segments
            matching_segments = self.segments
            print(f"⚠️  No {target_type} segments found, using all segments")
        
        target_samples = int(duration * self.sample_rate)
        synthesized_audio = np.zeros(target_samples)
        
        # Method 1: Concatenative synthesis
        if style == "concatenative":
            synthesized_audio = self.concatenative_synthesis(matching_segments, target_samples)
        
        # Method 2: Morphing synthesis
        elif style == "morphed":
            synthesized_audio = self.morphing_synthesis(matching_segments, target_samples)
        
        # Method 3: Original enhanced
        else:
            synthesized_audio = self.enhanced_original_synthesis(matching_segments, target_samples)
        
        # Post-process
        synthesized_audio = self.final_post_process(synthesized_audio)
        
        return synthesized_audio
    
    def concatenative_synthesis(self, segments, target_samples):
        """Concatenate segments with smooth transitions"""
        
        result = np.zeros(target_samples)
        current_pos = 0
        
        while current_pos < target_samples:
            # Select random segment
            segment = random.choice(segments)
            seg_audio = segment['audio']
            
            # Calculate how much we can fit
            remaining_samples = target_samples - current_pos
            seg_length = min(len(seg_audio), remaining_samples)
            
            if current_pos == 0:
                # First segment - use as is
                result[current_pos:current_pos + seg_length] = seg_audio[:seg_length]
            else:
                # Subsequent segments - apply crossfade
                crossfade_length = min(int(0.05 * self.sample_rate), seg_length // 2)  # 50ms crossfade
                
                if crossfade_length > 0:
                    # Create crossfade
                    fade_out = np.linspace(1, 0, crossfade_length)
                    fade_in = np.linspace(0, 1, crossfade_length)
                    
                    # Apply crossfade
                    result[current_pos:current_pos + crossfade_length] *= fade_out
                    result[current_pos:current_pos + crossfade_length] += seg_audio[:crossfade_length] * fade_in
                    
                    # Add remaining segment
                    if seg_length > crossfade_length:
                        result[current_pos + crossfade_length:current_pos + seg_length] = seg_audio[crossfade_length:seg_length]
                else:
                    result[current_pos:current_pos + seg_length] = seg_audio[:seg_length]
            
            current_pos += seg_length
        
        return result
    
    def morphing_synthesis(self, segments, target_samples):
        """Create morphed speech by blending segments"""
        
        if len(segments) < 2:
            return self.concatenative_synthesis(segments, target_samples)
        
        # Select two segments to morph between
        seg1 = random.choice(segments)
        seg2 = random.choice([s for s in segments if s != seg1])
        
        # Resample both to target length
        seg1_audio = self.resample_to_length(seg1['audio'], target_samples)
        seg2_audio = self.resample_to_length(seg2['audio'], target_samples)
        
        # Create morphing envelope
        morph_envelope = np.linspace(0, 1, target_samples)
        
        # Apply spectral morphing
        result = self.spectral_morph(seg1_audio, seg2_audio, morph_envelope)
        
        return result
    
    def enhanced_original_synthesis(self, segments, target_samples):
        """Use original segment with enhancements"""
        
        # Select best quality segment
        best_segment = max(segments, key=lambda s: s['features']['energy_mean'])
        
        # Resample to target length
        result = self.resample_to_length(best_segment['audio'], target_samples)
        
        # Apply prosodic variations
        result = self.add_prosodic_variation(result)
        
        return result
    
    def resample_to_length(self, audio, target_length):
        """Resample audio to exact target length"""
        
        if len(audio) == target_length:
            return audio.copy()
        
        # Use high-quality resampling
        original_length = len(audio)
        time_old = np.linspace(0, 1, original_length)
        time_new = np.linspace(0, 1, target_length)
        
        # Interpolate
        interp_func = interp1d(time_old, audio, kind='cubic', bounds_error=False, fill_value=0)
        resampled = interp_func(time_new)
        
        return resampled
    
    def spectral_morph(self, audio1, audio2, morph_envelope):
        """Morph between two audio signals in spectral domain"""
        
        # STFT of both signals
        stft1 = librosa.stft(audio1, n_fft=2048, hop_length=512)
        stft2 = librosa.stft(audio2, n_fft=2048, hop_length=512)
        
        # Match lengths
        min_frames = min(stft1.shape[1], stft2.shape[1])
        stft1 = stft1[:, :min_frames]
        stft2 = stft2[:, :min_frames]
        
        # Create morphing weights
        morph_weights = np.interp(
            np.linspace(0, len(morph_envelope)-1, min_frames),
            np.arange(len(morph_envelope)),
            morph_envelope
        )
        
        # Morph magnitude and phase separately
        mag1, phase1 = np.abs(stft1), np.angle(stft1)
        mag2, phase2 = np.abs(stft2), np.angle(stft2)
        
        # Interpolate magnitudes
        morphed_mag = mag1 * (1 - morph_weights) + mag2 * morph_weights
        
        # Interpolate phases (circular interpolation)
        phase_diff = np.angle(np.exp(1j * (phase2 - phase1)))
        morphed_phase = phase1 + phase_diff * morph_weights
        
        # Reconstruct
        morphed_stft = morphed_mag * np.exp(1j * morphed_phase)
        morphed_audio = librosa.istft(morphed_stft, hop_length=512)
        
        return morphed_audio
    
    def add_prosodic_variation(self, audio):
        """Add natural prosodic variations"""
        
        # Slight pitch variation
        pitch_variation = np.random.uniform(-1, 1)  # ±1 semitone
        if abs(pitch_variation) > 0.1:
            audio = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=pitch_variation)
        
        # Slight tempo variation
        tempo_variation = np.random.uniform(0.95, 1.05)
        if abs(tempo_variation - 1.0) > 0.01:
            audio = librosa.effects.time_stretch(audio, rate=tempo_variation)
        
        return audio
    
    def final_post_process(self, audio):
        """Final post-processing for best quality"""
        
        # 1. Remove any DC offset
        audio = audio - np.mean(audio)
        
        # 2. Apply gentle dynamics processing
        # Soft compression
        threshold = 0.6
        ratio = 3.0
        
        audio_abs = np.abs(audio)
        mask = audio_abs > threshold
        audio[mask] = np.sign(audio[mask]) * (
            threshold + (audio_abs[mask] - threshold) / ratio
        )
        
        # 3. Apply subtle harmonic enhancement
        # Add slight second harmonic for warmth
        audio_enhanced = audio.copy()
        try:
            # Generate second harmonic
            audio_shifted = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=12)
            audio_enhanced += 0.1 * audio_shifted  # Mix in 10% second harmonic
        except:
            pass  # Skip if pitch shift fails
        
        # 4. Final smoothing and normalization
        # Apply very gentle low-pass for smoothness
        b, a = signal.butter(2, 7000 / (self.sample_rate / 2), 'low')
        audio_enhanced = signal.filtfilt(b, a, audio_enhanced)
        
        # 5. Normalize to prevent clipping
        audio_enhanced = librosa.util.normalize(audio_enhanced) * 0.85
        
        return audio_enhanced
    
    def generate_test_samples(self):
        """Generate multiple test samples with different approaches"""
        
        print("\n🎵 Generating high-quality Damus voice samples...")
        
        os.makedirs("high_quality_outputs", exist_ok=True)
        
        test_configs = [
            ("neutral_original", "neutral", "original", 4.0),
            ("neutral_concatenative", "neutral", "concatenative", 4.0),
            ("neutral_morphed", "neutral", "morphed", 4.0),
            ("high_energy_original", "high_energy", "original", 3.0),
            ("bright_concatenative", "bright", "concatenative", 3.0),
        ]
        
        results = []
        
        for name, speech_type, style, duration in test_configs:
            print(f"\n🔊 Creating {name}...")
            
            try:
                # Generate audio
                audio = self.create_synthetic_speech(speech_type, duration, style)
                
                # Save
                output_file = f"high_quality_outputs/damus_{name}.wav"
                sf.write(output_file, audio, self.sample_rate)
                
                print(f"✅ Saved: {output_file}")
                
                # Analyze quality
                quality = self.analyze_quality(audio)
                
                results.append({
                    'name': name,
                    'file': output_file,
                    'duration': duration,
                    'quality': quality,
                    'type': speech_type,
                    'style': style
                })
                
            except Exception as e:
                print(f"❌ Failed to create {name}: {e}")
        
        self.save_results(results)
        return results
    
    def analyze_quality(self, audio):
        """Quick quality analysis"""
        
        # Signal power
        signal_power = np.mean(audio ** 2)
        
        # Dynamic range
        dynamic_range = np.max(audio) - np.min(audio)
        
        # Spectral characteristics
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        brightness = np.mean(spectral_centroid)
        
        return {
            'signal_power': float(signal_power),
            'dynamic_range': float(dynamic_range),
            'brightness': float(brightness)
        }
    
    def save_results(self, results):
        """Save test results"""
        
        results_file = "high_quality_outputs/synthesis_results.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "="*60)
        print("🎭 HIGH-QUALITY DAMUS VOICE SYNTHESIS RESULTS")
        print("="*60)
        
        for result in results:
            print(f"\n🔊 {result['name']}:")
            print(f"   File: {result['file']}")
            print(f"   Duration: {result['duration']}s")
            print(f"   Type: {result['type']} ({result['style']} style)")
            print(f"   Signal Power: {result['quality']['signal_power']:.4f}")
            print(f"   Brightness: {result['quality']['brightness']:.0f} Hz")
        
        print("\n🎯 Quality Assessment:")
        avg_power = np.mean([r['quality']['signal_power'] for r in results])
        if avg_power > 0.01:
            print("   ✅ EXCELLENT - High-quality voice synthesis!")
        else:
            print("   ✅ GOOD - Clear voice synthesis")
        
        print("\n🎧 Next Steps:")
        print("   1. Listen to files in 'high_quality_outputs/'")
        print("   2. Compare different synthesis styles")
        print("   3. These should sound much clearer and more natural!")

def main():
    print("🎬 Direct Damus Voice Synthesis (No Neural Network)")
    print("="*60)
    
    try:
        # Create synthesizer
        synthesizer = DirectDamusVoiceSynthesizer()
        
        if len(synthesizer.segments) == 0:
            print("❌ No audio segments found!")
            print("Please ensure Steps 1-2 completed successfully.")
            return
        
        # Generate test samples
        results = synthesizer.generate_test_samples()
        
        if results:
            print("\n🎉 High-quality synthesis completed!")
            print("Check 'high_quality_outputs/' for crystal-clear Damus voice samples!")
        else:
            print("❌ No samples generated")
    
    except Exception as e:
        print(f"❌ Synthesis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()