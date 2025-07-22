import librosa
import numpy as np
import soundfile as sf
import os
from pathlib import Path
import json

class AudioPreparation:
    def __init__(self):
        self.sample_rate = 24000
        self.segment_length = 10  # 10 second segments
        
    def prepare_single_audio_file(self, input_audio_path):
        """Prepare the 5-minute Dashamoolam Damus audio file"""
        
        print(f"🎵 Processing audio file: {input_audio_path}")
        
        # Check if file exists
        if not os.path.exists(input_audio_path):
            print(f"❌ Audio file not found: {input_audio_path}")
            return False
            
        # Load audio
        try:
            audio, original_sr = librosa.load(input_audio_path, sr=None)
            print(f"✓ Original sample rate: {original_sr} Hz")
            print(f"✓ Original duration: {len(audio)/original_sr:.2f} seconds")
            
            # Resample to 24kHz if needed
            if original_sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.sample_rate)
                print(f"✓ Resampled to {self.sample_rate} Hz")
                
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            return False
            
        # Clean and enhance audio
        enhanced_audio = self.enhance_audio(audio)
        
        # Split into segments for training
        segments = self.split_into_segments(enhanced_audio)
        
        # Save processed segments
        self.save_audio_segments(segments, enhanced_audio)
        
        # Analyze voice characteristics
        voice_analysis = self.analyze_voice_characteristics(enhanced_audio)
        
        return voice_analysis
        
    def enhance_audio(self, audio):
        """Clean and enhance the audio quality"""
        print("🔧 Enhancing audio quality...")
        
        # Remove silence from beginning and end
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
        
        # Normalize audio level
        audio_normalized = librosa.util.normalize(audio_trimmed)
        
        # Reduce noise (simple spectral gating)
        # Get noise profile from first 0.5 seconds
        noise_sample = audio_normalized[:int(0.5 * self.sample_rate)]
        noise_power = np.mean(noise_sample ** 2)
        
        # Apply simple noise gate
        gate_threshold = noise_power * 3
        audio_gated = np.where(audio_normalized ** 2 > gate_threshold, 
                              audio_normalized, 
                              audio_normalized * 0.1)
        
        print("✓ Audio enhancement completed")
        return audio_gated
        
    def split_into_segments(self, audio):
        """Split 5-minute audio into 10-second training segments"""
        print("✂️  Splitting audio into training segments...")
        
        segment_samples = self.segment_length * self.sample_rate
        total_segments = len(audio) // segment_samples
        
        segments = []
        for i in range(total_segments):
            start_idx = i * segment_samples
            end_idx = start_idx + segment_samples
            segment = audio[start_idx:end_idx]
            
            # Only keep segments with sufficient energy (speech)
            if np.mean(segment ** 2) > 0.001:  # Energy threshold
                segments.append(segment)
                
        print(f"✓ Created {len(segments)} training segments")
        return segments
        
    def save_audio_segments(self, segments, full_audio):
        """Save processed audio segments"""
        # Create directories
        os.makedirs("audio_samples/dashamoolam_damus", exist_ok=True)
        os.makedirs("training_data/segments", exist_ok=True)
        
        # Save full enhanced audio
        full_path = "audio_samples/dashamoolam_damus/damus_full_enhanced.wav"
        sf.write(full_path, full_audio, self.sample_rate)
        print(f"✓ Saved enhanced full audio: {full_path}")
        
        # Save individual segments
        for i, segment in enumerate(segments):
            segment_path = f"training_data/segments/damus_segment_{i+1:03d}.wav"
            sf.write(segment_path, segment, self.sample_rate)
            
        print(f"✓ Saved {len(segments)} segments to training_data/segments/")
        
    def convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        else:
            return obj
        
    def analyze_voice_characteristics(self, audio):
        """Analyze Dashamoolam Damus voice characteristics"""
        print("🔍 Analyzing Dashamoolam Damus voice characteristics...")
        
        analysis = {}
        
        # 1. Pitch Analysis
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, fmin=50, fmax=400, sr=self.sample_rate
        )
        
        # Remove NaN values
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) > 0:
            analysis['pitch'] = {
                'mean_f0': float(np.mean(f0_clean)),
                'min_f0': float(np.min(f0_clean)),
                'max_f0': float(np.max(f0_clean)),
                'f0_std': float(np.std(f0_clean)),
                'voiced_percentage': float(np.mean(voiced_flag))
            }
        else:
            analysis['pitch'] = {'error': 'Could not extract pitch'}
            
        # 2. Spectral Features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
        
        analysis['spectral'] = {
            'centroid_mean': float(np.mean(spectral_centroids)),
            'centroid_std': float(np.std(spectral_centroids)),
            'rolloff_mean': float(np.mean(spectral_rolloff)),
            'bandwidth_mean': float(np.mean(spectral_bandwidth))
        }
        
        # 3. Voice Quality
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        analysis['voice_quality'] = {
            'mfcc_means': [float(x) for x in np.mean(mfcc, axis=1)],
            'mfcc_stds': [float(x) for x in np.std(mfcc, axis=1)]
        }
        
        # 4. Speaking Rate
        onset_frames = librosa.onset.onset_detect(y=audio, sr=self.sample_rate)
        speaking_rate = len(onset_frames) / (len(audio) / self.sample_rate)
        analysis['speaking_rate'] = float(speaking_rate)
        
        # 5. Energy and Rhythm
        energy = librosa.feature.rms(y=audio)[0]
        analysis['energy'] = {
            'mean_energy': float(np.mean(energy)),
            'energy_variance': float(np.var(energy))
        }
        
        # Convert all numpy types to Python native types
        analysis = self.convert_numpy_types(analysis)
        
        # Save analysis results
        with open("training_data/damus_voice_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
            
        print("✓ Voice analysis completed and saved")
        self.print_voice_summary(analysis)
        
        return analysis
        
    def print_voice_summary(self, analysis):
        """Print a summary of Dashamoolam Damus voice characteristics"""
        print("\n" + "="*50)
        print("🎭 DASHAMOOLAM DAMUS VOICE PROFILE")
        print("="*50)
        
        if 'pitch' in analysis and 'mean_f0' in analysis['pitch']:
            pitch = analysis['pitch']
            print(f"🎵 Pitch Range: {pitch['min_f0']:.1f} - {pitch['max_f0']:.1f} Hz")
            print(f"🎵 Average Pitch: {pitch['mean_f0']:.1f} Hz")
            print(f"🎵 Pitch Variation: {pitch['f0_std']:.1f} Hz")
            print(f"🎵 Voice Percentage: {pitch['voiced_percentage']*100:.1f}%")
            
        spectral = analysis['spectral']
        print(f"🔊 Voice Brightness: {spectral['centroid_mean']:.0f} Hz")
        print(f"🔊 Voice Richness: {spectral['rolloff_mean']:.0f} Hz")
        
        print(f"⏱️  Speaking Rate: {analysis['speaking_rate']:.1f} syllables/second")
        print(f"⚡ Voice Energy: {analysis['energy']['mean_energy']:.4f}")
        print("="*50)

def main():
    print("🎤 Step 1: Audio Preparation for Dashamoolam Damus Voice")
    print("="*60)
    
    # Check if we already have processed segments
    if os.path.exists("training_data/segments") and len(os.listdir("training_data/segments")) > 0:
        print("✓ Audio segments already exist. Analyzing existing audio...")
        
        # Load the enhanced audio and analyze
        enhanced_audio_path = "audio_samples/dashamoolam_damus/damus_full_enhanced.wav"
        if os.path.exists(enhanced_audio_path):
            audio, sr = librosa.load(enhanced_audio_path, sr=24000)
            processor = AudioPreparation()
            voice_analysis = processor.analyze_voice_characteristics(audio)
            
            if voice_analysis:
                print("\n🎉 Step 1 Completed Successfully!")
                print("\nFiles created:")
                print(f"✓ Enhanced audio: {enhanced_audio_path}")
                print(f"✓ Training segments: training_data/segments/ ({len(os.listdir('training_data/segments'))} files)")
                print("✓ Voice analysis: training_data/damus_voice_analysis.json")
                print("\n➡️  Ready for Step 2!")
                return
    
    # Get audio file path from user
    audio_file = input("📁 Enter the path to your 5-minute Dashamoolam Damus audio file: ").strip()
    
    if not audio_file:
        print("Using default path: audio_samples/damus_original.wav")
        audio_file = "audio_samples/damus_original.wav"
        
    processor = AudioPreparation()
    result = processor.prepare_single_audio_file(audio_file)
    
    if result:
        print("\n🎉 Step 1 Completed Successfully!")
        print("\nNext steps:")
        print("1. Check training_data/segments/ for audio segments")
        print("2. Review training_data/damus_voice_analysis.json for voice characteristics")
        print("3. Run Step 2 for feature extraction")
    else:
        print("\n❌ Step 1 failed. Please check your audio file and try again.")

if __name__ == "__main__":
    main()