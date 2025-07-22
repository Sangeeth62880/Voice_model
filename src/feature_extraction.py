import numpy as np
import librosa
import torch
import json
import os
from pathlib import Path
import pickle

class DamusFeatureExtractor:
    def __init__(self):
        self.sample_rate = 24000
        self.n_mfcc = 13
        self.n_fft = 1024
        self.hop_length = 256
        self.win_length = 1024
        
    def extract_segment_features(self, audio_path):
        """Extract detailed features from one audio segment"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            features = {}
            
            # 1. MFCC Features (voice timbre)
            mfcc = librosa.feature.mfcc(
                y=audio, sr=sr, n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, hop_length=self.hop_length
            )
            features['mfcc'] = mfcc
            
            # 2. Pitch (F0) features
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=50, fmax=400, sr=sr, hop_length=self.hop_length
            )
            # Fill NaN values with interpolation
            f0_clean = self.interpolate_f0(f0, voiced_flag)
            features['f0'] = f0_clean
            features['voiced'] = voiced_flag.astype(float)
            
            # 3. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio, sr=sr, hop_length=self.hop_length
            )[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio, sr=sr, hop_length=self.hop_length
            )[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio, sr=sr, hop_length=self.hop_length
            )[0]
            
            features['spectral_centroid'] = spectral_centroids
            features['spectral_rolloff'] = spectral_rolloff
            features['spectral_bandwidth'] = spectral_bandwidth
            
            # 4. Energy features
            energy = librosa.feature.rms(
                y=audio, hop_length=self.hop_length
            )[0]
            features['energy'] = energy
            
            # 5. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(
                audio, hop_length=self.hop_length
            )[0]
            features['zcr'] = zcr
            
            # 6. Mel-spectrogram (for synthesis target)
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=80,
                n_fft=self.n_fft, hop_length=self.hop_length
            )
            features['mel_spec'] = librosa.power_to_db(mel_spec)
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting features from {audio_path}: {e}")
            return None
    
    def interpolate_f0(self, f0, voiced_flag):
        """Interpolate missing F0 values"""
        f0_interp = f0.copy()
        
        # For unvoiced regions, use the mean F0 of voiced regions
        voiced_f0 = f0[voiced_flag]
        if len(voiced_f0) > 0:
            mean_f0 = np.nanmean(voiced_f0)
            f0_interp[np.isnan(f0_interp)] = mean_f0 * 0.5  # Lower for unvoiced
        else:
            f0_interp[np.isnan(f0_interp)] = 150.0  # Default pitch
            
        return f0_interp
    
    def process_all_segments(self):
        """Process all training segments and extract features"""
        segments_dir = "training_data/segments"
        features_dir = "training_data/features"
        
        os.makedirs(features_dir, exist_ok=True)
        
        segment_files = sorted(list(Path(segments_dir).glob("*.wav")))
        
        if not segment_files:
            print("❌ No audio segments found. Run Step 1 first.")
            return None
            
        print(f"🔧 Processing {len(segment_files)} audio segments...")
        
        all_features = []
        successful_segments = 0
        
        for i, segment_file in enumerate(segment_files):
            print(f"Processing segment {i+1}/{len(segment_files)}: {segment_file.name}")
            
            features = self.extract_segment_features(str(segment_file))
            
            if features is not None:
                # Save individual feature file (fixed path operation)
                feature_filename = f"features_{segment_file.stem}.pkl"
                feature_file = os.path.join(features_dir, feature_filename)
                
                with open(feature_file, 'wb') as f:
                    pickle.dump(features, f)
                
                all_features.append({
                    'segment_name': segment_file.name,
                    'features': features
                })
                successful_segments += 1
            
        print(f"✓ Successfully processed {successful_segments}/{len(segment_files)} segments")
        
        if successful_segments > 0:
            # Save combined features
            combined_features_file = "training_data/damus_features.pkl"
            with open(combined_features_file, 'wb') as f:
                pickle.dump(all_features, f)
            
            print(f"✓ Saved combined features to: {combined_features_file}")
            
            # Create training statistics
            stats = self.calculate_feature_statistics(all_features)
            
            return all_features, stats
        
        return None, None
    
    def calculate_feature_statistics(self, all_features):
        """Calculate statistics across all features for normalization"""
        print("📊 Calculating feature statistics...")
        
        stats = {}
        
        # Collect all feature values
        all_mfcc = []
        all_f0 = []
        all_energy = []
        all_spectral_centroid = []
        all_spectral_rolloff = []
        all_spectral_bandwidth = []
        all_zcr = []
        
        for segment_data in all_features:
            features = segment_data['features']
            
            all_mfcc.append(features['mfcc'])
            all_f0.extend(features['f0'])
            all_energy.extend(features['energy'])
            all_spectral_centroid.extend(features['spectral_centroid'])
            all_spectral_rolloff.extend(features['spectral_rolloff'])
            all_spectral_bandwidth.extend(features['spectral_bandwidth'])
            all_zcr.extend(features['zcr'])
        
        # Calculate statistics
        stats['mfcc'] = {
            'mean': np.mean(np.concatenate(all_mfcc, axis=1), axis=1),
            'std': np.std(np.concatenate(all_mfcc, axis=1), axis=1)
        }
        
        stats['f0'] = {
            'mean': np.mean(all_f0),
            'std': np.std(all_f0),
            'min': np.min(all_f0),
            'max': np.max(all_f0)
        }
        
        stats['energy'] = {
            'mean': np.mean(all_energy),
            'std': np.std(all_energy)
        }
        
        stats['spectral_centroid'] = {
            'mean': np.mean(all_spectral_centroid),
            'std': np.std(all_spectral_centroid)
        }
        
        stats['spectral_rolloff'] = {
            'mean': np.mean(all_spectral_rolloff),
            'std': np.std(all_spectral_rolloff)
        }
        
        stats['spectral_bandwidth'] = {
            'mean': np.mean(all_spectral_bandwidth),
            'std': np.std(all_spectral_bandwidth)
        }
        
        stats['zcr'] = {
            'mean': np.mean(all_zcr),
            'std': np.std(all_zcr)
        }
        
        # Save statistics
        stats_file = "training_data/feature_statistics.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_stats = {}
        for key, value in stats.items():
            if isinstance(value, dict):
                json_stats[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        json_stats[key][subkey] = subvalue.tolist()
                    else:
                        json_stats[key][subkey] = float(subvalue)
            else:
                json_stats[key] = value
        
        with open(stats_file, 'w') as f:
            json.dump(json_stats, f, indent=2)
        
        print(f"✓ Feature statistics saved to: {stats_file}")
        self.print_feature_summary(json_stats)
        
        return stats
    
    def print_feature_summary(self, stats):
        """Print summary of extracted features"""
        print("\n" + "="*60)
        print("🎯 DASHAMOOLAM DAMUS FEATURE ANALYSIS")
        print("="*60)
        
        print(f"🎵 Pitch Characteristics:")
        print(f"   Average F0: {stats['f0']['mean']:.1f} Hz")
        print(f"   F0 Range: {stats['f0']['min']:.1f} - {stats['f0']['max']:.1f} Hz")
        print(f"   F0 Variation: ±{stats['f0']['std']:.1f} Hz")
        
        print(f"\n🔊 Voice Quality:")
        print(f"   Brightness: {stats['spectral_centroid']['mean']:.0f} Hz")
        print(f"   Richness: {stats['spectral_rolloff']['mean']:.0f} Hz")
        print(f"   Bandwidth: {stats['spectral_bandwidth']['mean']:.0f} Hz")
        
        print(f"\n⚡ Energy Profile:")
        print(f"   Average Energy: {stats['energy']['mean']:.4f}")
        print(f"   Energy Variation: ±{stats['energy']['std']:.4f}")
        
        print(f"\n🎼 Voice Texture:")
        print(f"   Zero Crossing Rate: {stats['zcr']['mean']:.4f}")
        print(f"   MFCC Range: {len(stats['mfcc']['mean'])} coefficients")
        
        print("="*60)

def main():
    print("🎯 Step 2: Feature Extraction for Dashamoolam Damus Voice")
    print("="*60)
    
    # Check if Step 1 was completed
    if not os.path.exists("training_data/segments"):
        print("❌ No training segments found. Please complete Step 1 first.")
        return
    
    extractor = DamusFeatureExtractor()
    
    # Check if features already exist
    if os.path.exists("training_data/damus_features.pkl"):
        print("⚠️  Features already exist. Do you want to re-extract? (y/n): ", end="")
        response = input().strip().lower()
        if response != 'y':
            print("✓ Using existing features.")
            # Load and display existing statistics
            if os.path.exists("training_data/feature_statistics.json"):
                with open("training_data/feature_statistics.json", 'r') as f:
                    stats = json.load(f)
                extractor.print_feature_summary(stats)
            print("\n➡️  Ready for Step 3!")
            return
    
    # Extract features
    features, stats = extractor.process_all_segments()
    
    if features and stats:
        print("\n🎉 Step 2 Completed Successfully!")
        print("\nFiles created:")
        print("✓ Individual feature files: training_data/features/")
        print("✓ Combined features: training_data/damus_features.pkl")
        print("✓ Feature statistics: training_data/feature_statistics.json")
        print("\n➡️  Ready for Step 3: Voice Model Training!")
    else:
        print("\n❌ Step 2 failed. Please check the error messages above.")

if __name__ == "__main__":
    main()