import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

class DamusVoiceDataset(Dataset):
    """Dataset for Dashamoolam Damus voice training"""
    
    def __init__(self, features_file, stats_file):
        # Load features and statistics
        with open(features_file, 'rb') as f:
            self.all_features = pickle.load(f)
            
        with open(stats_file, 'r') as f:
            self.stats = json.load(f)
            
        self.prepare_training_data()
        
    def prepare_training_data(self):
        """Prepare features for training"""
        self.training_samples = []
        
        for segment_data in self.all_features:
            features = segment_data['features']
            
            # Get the minimum length across all feature types
            mel_frames = features['mel_spec'].shape[1]
            mfcc_frames = features['mfcc'].shape[1]
            f0_frames = len(features['f0'])
            
            min_frames = min(mel_frames, mfcc_frames, f0_frames)
            
            # Create input features (MFCC + F0 + spectral features)
            input_features = []
            
            # MFCC features (13 coefficients)
            mfcc_norm = self.normalize_mfcc(features['mfcc'][:, :min_frames])
            input_features.append(mfcc_norm)
            
            # F0 feature (1 value per frame)
            f0_norm = self.normalize_f0(features['f0'][:min_frames])
            input_features.append(f0_norm.reshape(1, -1))
            
            # Spectral features (3 features)
            spectral_features = np.array([
                features['spectral_centroid'][:min_frames],
                features['spectral_rolloff'][:min_frames],
                features['spectral_bandwidth'][:min_frames]
            ])
            spectral_norm = self.normalize_spectral(spectral_features)
            input_features.append(spectral_norm)
            
            # Energy and ZCR features (2 features)
            energy_zcr = np.array([
                features['energy'][:min_frames],
                features['zcr'][:min_frames]
            ])
            energy_norm = self.normalize_energy_zcr(energy_zcr)
            input_features.append(energy_norm)
            
            # Combine all input features (13 + 1 + 3 + 2 = 19 features per frame)
            combined_input = np.concatenate(input_features, axis=0)
            
            # Target output (mel-spectrogram - 80 mel bins)
            target_mel = features['mel_spec'][:, :min_frames]
            
            # Convert to torch tensors
            input_tensor = torch.FloatTensor(combined_input.T)  # Shape: (frames, 19)
            target_tensor = torch.FloatTensor(target_mel.T)    # Shape: (frames, 80)
            
            # Split into smaller sequences for training
            sequence_length = 64  # 64 frames per sequence
            
            for i in range(0, min_frames - sequence_length, sequence_length // 2):
                input_seq = input_tensor[i:i+sequence_length]
                target_seq = target_tensor[i:i+sequence_length]
                
                if input_seq.shape[0] == sequence_length:
                    self.training_samples.append({
                        'input': input_seq,
                        'target': target_seq,
                        'segment': segment_data['segment_name']
                    })
        
        print(f"✓ Created {len(self.training_samples)} training sequences")
    
    def normalize_mfcc(self, mfcc):
        """Normalize MFCC features"""
        mean = np.array(self.stats['mfcc']['mean']).reshape(-1, 1)
        std = np.array(self.stats['mfcc']['std']).reshape(-1, 1)
        return (mfcc - mean) / (std + 1e-8)
    
    def normalize_f0(self, f0):
        """Normalize F0 features"""
        mean = self.stats['f0']['mean']
        std = self.stats['f0']['std']
        return (f0 - mean) / (std + 1e-8)
    
    def normalize_spectral(self, spectral):
        """Normalize spectral features"""
        # Simple min-max normalization for spectral features
        return (spectral - spectral.min()) / (spectral.max() - spectral.min() + 1e-8)
    
    def normalize_energy_zcr(self, energy_zcr):
        """Normalize energy and ZCR features"""
        # Simple standardization
        mean = np.mean(energy_zcr, axis=1, keepdims=True)
        std = np.std(energy_zcr, axis=1, keepdims=True)
        return (energy_zcr - mean) / (std + 1e-8)
    
    def __len__(self):
        return len(self.training_samples)
    
    def __getitem__(self, idx):
        sample = self.training_samples[idx]
        return sample['input'], sample['target']

class DamusVoiceModel(nn.Module):
    """Neural network model for Dashamoolam Damus voice synthesis"""
    
    def __init__(self, input_dim=19, hidden_dim=256, output_dim=80):
        super(DamusVoiceModel, self).__init__()
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers for temporal modeling
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, dropout=0.3)
        
        # Voice characteristics embedding
        self.voice_embedding = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights"""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LSTM):
                for param in layer.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)
    
    def forward(self, x):
        # Input processing
        x = torch.relu(self.input_layer(x))
        
        # LSTM processing
        lstm_out, _ = self.lstm1(x)
        lstm_out, _ = self.lstm2(lstm_out)
        
        # Voice characteristics
        voice_embed = self.voice_embedding(lstm_out)
        
        # Combine features
        combined = torch.cat([lstm_out, voice_embed], dim=-1)
        
        # Generate output
        output = self.output_layer(combined)
        
        return output

class DamusVoiceTrainer:
    """Trainer for Dashamoolam Damus voice model"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'   Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train_model(self, train_loader, val_loader, epochs=50):
        """Train the complete model"""
        print(f"🚀 Starting training for {epochs} epochs...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 15
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 40)
            
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, 'models/dashamoolam_voice/best_model.pth')
                
                print("✓ New best model saved!")
                
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        print(f"\n🎉 Training completed! Best validation loss: {best_val_loss:.4f}")
        
        return self.train_losses, self.val_losses

def main():
    print("🤖 Step 3: Voice Model Creation and Training")
    print("="*60)
    
    # Check if previous steps completed
    features_file = "training_data/damus_features.pkl"
    stats_file = "training_data/feature_statistics.json"
    
    if not os.path.exists(features_file) or not os.path.exists(stats_file):
        print("❌ Missing files from previous steps. Please complete Steps 1 and 2 first.")
        return
    
    # Create model directory
    os.makedirs("models/dashamoolam_voice", exist_ok=True)
    
    # Check if model already exists
    model_file = "models/dashamoolam_voice/best_model.pth"
    if os.path.exists(model_file):
        print("⚠️  Trained model already exists. Do you want to retrain? (y/n): ", end="")
        response = input().strip().lower()
        if response != 'y':
            print("✓ Using existing model.")
            print(f"Model location: {model_file}")
            print("\n➡️  Ready for Step 4: Voice Synthesis Testing!")
            return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Create dataset
    print("📊 Loading training data...")
    dataset = DamusVoiceDataset(features_file, stats_file)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    print(f"✓ Training samples: {train_size}")
    print(f"✓ Validation samples: {val_size}")
    
    # Create model
    print("🏗️  Creating Dashamoolam Damus voice model...")
    model = DamusVoiceModel(input_dim=19, hidden_dim=256, output_dim=80)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model created with {total_params:,} total parameters")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = DamusVoiceTrainer(model, device)
    
    # Train model
    epochs = int(input("Enter number of training epochs (default 30): ") or "30")
    
    train_losses, val_losses = trainer.train_model(
        train_loader, val_loader, epochs=epochs
    )
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs': len(train_losses)
    }
    
    with open("models/dashamoolam_voice/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print("\n🎉 Step 3 Completed Successfully!")
    print("\nFiles created:")
    print("✓ Trained model: models/dashamoolam_voice/best_model.pth")
    print("✓ Training history: models/dashamoolam_voice/training_history.json")
    print("\n➡️  Ready for Step 4: Voice Synthesis Testing!")

if __name__ == "__main__":
    main()