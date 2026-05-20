import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

FRAMES_PATH = "/content/shanghaitech/SHANGHAI/SHANGHAI_TRAIN/frames"
LABEL_PATH = "/content/shanghaitech/SHANGHAI/SHANGHAI_TRAIN/label"
FEATURES_PATH = "/content/shanghaitech/features"
MODEL_SAVE_PATH = "/content/shanghaitech/resnet_lstm_best.pth"
CHECKPOINT_PATH = "/content/shanghaitech/resnet_lstm_checkpoint.pth"

IMAGE_SIZE = 112
SEQUENCE_LENGTH = 16
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2
PATIENCE = 7

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if os.path.exists(FRAMES_PATH) and os.path.exists(LABEL_PATH):
    scenes = sorted(os.listdir(FRAMES_PATH))
    print(f"Total scenes: {len(scenes)}")

    for scene in scenes[:3]:
        scene_frame_dir = os.path.join(FRAMES_PATH, scene)
        scene_label_file = os.path.join(LABEL_PATH, f"{scene}.npy")
        if os.path.exists(scene_frame_dir) and os.path.exists(scene_label_file):
            frames = sorted(os.listdir(scene_frame_dir))
            labels = np.load(scene_label_file)
            print(f"Scene {scene}:")
            print(f"  Frames : {len(frames)}")
            print(f"  Labels : {len(labels)}")
            print(f"  Anomaly: {labels.sum()} / {len(labels)}")

    os.makedirs(FEATURES_PATH, exist_ok=True)

    transform_extract = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    backbone = models.resnet18(pretrained=True)
    backbone.fc = nn.Identity()
    backbone = backbone.to(device)
    backbone.eval()

    scenes = sorted(os.listdir(FRAMES_PATH))
    for scene_idx, scene in enumerate(scenes):
        scene_frame_dir = os.path.join(FRAMES_PATH, scene)
        scene_label_file = os.path.join(LABEL_PATH, f"{scene}.npy")
        out_feat_file = os.path.join(FEATURES_PATH, f"{scene}_features.pt")
        out_label_file = os.path.join(FEATURES_PATH, f"{scene}_labels.npy")
        if os.path.exists(out_feat_file) and os.path.exists(out_label_file):
            print(f"Features & Labels already exist for scene {scene_idx + 1} / {len(scenes)}")
            continue
        if not os.path.exists(scene_label_file):
            print(f"Label file not found for scene {scene}. Skipping...")
            continue
        labels = np.load(scene_label_file)
        frames = sorted(os.listdir(scene_frame_dir))
        if len(frames) != len(labels):
            print(f"Frame/label count mismatch for scene {scene}")
            continue
        scene_features = []
        batch_imgs = []
        for i, frame_file in enumerate(frames):
            img = Image.open(os.path.join(scene_frame_dir, frame_file)).convert("RGB")
            batch_imgs.append(transform_extract(img))
            if len(batch_imgs) == 64 or i == len(frames) - 1:
                batch_tensor = torch.stack(batch_imgs).to(device)
                with torch.no_grad():
                    feats = backbone(batch_tensor)
                scene_features.append(feats.cpu())
                batch_imgs = []
        scene_features = torch.cat(scene_features, dim=0)
        torch.save(scene_features, out_feat_file)
        np.save(out_label_file, labels)
        print(f"Saved features & labels for {scene}")

    print("Feature extraction completed.")

class TemporalAnomalyDataset(Dataset):
    def __init__(self, features_path, sequence_length=16, step=4):
        self.sequences = []
        self.labels = []
        if not os.path.exists(features_path):
            return
        scenes = [f.split('_features.pt')[0] for f in os.listdir(features_path) if f.endswith('_features.pt')]
        for scene in scenes:
            feat_file = os.path.join(features_path, f"{scene}_features.pt")
            label_file = os.path.join(features_path, f"{scene}_labels.npy")
            if not os.path.exists(feat_file) or not os.path.exists(label_file):
                continue
            feats = torch.load(feat_file)
            labels = np.load(label_file)
            num_frames = feats.size(0)
            if num_frames < sequence_length:
                continue
            for start in range(0, num_frames - sequence_length + 1, step):
                end = start + sequence_length
                seq_feats = feats[start:end]
                seq_labels = labels[start:end]
                anomaly_in_seq = 1 if seq_labels.sum() > 0 else 0
                self.sequences.append(seq_feats)
                self.labels.append(anomaly_in_seq)
        print(f"Loaded {len(self.sequences)} sequences | Anomaly sequences: {sum(self.labels)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], torch.tensor(self.labels[idx], dtype=torch.long)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_out):
        scores = self.attn(rnn_out).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = torch.sum(rnn_out * weights, dim=1)
        return context, weights

class TemporalModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=2, num_classes=2, model_type='lstm', bidirectional=True, use_attention=True):
        super().__init__()
        self.model_type = model_type.lower()
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        
        if self.model_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.5 if num_layers > 1 else 0.0
            )
        else:
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.5 if num_layers > 1 else 0.0
            )
            
        direction_mult = 2 if bidirectional else 1
        rnn_out_dim = hidden_dim * direction_mult
        
        self.fc = nn.Sequential(
            nn.Linear(rnn_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        if use_attention:
            self.attention = TemporalAttention(rnn_out_dim)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        if self.use_attention:
            context, weights = self.attention(rnn_out)
            out = self.fc(context)
            return out, weights
        else:
            last_step = rnn_out[:, -1, :]
            out = self.fc(last_step)
            return out, None

def train_temporal_model():
    if not os.path.exists(FEATURES_PATH) or len(os.listdir(FEATURES_PATH)) == 0:
        print("Features directory is empty or does not exist. Skipping training.")
        return
        
    full_dataset = TemporalAnomalyDataset(FEATURES_PATH, sequence_length=SEQUENCE_LENGTH, step=4)
    if len(full_dataset) == 0:
        print("Dataset contains zero samples. Skipping.")
        return
        
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    model = TemporalModel(
        input_dim=512,
        hidden_dim=256,
        num_layers=2,
        num_classes=2,
        model_type='gru',
        bidirectional=True,
        use_attention=True
    ).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer_model = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer_model, T_max=NUM_EPOCHS)
    
    best_f1 = 0.0
    best_model_state = None
    epochs_no_improve = 0
    train_losses, val_losses, val_accuracies, val_f1s = [], [], [], []
    
    print("Beginning Sequential Training Loop...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer_model.zero_grad()
            outputs, _ = model(seqs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer_model.step()
            running_loss += loss.item() * seqs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        model.eval()
        running_val_loss = 0.0
        correct, total = 0, 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                outputs, _ = model(seqs)
                val_loss = loss_fn(outputs, labels)
                running_val_loss += val_loss.item() * seqs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        accuracy = correct / total
        val_accuracies.append(accuracy)
        
        tp = sum(p == 1 and l == 1 for p, l in zip(all_preds, all_labels))
        fp = sum(p == 1 and l == 0 for p, l in zip(all_preds, all_labels))
        fn = sum(p == 0 and l == 1 for p, l in zip(all_preds, all_labels))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        val_f1s.append(f1)
        
        scheduler.step()
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Acc: {accuracy*100:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(best_model_state, CHECKPOINT_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}.")
                break
                
    if best_model_state:
        print(f"Best sequential model saved with F1-score: {best_f1:.4f}")
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, MODEL_SAVE_PATH)
        
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1, 3, 2)
    plt.plot([a * 100 for a in val_accuracies], label='Val Accuracy (%)')
    plt.legend()
    plt.title('Accuracy')
    plt.subplot(1, 3, 3)
    plt.plot(val_f1s, label='Val F1')
    plt.legend()
    plt.title('F1 Score')
    plt.tight_layout()
    plt.savefig("/content/shanghaitech/metrics_plot.png")
    plt.show()

if __name__ == "__main__":
    train_temporal_model()