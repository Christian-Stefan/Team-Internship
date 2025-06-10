import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset,DataLoader

class SEBlock3D(nn.Module):
    def __init__(self, channels, reduction=8):  # try reduction=8 instead of 16
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y  # broadcasting

class ResidualSEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, drop_rate=0.2):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.se    = SEBlock3D(out_ch, reduction=8)
        self.drop  = nn.Dropout3d(p=drop_rate)

        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + identity
        out = self.drop(out)       # dropout after addition
        return self.relu(out)

class LocalBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # WIDER stem: 1→32 channels
        self.stem = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)  # D→D/2, H→H/2, W→W/2
        )
        # DEPTH: add two more blocks, channels doubling each time
        self.resblock1 = ResidualSEBlock(16, 32, stride=2)   # D/2→D/4
        self.resblock2 = ResidualSEBlock(32, 64, stride=2)  # D/4→D/8
        self.resblock3 = ResidualSEBlock(64, 128, stride=1) # D/8→D/8
        self.pool      = nn.AdaptiveAvgPool3d((1,1,1))       # → [B,128,1,1,1]

    def forward(self, x):
        x = self.stem(x)      # [B, 32, D/2, H/2, W/2]
        x = self.resblock1(x) # [B, 64, D/4, H/4, W/4]
        x = self.resblock2(x) # [B, 128, D/8, H/8, W/8]
        x = self.resblock3(x) # [B, 128, D/8, H/8, W/8]
        x = self.pool(x)      # [B, 128, 1,1,1]
        return x.view(x.size(0), -1)  # [B, 128]


class ContextBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1, bias=False),  # 1→32
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),           # (D,H,W)→(D/2,H/2,W/2)

            nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False), # 32→64
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),           # (D/2→D/4, H/2→H/4, W/2→W/4)

            nn.AdaptiveAvgPool3d((1,1,1))  # → [B, 64,1,1,1]
        )

    def forward(self, x):
        x = self.conv(x)
        return x.view(x.size(0), -1)  # [B, 64]
    
class RadiomicsBranch(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),      # LayerNorm can be more stable for small batches
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32)
        )
    def forward(self, x):
        return self.fc(x)  # [B,32]
    

class AttentionFusion(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.attn(x)
        return x * weights  # Element-wise attention
    
class TripleFusionModel(nn.Module):
    def __init__(self, num_classes, radiomics_dim=25):
        super().__init__()
        self.local_branch     = LocalBranch()           # output: [B,128]
        self.context_branch   = ContextBranch()         # output: [B,64]
        self.radiomics_branch = RadiomicsBranch(radiomics_dim)  # [B,32]

        fused_dim = 128 + 64 + 32  # = 224
        self.attn_fusion = AttentionFusion(fused_dim)

        # balance the local features with a learnable scalar
        self.alpha = nn.Parameter(torch.tensor(1.0))

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, volume_local, volume_context, radiomics):
        local_feat = self.local_branch(volume_local)         # [B,128]
        context_feat = self.context_branch(volume_context)   # [B,64]
        radio_feat = self.radiomics_branch(radiomics)        # [B,32]

        # scale local by alpha, then concatenate all three
        fused = torch.cat([self.alpha * local_feat, context_feat, radio_feat], dim=1)  # [B,224]
        fused_attn = self.attn_fusion(fused)  # [B,224], each feature reweighted
        out = self.classifier(fused_attn)     # [B,num_classes]
        return out
    
class TripleFusionDataset(Dataset):
    def __init__(self, all_locals, all_contexts, X_radiomics, labels_df):
        self.all_locals = all_locals
        self.all_contexts = all_contexts
        self.radiomics = X_radiomics
        self.labels = labels_df

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        local_volume = torch.tensor(self.all_locals[idx], dtype=torch.float32)
        context_volume = torch.tensor(self.all_contexts[idx], dtype=torch.float32)

        # Add channel dimension at dim=0 → shape becomes [1, D, H, W]
        sample = {
            "volume_local": local_volume.unsqueeze(0),
            "volume_context": context_volume.unsqueeze(0),
            "radiomics": torch.tensor(self.radiomics[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return sample



class ModelClassification:
    def __init__(self, modelpkl: str, volume_local, volume_context, radiomics, labels):
        self.modelpkl = modelpkl
        self.volume_local = volume_local
        self.volume_context = volume_context
        self.radiomics = radiomics
        self.labels = labels
        self.model = TripleFusionModel(num_classes=15, radiomics_dim=25)

    def get_result(self):
        dataset = TripleFusionDataset(
            all_locals=self.volume_local,
            all_contexts=self.volume_context,
            X_radiomics=self.radiomics,
            labels_df=self.labels 
        )

        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        state_dict = torch.load(self.modelpkl,weights_only=True)
        self.model.load_state_dict(state_dict)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        class_names = [
        "Active Infection",
        "Adenocarcinoma",
        "Adenoid Cystic Carcinoma",
        "Bronchioloalveolar Hyperplasia",
        "Carcinoid Tumors",
        "Granuloma",
        "Hamartoma",
        "Intrapulmonary Lymph Nodes",
        "Large Cell (Undifferentiated) Carcinoma",
        "Lymphoma",
        "Metastatic Tumors",
        "Sarcoidosis",
        "Sarcomatoid Carcinoma",
        "Small Cell Lung Cancer (SCLC)",
        "Squamous Cell Carcinoma",
        ]

        
        class_names = list(class_names)  # 0 to 14 inclusive
        self.__evaluate_model(data_loader, device, class_names)


    def __predict(self, dataloader, device):
        self.model.eval()
        topk_preds = [[] for _ in range(5)]
        topk_probs = [[] for _ in range(5)]
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                vol_local   = batch["volume_local"].to(device)
                vol_context = batch["volume_context"].to(device)
                radiomics   = batch["radiomics"].to(device)
                labels      = batch["label"].to(device)

                outputs = self.model(vol_local, vol_context, radiomics)
                probs = F.softmax(outputs, dim=1)
                top5 = torch.topk(probs, k=5, dim=1)

                for i in range(5):
                    topk_preds[i].extend(top5.indices[:, i].cpu().numpy())
                    topk_probs[i].extend(top5.values[:, i].cpu().numpy())

                all_labels.extend(labels.cpu().numpy())

        return {
            "top1_preds": np.array(topk_preds[0]),
            "top2_preds": np.array(topk_preds[1]),
            "top3_preds": np.array(topk_preds[2]),
            "top4_preds": np.array(topk_preds[3]),
            "top5_preds": np.array(topk_preds[4]),
            "top1_probs": np.array(topk_probs[0]),
            "top2_probs": np.array(topk_probs[1]),
            "top3_probs": np.array(topk_probs[2]),
            "top4_probs": np.array(topk_probs[3]),
            "top5_probs": np.array(topk_probs[4]),
            "labels":     np.array(all_labels)
        }

    def __evaluate_model(self, dataloader, device, class_names):
        results = self.__predict(dataloader, device)

        # Ensure we have at least one sample
        if len(results["labels"]) == 0:
            print("No samples found in dataloader!")
            return

        i = 0  # index of the single sample we want to evaluate

        y_true = results["labels"][i]
        top1 = results["top1_preds"][i]
        top2 = results["top2_preds"][i]
        top3 = results["top3_preds"][i]
        top4 = results["top4_preds"][i]
        top5 = results["top5_preds"][i]

        probs1 = results["top1_probs"][i]
        probs2 = results["top2_probs"][i]
        probs3 = results["top3_probs"][i]
        probs4 = results["top4_probs"][i]
        probs5 = results["top5_probs"][i]

        print(f"True: {class_names[y_true]}, "
            f"Top-1: {class_names[top1]} ({probs1*100:.1f}%), "
            f"Top-2: {class_names[top2]} ({probs2*100:.1f}%), "
            f"Top-3: {class_names[top3]} ({probs3*100:.1f}%)", end='')

        p3sum = probs1 + probs2 + probs3
        if p3sum < 0.75:
            print(f", Top-4: {class_names[top4]} ({probs4*100:.1f}%)", end='')
            p4sum = p3sum + probs4
            if p4sum < 0.85:
                print(f", Top-5: {class_names[top5]} ({probs5*100:.1f}%)", end='')
        print()  # Newline at end


            
