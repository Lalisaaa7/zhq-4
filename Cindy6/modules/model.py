# modules/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score, matthews_corrcoef
from torch_geometric.nn import GINEConv
from torch_geometric.nn import global_add_pool
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, GATConv
import numpy as np
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class GCN_with_MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, gcn_out_channels, mlp_hidden=128, out_classes=2):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, gcn_out_channels)
        # Update MLP input size to in_channels + gcn_out_channels for concatenation
        self.mlp = nn.Sequential(
            nn.Linear(gcn_out_channels + in_channels, mlp_hidden),  # increased input dim
            nn.ReLU(),
            nn.Linear(mlp_hidden, out_classes)
        )
   
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(gcn_out_channels)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, edge_index):

        original_x = x  # preserve original ESM features
        x = self.gcn1(x, edge_index)
        x = F.relu(self.norm1(x))
        x = self.dropout(x)
        x = self.gcn2(x, edge_index)
        x = F.relu(self.norm2(x))
        x = self.dropout(x)
        x = torch.cat([x, original_x], dim=1)
        out = self.mlp(x)

        return out

class Enhanced_GCN_with_Attention(nn.Module):
    def __init__(self, in_channels, hidden_channels, gcn_out_channels, mlp_hidden=128, out_classes=2, heads=4):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.attn = GATConv(hidden_channels, hidden_channels, heads=heads, concat=True, dropout=0.2)
        self.norm1 = nn.LayerNorm(hidden_channels * heads)
        self.gcn2 = GCNConv(hidden_channels * heads, gcn_out_channels)
        self.norm2 = nn.LayerNorm(gcn_out_channels)
        self.dropout = nn.Dropout(0.3)

        self.mlp = nn.Sequential(
            nn.Linear(gcn_out_channels + in_channels, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, out_classes)
        )

    def forward(self, x, edge_index):
        original_x = x  # 保留原始 ESM 表征
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.attn(x, edge_index)
        x = self.norm1(x)
        x = self.dropout(x)
        x = self.gcn2(x, edge_index)
        x = F.relu(self.norm2(x))
        x = self.dropout(x)

        x = torch.cat([x, original_x], dim=1)
        out = self.mlp(x)
        return out


class GINE_with_MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, gine_out_channels, mlp_hidden=128, out_classes=2):
        super().__init__()
        self.gine1 = GINEConv(in_channels, hidden_channels)
        self.gine2 = GINEConv(hidden_channels, gine_out_channels)

        self.mlp = nn.Sequential(
            nn.Linear(gine_out_channels, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, out_classes)
        )

    def forward(self, x, edge_index, batch):
        x = self.gine1(x, edge_index)
        x = F.relu(x)
        x = self.gine2(x, edge_index)
        x = global_add_pool(x, batch)  # 聚合池化操作
        x = self.mlp(x)
        return x


# 训练函数
# 修改 train_model 函数，让它接受 feature_dim 参数
def train_model(balanced_data, feature_dim, device, gcn_out_channels=64):
    print("Training model with balanced dataset...")
    loader = DataLoader(balanced_data, batch_size=8, shuffle=True)
    model = Enhanced_GCN_with_Attention(
        in_channels=feature_dim,
        hidden_channels=64,
        gcn_out_channels=gcn_out_channels,
        mlp_hidden=128,
        out_classes=2
    ).to(device)

    # 设置正类的权重（例如，正类的权重设置为5倍）
    weight = torch.tensor([1.0, 10.0]).to(device)  # 正类加权5倍
    criterion = FocalLoss(alpha=1.0, gamma=2.0, weight=weight)  # 使用 Focal Loss

    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_acc = 0.0
    for epoch in range(200):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in loader:
            batch = batch.to(device)  # Move the batch to the correct device
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            pred = out.argmax(dim=1)
            correct += (pred == batch.y.to(device)).sum().item()  # Move batch.y to device
            total += batch.num_nodes
            total_loss += loss.item()

        acc = correct / total
        print(f"Epoch {epoch + 1}/400 - Loss: {total_loss / len(loader):.4f} - Acc: {acc:.4f}")

        # 保存最优模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'Weights/best_model.pt')  # Ensure path is correct
            print(f"Saved best model at epoch {epoch + 1} (Acc: {acc:.4f})")

    return model


def test_model(model, test_data, device):
    print("\n📊 Evaluating on test set...")
    model.eval()
    loader = DataLoader(test_data, batch_size=16, shuffle=False)

    total, correct = 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)  # logits
        probs = F.softmax(out, dim=1)[:, 1]  # 取正类概率
        pred = out.argmax(dim=1)

        correct += (pred == batch.y).sum().item()
        total += batch.num_nodes

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1)
    mcc = matthews_corrcoef(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0  # 防止仅含单类时报错

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"AUC: {auc:.4f}")

    # ROC 曲线可视化
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()

    return accuracy, f1, mcc, auc