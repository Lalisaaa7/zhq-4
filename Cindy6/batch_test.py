import os
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, roc_curve

from data_loader_from_raw import parse_txt_file
from modules.model import GCN_with_MLP  # 根据你的模型结构调整导入
import matplotlib.pyplot as plt

def test_model(model, test_data, device, file_name):
    model.eval()
    loader = DataLoader(test_data, batch_size=1, shuffle=False)

    total, correct = 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        probs = F.softmax(out, dim=1)[:, 1]
        pred = out.argmax(dim=1)

        correct += (pred == batch.y).sum().item()
        total += batch.num_nodes

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
        all_probs.extend(probs.cpu().detach().numpy())

    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1)
    mcc = matthews_corrcoef(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    print(f"\n📁 File: {file_name}")
    print(f"✅ Test Accuracy: {accuracy:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")
    print(f"✅ MCC: {mcc:.4f}")
    print(f"✅ AUC: {auc:.4f}")

    # 可选：保存 ROC 曲线图
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {file_name}")
    plt.legend()
    plt.grid()
    os.makedirs("ROC_Curves", exist_ok=True)
    plt.savefig(f"ROC_Curves/ROC_{file_name}.png")
    plt.close()

    return accuracy, f1, mcc, auc


def batch_test(raw_folder, model_weight_path, device):
    results = []

    for filename in os.listdir(raw_folder):
        if filename.endswith(".txt") and "Test" in filename:
            file_path = os.path.join(raw_folder, filename)
            test_data = parse_txt_file(file_path)
            if len(test_data) == 0:
                print(f"❌ Skipping {filename}, no test data found.")
                continue

            feature_dim = test_data[0].x.shape[1]
            model = GCN_with_MLP(
                in_channels=feature_dim,
                hidden_channels=64,
                gcn_out_channels=64,
                mlp_hidden=64,
                out_classes=2
            ).to(device)

            if not os.path.exists(model_weight_path):
                print(f"❌ Model weights not found at {model_weight_path}")
                return

            model.load_state_dict(torch.load(model_weight_path))
            model.eval()

            acc, f1, mcc, auc = test_model(model, test_data, device, filename)
            results.append((filename, acc, f1, mcc, auc))

    print("\n📊 All Results Summary:")
    for r in results:
        print(f"{r[0]} -> Acc: {r[1]:.4f}, F1: {r[2]:.4f}, MCC: {r[3]:.4f}, AUC: {r[4]:.4f}")


if __name__ == '__main__':
    raw_folder = './Raw_data'
    model_weight_path = './Weights/best_model.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_test(raw_folder, model_weight_path, device)
