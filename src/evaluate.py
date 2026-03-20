import torch
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score
import torch.nn.functional as F

def evaluate_model(model, dataloader, device, num_classes, target_names=None):
    """
    Chạy mô hình trên tập validation/test và tính toán các metrics.
    """
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []

    # Tắt tính toán gradient để tăng tốc độ và tiết kiệm bộ nhớ
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            
            # Tính xác suất bằng Softmax
            probs = F.softmax(outputs, dim=1)
            # Lấy class có xác suất cao nhất làm dự đoán
            preds = torch.argmax(probs, dim=1)

            # Đưa dữ liệu về CPU và chuyển thành numpy array để dùng sklearn
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # 1. Tính toán Accuracy và F1-Score (dùng macro để đánh giá công bằng các class nhỏ)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro') 

    # 2. Tính toán AUC-ROC
    # Xử lý linh hoạt cho bài toán nhị phân (Binary) hoặc đa lớp (Multiclass)
    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_probs[:, 1])
        else:
            auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
    except ValueError:
        auc = float('nan') # Đề phòng trường hợp batch/dataset thiếu class

    # In kết quả
    print("\n" + "="*40)
    print("BÁO CÁO ĐÁNH GIÁ (EVALUATION REPORT)")
    print("="*40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-Score (Macro): {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print("\nChi tiết Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

    return y_true, y_pred, y_probs, {"accuracy": acc, "f1": f1, "auc": auc}