import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def train_model(model, train_loader, val_loader, num_epochs, device, train_labels):
    # 1. Xử lý data imbalance bằng Class Weights
    classes = np.unique(train_labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    model.to(device)
    best_val_loss = float('inf') # Để theo dõi và lưu mô hình tốt nhất

    for epoch in range(num_epochs):
        # --- PHASE: TRAINING ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # Tính độ chính xác tập train (tùy chọn)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct_train / total_train

        # --- PHASE: VALIDATION ---
        model.eval() # Chuyển sang chế độ đánh giá (tắt Dropout, BatchNorm...)
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad(): # Không tính gradient để tiết kiệm bộ nhớ và nhanh hơn
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct_val / total_val

        # In kết quả sau mỗi epoch
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  [Train] Loss: {epoch_train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  [Val]   Loss: {epoch_val_loss:.4f}, Acc: {val_acc:.2f}%")

        # Lưu lại mô hình nếu đạt Val Loss thấp nhất
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("  --> Đã lưu mô hình tốt nhất (Best Model Saved)")
        
        print("-" * 30)
    
    return model
    