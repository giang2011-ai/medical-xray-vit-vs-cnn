import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# --- 1. ĐỊNH NGHĨA HÀM TRAIN (Dán cái hàm train_model của chúng ta vào đây) ---
def train_model(model, train_loader, val_loader, num_epochs, device, train_labels):
    # ... (Giữ nguyên nội dung hàm train_model mình đã bổ sung ở trên) ...
    pass 

# --- 2. HÀM MAIN ĐỂ CHẠY ---
if __name__ == "__main__":
    # Cấu hình thiết bị (Ưu tiên GPU nếu có)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang chạy trên: {device}")

    # Bước 1: Chuẩn bị Transforms (Biến đổi ảnh về cùng kích cỡ)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Bước 2: Load dữ liệu từ folder của bạn
    train_dataset = datasets.ImageFolder(root='train', transform=transform)
    val_dataset = datasets.ImageFolder(root='val', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Bước 3: Khởi tạo mô hình (Ví dụ: ResNet18)
    # Vì bạn đang làm X-quang phổi (2 lớp: Bình thường & Viêm phổi)
    model = models.resnet18(weights='DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # Thay đổi đầu ra thành 2 lớp

    # Bước 4: Gọi hàm train
    print("Bắt đầu huấn luyện...")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        device=device,
        train_labels=train_dataset.targets
    )
    
    print("Huấn luyện hoàn tất!")