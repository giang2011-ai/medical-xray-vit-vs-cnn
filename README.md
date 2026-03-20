# Phân loại ảnh X-quang y tế mất cân bằng: So sánh ViT và CNN

## Mục tiêu bài tập
Dự án này được thực hiện nhằm đánh giá và so sánh khả năng trích xuất đặc trưng của kiến trúc Vision Transformer (ViT) và Mạng nơ-ron tích chập (CNN, sử dụng ResNet-50 làm baseline) trong bài toán phân loại ảnh X-quang y tế. 

Trong thực tế chẩn đoán lâm sàng, các bộ dữ liệu hình ảnh thường gặp tình trạng phân bố lớp bị mất cân bằng nghiêm trọng (số lượng ca bệnh ít hơn rất nhiều so với ca bình thường). Do đó, dự án tập trung vào việc áp dụng các kỹ thuật huấn luyện với dữ liệu mất cân bằng (như Weighted Cross Entropy) và sử dụng các thang đo đánh giá toàn diện hơn như F1-Score (Macro), Recall và AUC-ROC.

##  Thông tin thực hiện
- **Nhóm 11**
- **Gồm các thành viên:**
- *Lê Thị Khánh Linh*
- *Nguyễn Huyền Thương*
- *Trần Bùi Hà Giang*

##  Cấu trúc thư mục

medical-xray-vit-vs-cnn/
│
├── data/
│   ├── train/            # Dữ liệu huấn luyện (chia theo thư mục class)
│   ├── val/              # Dữ liệu validation trong quá trình train
│   └── test/             # Dữ liệu test độc lập
│
├── src/
│   ├── dataset.py        # Pipeline load dữ liệu và augmentation
│   ├── models.py         # Khởi tạo kiến trúc ResNet-50 và ViT
│   ├── train.py          # Vòng lặp huấn luyện, xử lý class weights
│   ├── evaluate.py       # Tính toán các metrics (F1, AUC, Recall)
│   └── utils.py          # Vẽ đồ thị (Loss, Accuracy, Confusion Matrix)
│
├── outputs/              # Lưu model weights (.pth) và biểu đồ (.png)
├── requirements.txt      # Danh sách các thư viện phụ thuộc
├── main.py               # Script thực thi chính
└── README.md             # Tài liệu hướng dẫn

##  Hướng dẫn cài đặt

**1. Clone repository về máy:**
\`\`\`bash
git clone https://github.com/giang2011-ai/medical-xray-vit-vs-cnn.git
cd medical-xray-vit-vs-cnn
\`\`\`

**2. Thiết lập môi trường ảo (Khuyến nghị):**
\`\`\`bash
python -m venv venv
source venv/bin/activate  # Trên Windows dùng: venv\Scripts\activate
\`\`\`

**3. Cài đặt các thư viện cần thiết:**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Dữ liệu (Dataset)

- Link tải dataset: 
- Sau khi tải, hãy đảm bảo dữ liệu được giải nén và cấu trúc đúng như phần **Cấu trúc thư mục** bên trên.

## Cách chạy mô hình

Bạn có thể chạy toàn bộ pipeline từ huấn luyện đến kiểm thử thông qua file `main.py`. (Script này sẽ tự động gọi các module trong `src/`).

Ví dụ lệnh chạy cơ bản (cần cấu hình argument parser trong `main.py`):
\`\`\`bash
python main.py --model resnet50 --epochs 20 --batch_size 32 --data_dir ./data
\`\`\`
hoặc đối với ViT:
\`\`\`bash
python main.py --model vit --epochs 20 --batch_size 16 --data_dir ./data
\`\`\`

Kết quả đánh giá và các biểu đồ (Confusion Matrix, ROC Curve) sẽ được tự động lưu vào thư mục `outputs/`.