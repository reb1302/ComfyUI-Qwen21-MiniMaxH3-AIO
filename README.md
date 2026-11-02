# ComfyUI_Qwen21_MiniMaxH3_AIO (Gói All-in-One Cho NVIDIA H200 Linux)

Đây là **gói Custom Node kiêm bộ cài đặt tự động từ A-Z** dành cho máy chủ **NVIDIA H200 (141GB VRAM) chạy Linux**. Chỉ cần chép duy nhất thư mục `ComfyUI_Qwen21_MiniMaxH3_AIO` này lên server H200 và chạy 1 lệnh.

---

## 1. Hướng Dẫn Chạy 1 Lệnh Từ A-Z Trên Máy H200 (Linux)

Sau khi copy thư mục `ComfyUI_Qwen21_MiniMaxH3_AIO` lên server Linux H200:

```bash
cd ComfyUI_Qwen21_MiniMaxH3_AIO
chmod +x install_h200_all_in_one.sh

# Cách 1: Cài đặt từ A-Z + Tải 12 model chuẩn (86.26 GB) + Tự bật ComfyUI ở port 8188
./install_h200_all_in_one.sh --run

# Cách 2: Tận dụng 141GB VRAM của H200, tải thêm cả bộ Full Precision BF16 (+120 GB)
./install_h200_all_in_one.sh --bf16 --run
```

### Script `install_h200_all_in_one.sh` sẽ tự động làm gì từ A-Z?
1. **Cài gói hệ thống Linux:** `aria2`, `git`, `ffmpeg`, `python3-venv`, `libgl1`.
2. **Cài đặt ComfyUI (v0.43.0+) & PyTorch CUDA 13.0 (`cu130`):** Tạo `venv`, cài đặt `torch` bản `cu130`, `triton`, và `sageattention` tối ưu hóa riêng cho kiến trúc Hopper H200 (`sm_90a`).
3. **Cài đặt ComfyUI-Manager & Custom Node All-in-One:** Tự động gắn gói `ComfyUI_Qwen21_MiniMaxH3_AIO` vào `ComfyUI/custom_nodes/`.
4. **Tích hợp sẵn Workflows & Embeddings:** Tự động nạp 12 workflow chính thức + 10 file hiệu ứng `minimaxh3_*.safetensors` vào ComfyUI.
5. **Tải 12 Model cốt lõi (86.26 GB) tốc độ cao:** Dùng `aria2c -x 16 -s 16` (16 luồng song song, tự động resume).
6. **Tạo script khởi chạy `start_h200.sh`:** Chạy ComfyUI với cờ `--highvram --use-sage-attention --listen 0.0.0.0 --port 8188` để giữ toàn bộ model trên 141GB VRAM của H200, không bao giờ bị nghẽn cổ chai CPU offload.

---

## 2. Bộ 3 Node All-in-One Tích Hợp Trong Giao Diện ComfyUI

Khi mở ComfyUI (Chuột phải -> **`AIO H200 (Qwen 2.1 & MiniMax H3)`**), bạn sẽ có 3 Node All-in-One:

1. **`AIO H200 Model Downloader & Verifier` (`AIO_H200_ModelManager`)**
   - Cho phép kiểm tra và tải bổ sung bất kỳ gói model nào (`Standard INT8` hoặc `H200 Full BF16`) ngay từ giao diện web ComfyUI.
2. **`Qwen-Image 2.1 All-in-One (Image Edit / I2I)` (`Qwen21_AllInOne_I2I`)**
   - **Gộp toàn bộ 10+ node** của workflow Qwen-Image-2.1 vào **1 Node duy nhất**.
   - Chỉ cần nối `image_1`, `image_2`, nhập `prompt` (ví dụ: `"Keep the person in <image1>, wear the outfit from <image2>"`) -> Xuất thẳng ra ảnh `IMAGE` 2K hoàn chỉnh! Tự động tải model nếu trên máy chưa có.
3. **`MiniMax H3 All-in-One (I2V & R2V + Audio)` (`MiniMaxH3_AllInOne_Pipeline`)**
   - **Gộp toàn bộ cụm Loader + Text Encoder 32B + Video/Audio VAE + Turbo LoRA + SigmaShift + I2V/R2V Conditioning** vào **1 Node duy nhất**.
   - Hỗ trợ chuyển đổi tức thì giữa 2 chế độ `Image_to_Video (FL2VA)` và `Reference_to_Video (R2V)` cũng như chọn độ chính xác `pruned_int8_convrot` hoặc `pruned_bf16` trên H200.

---

## 3. Cấu Trúc Gói `ComfyUI_Qwen21_MiniMaxH3_AIO`

```text
ComfyUI_Qwen21_MiniMaxH3_AIO/
├── install_h200_all_in_one.sh   # Script cài đặt tự động từ A-Z trên Linux H200
├── __init__.py                  # Entrypoint tự động nạp node, workflows & embeddings
├── aio_nodes.py                 # Mã nguồn 3 Node All-in-One + Auto-Downloader
├── README.md                    # Tài liệu hướng dẫn này
├── embeddings/                  # [ĐÃ ĐÓNG GÓI SẴN] 10 file MiniMax-H3 Embeddings
└── workflows/                   # [ĐÃ ĐÓNG GÓI SẴN] 12 file Workflow JSON chính thức
    ├── qwen_image_2.1/
    └── minimax_h3/
```
