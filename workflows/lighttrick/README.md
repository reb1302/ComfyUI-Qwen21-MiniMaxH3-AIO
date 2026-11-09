# Lightricks LTX-2.5 & LTX-2.5 Creative Lab — Thư Mục Chuyên Dụng (`lighttrick`)

Thư mục này gom toàn bộ **25 Workflows chính thức & nâng cao**, **Script cài đặt phụ thuộc (`ComfyUI-LTXVideo`, `ComfyUI-KJNodes`, `pip`)**, và **Trình tải trọn bộ Core LTX-2.5 22B + 15 bộ IC-LoRA/LoRA Creative Lab** dành cho cả **Thư viện H200 VStone** và **Windows Portable**.

---

## 1. Cách Cài Đặt 1-Click Trên Server H200 (Không Cần Gõ Lệnh)

1. Mở ComfyUI trên H200 -> vào thư mục **`workflows/lighttrick/`** -> mở workflow:
   - **`00_Lightricks_Creative_Lab_1Click_Install_H200.json`**
2. **Bấm Queue lần 1:**
   - Node `stoneV_CloudSyncHub` với chế độ **`0B. Cài Lightricks Creative Lab đầy đủ (Pack + Core 22B + 15 Creative Lab IC-LoRAs, tự khởi động lại)`** sẽ tự động clone/cập nhật `Lightricks/ComfyUI-LTXVideo` và `kijai/ComfyUI-KJNodes`, cài đặt `requirements.txt` an toàn (không đè PyTorch), nạp nóng node và tự khởi động lại ComfyUI.
3. **Chờ 1–2 phút, nhấn F5 rồi bấm Queue lần 2:**
   - Hệ thống tự động tải bằng `aria2c` 16 luồng toàn bộ **Core LTX-2.5 22B** + **15 bộ IC-LoRA / LoRA Creative Lab** vào đúng các thư mục `models/diffusion_models`, `models/text_encoders`, `models/vae`, `models/latent_upscale_models`, `models/loras`, `models/embeddings`.
   - **Hỗ trợ cả Ungated Mirror (Không cần Token) lẫn HuggingFace Token:**
     - Để trống ô `custom_url`: Tự động tải qua các **Mirror Ungated HTTP 200** (`yuvraj108c/LTX-2.5`, `buckets/raubatz/ltx`, `Comfy-Org/gemma-4`).
     - Dán mã `hf_xxxx` vào ô `custom_url`: Tự động lưu `HF_TOKEN` vào `~/.cache/huggingface/token` và tải trực tiếp từ các repo Gated chính chủ `Lightricks/LTX-2.5-22b-*`.

---

## 2. Cách Chạy Script Cài Đặt Độc Lập (CLI)

### Trên Server H200 Linux:
```bash
bash lighttrick/download_lighttrick_h200.sh /workspace/ComfyUI
# Hoặc truyền kèm HF_TOKEN nếu đã Accept Terms trên HuggingFace:
bash lighttrick/download_lighttrick_h200.sh /workspace/ComfyUI hf_xxxxxxxxxxxxxxxxx
```

### Trên Windows Portable:
- Chạy trực tiếp file **`DOWNLOAD_LIGHTTRICK_ALL.bat`**, hoặc trong PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File .\lighttrick\download_lighttrick_models.ps1
```

---

## 3. Danh Sách 25 Workflows Trong `lighttrick/workflows/`

| STT | Tên File Workflow | Chức năng chính |
|---|---|---|
| 1 | `00_Lightricks_Creative_Lab_1Click_Install_H200.json` | **Cài đặt 1-Click** toàn bộ Pack + Core 22B + 15 Creative Lab IC-LoRAs trên H200 |
| 2 | `LTX25_Cai_Dat_1_Click.json` | Cài đặt nhanh bộ LTX-2.5 Outpaint tiêu chuẩn |
| 3 | `LTX-2.5_CreativeLab_All_In_One_V2V_Selector.json` | **All-in-One Creative Lab V2V** (Day-To-Night, Clean-Plate, Water, Deblur, Decompression, Colorization, SDR-To-HDR, Refine-Details, Restore, Alpha-Gen) |
| 4 | `LTX-2.5_I2V_Speed_Control_SlowMo_Cinemagraph.json` | **Creative Lab I2V:** Slow-Motion-Control (`-1.0` -> `1.0`) & Cinemagraph |
| 5 | `LTX-2.5_ICLoRA_Layout_To_Render_Two_Stage_Distilled.json` | **Creative Lab Layout-To-Render:** Biến bản dựng 3D Viewport / Clay thành video chân thực 2-Stage |
| 6 | `LTX25_Outpaint_Official.json` | Outpaint Video LTX-2.5 chuẩn Official |
| 7 | `stoneV_LTX25_Outpaint_TwoStage_Pro_H200.json` | Outpaint Video LTX-2.5 2-Stage Pro tích hợp `stoneV_LTX25_Outpaint_Loader` |
| 8 | `LTX-2.5_ICLoRA_Outpaint_Two_Stage_Distilled.json` | Official LTX-2.5 Outpaint 2-Stage Distilled |
| 9 | `LTX-2.5_ICLoRA_Inpaint_Two_Stage_Distilled.json` | Official LTX-2.5 Inpaint 2-Stage Distilled |
| 10 | `LTX-2.5_ICLoRA_Ingredients_Single_Stage_Distilled.json` | Official LTX-2.5 Ingredients (Ghép đa ảnh tham chiếu vào 1 cảnh video) |
| 11 | `LTX-2.5_ICLoRA_Motion_Track_Distilled.json` | Official LTX-2.5 Motion Track Control |
| 12 | `LTX-2.5_ICLoRA_Union_Control_Distilled.json` | Official LTX-2.5 Union Control (Depth / Pose / Canny / Edge) |
| 13 | `LTX-2.5_V2V_ICLoRA_Single_Stage_Distilled.json` | Official LTX-2.5 Video-to-Video IC-LoRA Single Stage |
| 14 | `LTX-2.5_V2V_TiledFusion_Native_4K_8K.json` | Official LTX-2.5 TiledFusion Native 4K/8K (`Day-To-Night` + `Refine-Details`) |
| 15 | `LTX-2.5_V2V_TiledFusion_Upscale.json` | Official LTX-2.5 TiledFusion Upscale (`Deblur` + `Refine-Details`) |
| 16 | `LTX-2.5_T2V_I2V_Single_Stage_Distilled.json` | Official LTX-2.5 Text/Image-to-Video Single Stage |
| 17 | `LTX-2.5_T2V_I2V_Two_Stage_Distilled.json` | Official LTX-2.5 Text/Image-to-Video Two Stage |
| 18 | `LTX-2.5_A2V_Two_Stage_Distilled.json` | Official LTX-2.5 Audio-to-Video Two Stage |
| 19 | `LTX-2.5_T2A_Single_Stage_Distilled.json` | Official LTX-2.5 Text-to-Audio Single Stage |
| 20 | `Official_EditAnything_LTX25.json` | EditAnything LTX-2.5 |
| 21 | `EditAnything_LTX25_With_Ref_Or_Motion_Transfer.json` | EditAnything LTX-2.5 kèm ảnh Ref / Motion Transfer |
| 22 | `AutoDetect_Wire_Object_Removal_SAM31_LTX25_BFS_AutoCrop.json` | Xóa dây cáp / vật thể tự động bằng SAM 3.1 + LTX-2.5 |
| 23 | `Biosal Clip Upscale VOSR2 SeedVR2.json` | Upscale Video kết hợp LTX-2.5 & SeedVR2 |
| 24 | `Official_Qwen_Image_2_1_Edit.json` | Sửa frame đầu bằng Qwen Image 2.1 trước khi đưa vào LTX-2.5 |
| 25 | `qwenedit.json` | Workflow phụ trợ Qwen Edit frame đầu |

---

## 4. Danh Mục Đầy Đủ Model Được Tải Vào Thư Viện H200

### A. Core Models (`diffusion_models`, `text_encoders`, `vae`, `latent_upscale_models`)
- `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` (~46 GB)
- `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` (~46 GB)
- `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` (~3.8 GB)
- `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` (~26.5 GB)
- `text_encoders/gemma4_e2b_it_bf16.safetensors` (~9.2 GB)
- `vae/ltx-2.5-video-vae-bf16.safetensors` (~2.38 GB)
- `vae/ltx-2.5-audio-vae-bf16.safetensors` (~215 MB)
- `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` (~950 MB)
- `latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` (~250 MB)

### B. 15 Bộ IC-LoRA & LoRA Trong `Lightricks/ltx-25-creative-lab` (`models/loras` & `models/embeddings`)
1. `ltx-2.5-22b-ic-lora-day-to-night-0.9.safetensors` (327 MB)
2. `ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors` (327 MB)
3. `ltx-2.5-22b-ic-lora-water-simulation-0.9.safetensors` (906 MB)
4. `ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors` (327 MB)
5. `ltx-2.5-22b-ic-lora-deblur-0.9.safetensors` (906 MB)
6. `ltx-2.5-22b-ic-lora-decompression-0.9.safetensors` (906 MB)
7. `ltx-2.5-22b-ic-lora-colorization-0.9.safetensors` (906 MB)
8. `ltx-2.5-22b-lora-cinemagraph-0.9.safetensors` (201 MB)
9. `ltx-2.5-22b-lora-slow-motion-control-1.0.safetensors` (655 MB)
10. `ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors` (1.31 GB)
11. `ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors` (1.31 GB) + `embeddings/ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors` (12.6 MB)
12. `ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors` (1.31 GB)
13. `ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors` (1.31 GB)
14. `ltx-2.5-22b-ic-lora-restore-1.0.safetensors` (1.71 GB)
15. `ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors` (1.31 GB)
*(Kèm đầy đủ các bản `ltx-2.3-22b-ic-lora-*` Ungated tương thích 100% với toàn bộ Official Workflows).*
