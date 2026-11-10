# MOCKIN VIRTUAL CLOUD DRIVE (CLOUDFLARE R2) — API ENDPOINTS

Tài liệu đặc tả đầy đủ các API Endpoint của hệ thống **Ổ Cứng Ảo Cloudflare R2 & Ghi Chú Nhanh (`mockin.app/room`)** dùng để đẩy/kéo tệp tin, thư mục và văn bản giữa các máy (Máy A $\leftrightarrow$ Máy B) hoặc tích hợp cho AI Agent tự động thực thi.

---

## 1. Thông Tin Kết Nối & Xác Thực (Authentication)

- **Giao diện Web (UI):** `https://mockin.app/room`
- **API Base URL:** `https://api.mockin.app/api/room` *(hoặc `https://mockin.app/api/room`)*
- **Mật khẩu truy cập bắt buộc:** `9999`
- **Cách truyền mật khẩu** *(bắt buộc ở tất cả các endpoint, chọn 1 trong 2 cách)*:
  1. **HTTP Header (Khuyên dùng):**
     ```http
     X-Room-Code: 9999
     ```
  2. **Query Parameter:**
     ```text
     ?code=9999
     ```
  *(Nếu thiếu hoặc sai mật khẩu, mọi endpoint sẽ trả về `401 Unauthorized`).*

---

## 2. Danh Mục Tổng Hợp Các Endpoint

| STT | Nhóm | Method | Endpoint | Chức năng |
| :-- | :--- | :--- | :--- | :--- |
| 1 | **Room** | `GET` | `/api/room/list` | Liệt kê tất cả các Room (Ổ cứng ảo) đang có |
| 2 | **Room** | `POST` | `/api/room/create` | Tạo Room mới (mã tự sinh `R-XXXX` hoặc mã tự đặt) |
| 3 | **Room** | `GET` | `/api/room/{room_id}` | Lấy thông tin Room, Ghi chú nhanh và cây thư mục gốc |
| 4 | **Room** | `DELETE` | `/api/room/{room_id}` | Xóa toàn bộ Room và toàn bộ dữ liệu trên R2 |
| 5 | **Ghi chú (Multi-Note)** | `POST` | `/api/room/{room_id}/notes` | Tạo mới hoặc cập nhật 1 thẻ ghi chú (`id`, `title`, `text`) |
| 6 | **Ghi chú (Multi-Note)** | `DELETE` | `/api/room/{room_id}/notes/{note_id}` | Xóa 1 thẻ ghi chú theo `note_id` |
| 7 | **Ghi chú (Đơn)** | `POST` | `/api/room/{room_id}/text` | Đẩy/cập nhật nhanh ghi chú đầu tiên (tương thích ngược) |
| 8 | **Thư mục (`fs`)** | `GET` | `/api/room/{room_id}/fs` | Xem danh sách thư mục con & file tại đường dẫn `path` (`ls` / `tree`) |
| 9 | **Thư mục (`fs`)** | `POST` | `/api/room/{room_id}/fs/mkdir` | Tạo thư mục mới (hỗ trợ tạo nhiều cấp `a/b/c`) |
| 10 | **Thư mục (`fs`)** | `POST` | `/api/room/{room_id}/fs/copy` | Copy 1 file hoặc nguyên cả 1 thư mục (đệ quy) trực tiếp trên R2 |
| 11 | **Thư mục (`fs`)** | `POST` | `/api/room/{room_id}/fs/move` | Di chuyển hoặc Đổi tên 1 file / cả thư mục trực tiếp trên R2 |
| 12 | **Thư mục (`fs`)** | `DELETE` | `/api/room/{room_id}/fs` | Xóa 1 file, xóa hàng loạt, hoặc xóa đệ quy cả thư mục trên R2 |
| 13 | **Đẩy File (`push`)** | `POST` | `/api/room/{room_id}/files/presign` | Xin Presigned PUT URL để bắn thẳng file/thư mục lên Cloudflare R2 |
| 14 | **Đẩy File (`push`)** | `POST` | `/api/room/{room_id}/files/confirm` | Xác nhận hoàn tất sau khi đã `PUT` file lên Cloudflare R2 |
| 15 | **Đẩy File (`push`)** | `POST` | `/api/room/{room_id}/files` | Đẩy file bằng `multipart/form-data` (`curl -F`) vào thư mục bất kỳ |
| 16 | **Kéo File (`pull`)** | `GET` | `/api/room/{room_id}/fs/download` | Tải 1 file theo đường dẫn `path` (`302 Redirect` thẳng tới R2) |
| 17 | **Kéo Thư mục (`pull`)** | `GET` | `/api/room/{room_id}/fs/manifest` | Xuất danh sách toàn bộ file con + link R2 trực tiếp để kéo cả thư mục |

---

## 3. Chi Tiết Từng Endpoint

### 3.1. Liệt kê các Room (`GET /api/room/list`)
- **Request:**
  ```bash
  curl -s "https://api.mockin.app/api/room/list" -H "X-Room-Code: 9999"
  ```
- **Response `200 OK`:**
  ```json
  {
    "rooms": [
      {
        "room_id": "MAY-A",
        "title": "Room MAY-A",
        "updated_at": 1762699200.5,
        "created_at": 1762699000.0,
        "folder_count": 2,
        "file_count": 5,
        "total_size": 10485760,
        "text_preview": "Ghi chu nhanh...",
        "has_text": true
      }
    ],
    "r2_enabled": true
  }
  ```

---

### 3.2. Tạo hoặc Mở Room (`POST /api/room/create`)
- **Headers:** `Content-Type: application/json`, `X-Room-Code: 9999`
- **Request Body:** *(Nếu truyền `{}` sẽ tự sinh mã `R-XXXX`)*
  ```json
  {
    "room_id": "MAY-A",
    "title": "Ổ cứng ảo Máy A",
    "text": "Ghi chú khởi tạo (tùy chọn)"
  }
  ```

---

### 3.3. Xem Thư Mục & Tệp Tin (`GET /api/room/{room_id}/fs`)
- **Query Parameters:**
  - `path` *(tùy chọn, mặc định `""` là thư mục gốc `/`)*: Đường dẫn thư mục muốn xem, ví dụ `models/loras`.
  - `recursive` *(tùy chọn, mặc định `false`)*: Đặt `true` nếu muốn liệt kê toàn bộ cây thư mục con và file bên dưới `path`.
- **Request:**
  ```bash
  curl -s "https://api.mockin.app/api/room/MAY-A/fs?path=models" -H "X-Room-Code: 9999"
  ```
- **Response `200 OK`:**
  ```json
  {
    "room_id": "MAY-A",
    "title": "Room MAY-A",
    "text": "Nội dung ô ghi chú nhanh",
    "current_path": "models",
    "recursive": false,
    "folder_count": 2,
    "file_count": 1,
    "total_size": 524288,
    "r2_enabled": true,
    "folders": [
      {
        "name": "loras",
        "path": "models/loras",
        "parent_path": "models",
        "file_count": 1,
        "total_size": 524288,
        "updated_at": 1762699200.5
      }
    ],
    "files": []
  }
  ```

---

### 3.4. Tạo Thư Mục Mới (`POST /api/room/{room_id}/fs/mkdir`)
- **Headers:** `Content-Type: application/json`, `X-Room-Code: 9999`
- **Request Body:**
  ```json
  {
    "path": "models/loras/sdxl"
  }
  ```
- **Response `200 OK`:** Trả về trạng thái Room kèm `"created_folder": "models/loras/sdxl"`.

---

### 3.5. Copy Thư Mục Hoặc File Trên R2 (`POST /api/room/{room_id}/fs/copy`)
Sao chép trực tiếp trên Cloudflare R2 (hỗ trợ copy 1 file hoặc copy đệ quy nguyên cả 1 thư mục sang đường dẫn mới hoặc sang một Room khác).
- **Headers:** `Content-Type: application/json`, `X-Room-Code: 9999`
- **Request Body:**
  ```json
  {
    "src_path": "models/loras",
    "dest_path": "backup/loras_v1",
    "dest_room_id": "MAY-A"
  }
  ```
- **Response `200 OK`:** Trả về trạng thái Room kèm `"copied_count": <số_file_đã_copy>`.

---

### 3.6. Di Chuyển Hoặc Đổi Tên Thư Mục / File (`POST /api/room/{room_id}/fs/move`)
- **Headers:** `Content-Type: application/json`, `X-Room-Code: 9999`
- **Request Body:**
  ```json
  {
    "src_path": "models/loras/old_name.safetensors",
    "dest_path": "models/loras/new_name.safetensors"
  }
  ```

---

### 3.7. Xóa File Hoặc Xóa Cả Thư Mục (`DELETE /api/room/{room_id}/fs`)
- **Cách 1 — Xóa 1 file hoặc cả 1 thư mục qua Query `?path=`:**
  ```bash
  curl -X DELETE "https://api.mockin.app/api/room/MAY-A/fs?path=backup/loras_v1" \
    -H "X-Room-Code: 9999"
  ```
- **Cách 2 — Xóa hàng loạt nhiều đường dẫn qua JSON Body:**
  ```bash
  curl -X DELETE "https://api.mockin.app/api/room/MAY-A/fs" \
    -H "X-Room-Code: 9999" \
    -H "Content-Type: application/json" \
    -d '{"paths": ["models/a.bin", "backup/loras_v1"]}'
  ```

---

### 3.8. Đẩy File Lên Thư Mục Bất Kỳ

#### Cách A: Đẩy trực tiếp lên Cloudflare R2 bằng Presigned URL (Khuyên dùng cho mọi file lớn / nhiều file)
1. **Bước 1 — Xin Presigned PUT URL (`POST /api/room/{room_id}/files/presign`):**
   ```json
   {
     "base_dir": "models/loras",
     "files": [
       {
         "path": "flux_lora.safetensors",
         "size": 157286400,
         "content_type": "application/octet-stream"
       }
     ]
   }
   ```
   $\rightarrow$ Trả về `uploads[0].upload_url`.
2. **Bước 2 — Bắn thẳng dữ liệu file lên `upload_url` bằng `PUT`:**
   ```bash
   curl -X PUT "<upload_url>" \
     -H "Content-Type: application/octet-stream" \
     --upload-file "./flux_lora.safetensors"
   ```
3. **Bước 3 — Xác nhận hoàn tất (`POST /api/room/{room_id}/files/confirm`):**
   ```json
   {
     "base_dir": "models/loras",
     "files": [
       {
         "path": "flux_lora.safetensors",
         "size": 157286400,
         "content_type": "application/octet-stream"
       }
     ]
   }
   ```

#### Cách B: Đẩy nhanh 1 lệnh bằng `curl -F` (`POST /api/room/{room_id}/files?path=...`)
```bash
curl -X POST "https://api.mockin.app/api/room/MAY-A/files?path=models/loras" \
  -H "X-Room-Code: 9999" \
  -F "file=@./flux_lora.safetensors"
```

---

### 3.9. Kéo 1 File Hoặc Kéo Nguyên Cả Thư Mục Từ R2 Về Máy

#### A. Kéo 1 file theo đường dẫn (`GET /api/room/{room_id}/fs/download`)
Endpoint tự động trả về `302 Redirect` thẳng sang Cloudflare R2 Presigned URL:
```bash
curl -L "https://api.mockin.app/api/room/MAY-A/fs/download?path=models/loras/flux_lora.safetensors" \
  -H "X-Room-Code: 9999" \
  -o flux_lora.safetensors
```

#### B. Kéo nguyên cả 1 thư mục (`GET /api/room/{room_id}/fs/manifest`)
Trả về toàn bộ danh sách file con kèm đường dẫn tương đối (`relative_path`) và link tải R2 trực tiếp (`direct_download_url`):
```bash
curl -s "https://api.mockin.app/api/room/MAY-A/fs/manifest?path=models" \
  -H "X-Room-Code: 9999"
```
- **Response `200 OK`:**
  ```json
  {
    "room_id": "MAY-A",
    "folder_path": "models",
    "file_count": 1,
    "total_size": 157286400,
    "files": [
      {
        "path": "models/loras/flux_lora.safetensors",
        "relative_path": "loras/flux_lora.safetensors",
        "name": "flux_lora.safetensors",
        "size": 157286400,
        "uploaded_at": 1762699200.5,
        "direct_download_url": "https://<r2-presigned-url>...",
        "download_path": "/api/room/MAY-A/fs/download?path=models%2Floras%2Fflux_lora.safetensors"
      }
    ]
  }
  ```

---

### 3.10. Quản Lý Nhiều Ghi Chú Nhanh (Multi-Note)

#### A. Tạo mới hoặc cập nhật 1 thẻ ghi chú (`POST /api/room/{room_id}/notes`)
- **Truyền `id`:** Cập nhật thẻ ghi chú đó. Nếu bỏ trống `id`, hệ thống tự tạo thẻ ghi chú mới ở đầu danh sách.
```bash
curl -X POST "https://api.mockin.app/api/room/MAY-A/notes" \
  -H "X-Room-Code: 9999" \
  -H "Content-Type: application/json" \
  -d '{"title": "code", "text": "Nội dung ghi chú số 1"}'
```

#### B. Xóa 1 thẻ ghi chú (`DELETE /api/room/{room_id}/notes/{note_id}`)
```bash
curl -X DELETE "https://api.mockin.app/api/room/MAY-A/notes/n_a1b2c3d4" \
  -H "X-Room-Code: 9999"
```

#### C. Đẩy/Đọc nhanh ghi chú đơn (Tương thích ngược — `POST /api/room/{room_id}/text`)
- **Đẩy ghi chú lên Room:**
  ```bash
  curl -X POST "https://api.mockin.app/api/room/MAY-A/text" \
    -H "X-Room-Code: 9999" \
    -H "Content-Type: application/json" \
    -d '{"text": "Nội dung cần copy nhanh qua lại giữa các máy"}'
  ```
- **Đọc danh sách ghi chú từ Room:** Lấy mảng `.notes` (hoặc trường `.text`) từ `GET /api/room/MAY-A`.

---

## 4. Script PowerShell Mẫu Cho Agent Tự Động Đẩy / Kéo Cả Thư Mục

### 4.1. Đẩy nguyên 1 thư mục từ máy local lên Ổ Cứng Ảo R2
```powershell
$RoomId = "MAY-A"
$Pass = "9999"
$LocalFolder = "C:\ComfyUI\models\loras"
$RemoteBaseDir = "models/loras"

$Headers = @{ "X-Room-Code" = $Pass }
$ApiBase = "https://api.mockin.app/api/room/$RoomId"

$LocalFiles = Get-ChildItem -Path $LocalFolder -File -Recurse
$MetaList = @()
foreach ($f in $LocalFiles) {
    $rel = $f.FullName.Substring($LocalFolder.Length).TrimStart('\', '/').Replace('\', '/')
    $MetaList += @{ path = $rel; size = $f.Length; content_type = "application/octet-stream" }
}

# 1. Xin Presigned URL từ R2
$PresignBody = @{ base_dir = $RemoteBaseDir; files = $MetaList } | ConvertTo-Json -Depth 5
$PresignRes = Invoke-RestMethod -Method Post -Uri "$ApiBase/files/presign" -Headers $Headers -ContentType "application/json" -Body $PresignBody

# 2. Bắn từng file thẳng lên Cloudflare R2
for ($i = 0; $i -lt $LocalFiles.Count; $i++) {
    $fileObj = $LocalFiles[$i]
    $uploadInfo = $PresignRes.uploads[$i]
    curl.exe -s -X PUT $uploadInfo.upload_url -H "Content-Type: application/octet-stream" --upload-file $fileObj.FullName
}

# 3. Xác nhận hoàn tất
$ConfirmBody = @{ base_dir = ""; files = $PresignRes.uploads } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri "$ApiBase/files/confirm" -Headers $Headers -ContentType "application/json" -Body $ConfirmBody
```

### 4.2. Kéo nguyên 1 thư mục từ Ổ Cứng Ảo R2 về máy local
```powershell
$RoomId = "MAY-A"
$Pass = "9999"
$RemoteFolder = "models/loras"
$DestLocalDir = "C:\Pulled_Loras"

$Headers = @{ "X-Room-Code" = $Pass }
$Manifest = Invoke-RestMethod -Uri "https://api.mockin.app/api/room/$RoomId/fs/manifest?path=$RemoteFolder" -Headers $Headers

foreach ($f in $Manifest.files) {
    $outPath = Join-Path $DestLocalDir ($f.relative_path -replace '/', '\')
    $outDir = Split-Path $outPath -Parent
    if (!(Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir -Force | Out-Null }
    curl.exe -L $f.direct_download_url -o $outPath
}
```
