# Quy tắc bắt buộc cho mọi AI agent trong repo này

Áp dụng cho Claude Code, Antigravity/Gemini và mọi agent khác. Người dùng ban hành ngày 2026-11-05, có hiệu lực từ nay về sau.

## Tình hình

- Sau 3 ngày, theo người dùng không có workflow nào do Claude và Gemini viết dùng được: 27 workflow MiniMax, 10 workflow Qwen và cả LTX.
- Đa số node tạo ra là do AI tự suy đoán (hallucinate), tưởng dùng được nhưng không dùng được, và gây rối cho người dùng.
- Lỗi lặp lại:
  - Không theo một style cụ thể.
  - Không theo quy tắc của node.
  - Quên nối dây.
  - Quên cài phụ thuộc.
  - Để ra nhiều parameter thừa.

## Được làm

- Tìm workflow, LoRA và model có sẵn.
- Tải workflow của người khác về và giữ nguyên.
- Cài phụ thuộc mà workflow đó cần.
- Thêm model cần thiết vào danh sách tải (`loras` trong `sync_manifest.json`).
- Nghiên cứu, trả lời câu hỏi, làm trợ lý.
- Viết 1 node helper, chỉ khi người dùng yêu cầu.

## Cấm

- Tạo workflow.
- Viết lại workflow của người khác.
- Tự viết node khi chưa được yêu cầu.
- Tự đặt tên node.
- Tự ý làm việc người dùng không yêu cầu.

## Khi được yêu cầu viết node

1. Hỏi người dùng node tên gì trước khi viết.
2. Chỉ viết đúng node được yêu cầu.
3. Theo đúng quy tắc node của ComfyUI và một style thống nhất.
4. Nối đủ dây và cài đủ phụ thuộc.
5. Chỉ để parameter thật sự cần.

## Môi trường

- ComfyUI chạy trên máy H200, port 8189. Laptop chỉ dùng để sửa code, không có GPU nên không chạy thử được.
- Người dùng không gõ lệnh được trên H200. Việc cài đặt phải đi qua node `stoneV_CloudSyncHub`.
- Chưa được người dùng xác nhận đã chạy thật trên H200 thì không được nói là dùng được.
- Nhiều agent cùng đẩy lên repo này: `git fetch` trước khi sửa và chỉ stage đúng file của mình.
