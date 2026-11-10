---
trigger: always_on
---

# Quy tắc bất khả xâm phạm đối với Custom Nodes & Workflows của bên thứ ba (Third-Party)

1. **TUYỆT ĐỐI KHÔNG can thiệp hoặc sửa đổi thiết kế/code của Custom Nodes bên thứ ba:**
   - Khi cài đặt custom node của tác giả khác (ví dụ: `Lightricks/ComfyUI-LTXVideo`, `kijai/ComfyUI-KJNodes`, `ComfyUI-Nukomfy-Suite`, v.v.), **chỉ được phép clone/tải nguyên bản từ repository gốc và cài đặt các thư viện phụ thuộc (`requirements.txt`)**.
   - **CẤM** sửa code bên trong thư mục custom node của người khác.
   - **CẤM** viết các class node giả (stub/fallback/mock/shim) trùng tên để đè lên (`NODE_CLASS_MAPPINGS`) hoặc thay thế node gốc của tác giả khác.
   - Nếu node gốc bị lỗi thiếu thư viện hoặc thiếu file khi clone (ví dụ do Git LFS), phải giải quyết ở tầng cài đặt môi trường (`pip install`, `git-lfs`, cấu hình git clone) chứ tuyệt đối không được chế lại ruột node.

2. **TUYỆT ĐỐI KHÔNG sửa đổi hoặc làm biến dạng Workflow Original của người khác:**
   - Các file workflow gốc (`example_workflows` / workflow original của tác giả khác) phải được giữ nguyên bản 100%, không tự ý thêm/bớt/thay thế node hoặc trộn lẫn các workflow tự chế vào thư mục workflow gốc khi chưa có yêu cầu rõ ràng của người dùng.
