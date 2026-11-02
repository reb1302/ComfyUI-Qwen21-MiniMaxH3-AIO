"""
VStone Node Helper Module (node_helper_vstone.py)
Chuyên dụng cho bài toán Screen X — Image-to-Image Extend (Outpainting mở rộng khung ngang) với Qwen-Image-2.1:
- Lấy chiều cao tấm ảnh (H) làm chuẩn.
- Node 1 (VStone_ExtendLeft_1xWidth): Mở rộng + bù qua TRÁI 1 lần chiều rộng (+1x Width Left).
- Node 2 (VStone_ExtendRight_1xWidth): Mở rộng + bù qua PHẢI 1 lần chiều rộng (+1x Width Right).
- Node 3 (VStone_ScreenX_ExtendBoth): Mở rộng Screen X tùy chỉnh cả Trái & Phải cùng lúc.
- Node 4 (VStone_SeamlessRestore): Ghép khôi phục 100% độ nét ảnh gốc tự động cho cả Trái, Phải hoặc 2 bên.
- Node 5 (VStone_Qwen21_Extend_AIO): Node All-in-One chạy Qwen-Image-2.1 mở rộng Trái / Phải / Screen X.
"""
from __future__ import annotations

import math
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = Any  # type: ignore
    F = None  # type: ignore


# ==============================================================================
# CORE HELPER FUNCTIONS (node_helper_vstone)
# ==============================================================================
def _create_side_pad(
    image: torch.Tensor,
    pad_w: int,
    side: str,
    fill_mode: str,
) -> torch.Tensor:
    """
    Tạo vùng ảnh bù (padding) bên trái ('left') hoặc bên phải ('right') với chiều cao giữ nguyên H
    và chiều rộng bằng pad_w.
    """
    B, H, W, C = image.shape
    device = image.device
    dtype = image.dtype

    if pad_w <= 0:
        return torch.empty((B, H, 0, C), dtype=dtype, device=device)

    if fill_mode.startswith("gray_0.5"):
        pad = torch.full((B, H, pad_w, C), 0.5, dtype=dtype, device=device)
    elif fill_mode.startswith("mirror_blur") or fill_mode.startswith("mirror_sharp"):
        # Tạo dải lật gương liên tục để khớp mép trái hoặc mép phải
        flipped = torch.flip(image, dims=[2])
        repeats = math.ceil(pad_w / W) + 1
        tiles = []
        for i in range(repeats):
            tiles.append(flipped if i % 2 == 0 else image)

        if side == "left":
            # Mép phải của vùng bù trái phải khớp với cột x=0 của ảnh gốc (tức cột cuối của flipped)
            tiled = torch.cat(list(reversed(tiles)), dim=2)
            pad = tiled[:, :, -pad_w:, :].clone()
        else:
            # Mép trái của vùng bù phải phải khớp với cột x=W-1 của ảnh gốc (tức cột đầu của flipped)
            tiled = torch.cat(tiles, dim=2)
            pad = tiled[:, :, :pad_w, :].clone()

        if fill_mode.startswith("mirror_blur"):
            k = max(3, (min(H, pad_w) // 8) | 1)
            pad_nchw = pad.movedim(-1, 1)
            low_res = F.avg_pool2d(pad_nchw, kernel_size=k, stride=max(1, k // 2), padding=k // 2)
            pad = F.interpolate(low_res, size=(H, pad_w), mode="bilinear", align_corners=False).movedim(1, -1)
    elif fill_mode.startswith("edge_stretch"):
        edge_col = image[:, :, :1, :] if side == "left" else image[:, :, -1:, :]
        pad = edge_col.expand(B, H, pad_w, C).clone()
    elif fill_mode.startswith("white"):
        pad = torch.ones((B, H, pad_w, C), dtype=dtype, device=device)
    elif fill_mode.startswith("black"):
        pad = torch.zeros((B, H, pad_w, C), dtype=dtype, device=device)
    else:
        pad = torch.full((B, H, pad_w, C), 0.5, dtype=dtype, device=device)

    if C == 4 and fill_mode.startswith("gray_0.5"):
        pad[:, :, :, 3] = 1.0

    return pad


def extend_image_horizontal_tensor(
    image: torch.Tensor,
    add_left_widths: float = 0.0,
    add_right_widths: float = 0.0,
    extra_pixels_left: int = 0,
    extra_pixels_right: int = 0,
    standard_height: int = 0,
    reference_width_image: torch.Tensor | None = None,
    previous_mask: torch.Tensor | None = None,
    fill_mode: str = "mirror_blur",
    feather_pixels: int = 48,
    overlap_pixels: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """
    Hàm lõi Screen X:
    Lấy chiều cao tấm ảnh làm chuẩn (H) và mở rộng + bù vào bên TRÁI và/hoặc bên PHẢI
    theo bội số chiều rộng của tấm ảnh.

    Returns:
        (extended_image [B, H, new_W, C], extend_mask [B, H, new_W], normalized_orig [B, H, W, C], new_W, H)
    """
    if image.ndim != 4:
        raise ValueError(f"Expected 4D IMAGE tensor [B, H, W, C], got shape {tuple(image.shape)}")

    B, orig_H, orig_W, C = image.shape
    device = image.device
    dtype = image.dtype

    # 1. Lấy chiều cao tấm ảnh làm chuẩn (nếu standard_height > 0 thì chuẩn hóa theo chiều cao đó)
    if standard_height > 0 and standard_height != orig_H:
        scale = float(standard_height) / float(orig_H)
        norm_H = int(standard_height)
        norm_W = max(32, int(round(orig_W * scale)))
        img_nchw = image.movedim(-1, 1)
        norm_image = F.interpolate(img_nchw, size=(norm_H, norm_W), mode="bicubic", align_corners=False).movedim(1, -1).clamp(0.0, 1.0)
    else:
        norm_H = orig_H
        norm_W = orig_W
        norm_image = image

    # 2. Xác định 1 đơn vị chiều rộng chuẩn (1x Width)
    if reference_width_image is not None and reference_width_image.ndim == 4:
        ref_H = int(reference_width_image.shape[1])
        ref_W = int(reference_width_image.shape[2])
        unit_W = int(round(ref_W * (float(norm_H) / float(ref_H)))) if ref_H > 0 else norm_W
    else:
        unit_W = norm_W

    # 3. Tính số pixel cần bù qua Trái và qua Phải
    pad_left_w = max(0, int(round(unit_W * float(add_left_widths))) + int(extra_pixels_left))
    pad_right_w = max(0, int(round(unit_W * float(add_right_widths))) + int(extra_pixels_right))

    # 4. Tạo vùng ảnh bù bên Trái và bên Phải cùng chiều cao chuẩn norm_H
    left_pad = _create_side_pad(norm_image, pad_left_w, side="left", fill_mode=fill_mode)
    right_pad = _create_side_pad(norm_image, pad_right_w, side="right", fill_mode=fill_mode)

    # Ghép [Khung bù Trái | Ảnh gốc ở giữa | Khung bù Phải]
    extended_image = torch.cat([left_pad, norm_image, right_pad], dim=2)
    new_W = pad_left_w + norm_W + pad_right_w

    # 5. Tạo mặt nạ vùng mở rộng (extend_mask):
    # 0.0 = Vùng ảnh gốc cần giữ nguyên 100%
    # 1.0 = Vùng khung mới kéo rộng ra cần Qwen-Image-2.1 vẽ bù vào
    if previous_mask is not None:
        pm = previous_mask.to(device=device, dtype=dtype)
        if pm.ndim == 2:
            pm = pm.unsqueeze(0)
        if pm.shape[-2:] != (norm_H, norm_W):
            pm = F.interpolate(pm.unsqueeze(1), size=(norm_H, norm_W), mode="bilinear", align_corners=False).squeeze(1)
        mask_center = pm.expand(B, norm_H, norm_W).clone()
    else:
        mask_center = torch.zeros((B, norm_H, norm_W), dtype=dtype, device=device)

    mask_left = torch.ones((B, norm_H, pad_left_w), dtype=dtype, device=device) if pad_left_w > 0 else torch.empty((B, norm_H, 0), dtype=dtype, device=device)
    mask_right = torch.ones((B, norm_H, pad_right_w), dtype=dtype, device=device) if pad_right_w > 0 else torch.empty((B, norm_H, 0), dtype=dtype, device=device)
    mask = torch.cat([mask_left, mask_center, mask_right], dim=2)

    # Làm mềm biên nối bên TRÁI (từ 1.0 ở vùng bù trái giảm mượt về 0.0 khi vào ảnh gốc)
    if pad_left_w > 0:
        left_seam_end = min(new_W, pad_left_w + int(overlap_pixels))
        mask[:, :, :left_seam_end] = 1.0
        if feather_pixels > 0 and left_seam_end < new_W:
            f_len = min(int(feather_pixels), new_W - left_seam_end)
            if f_len > 0:
                t = torch.linspace(0.0, 1.0, steps=f_len, dtype=dtype, device=device)
                ramp_down = 0.5 * (1.0 + torch.cos(math.pi * t))  # 1.0 -> 0.0
                cur = mask[:, :, left_seam_end : left_seam_end + f_len]
                mask[:, :, left_seam_end : left_seam_end + f_len] = torch.maximum(cur, ramp_down.view(1, 1, f_len))

    # Làm mềm biên nối bên PHẢI (từ 0.0 ở ảnh gốc tăng mượt lên 1.0 khi sang vùng bù phải)
    if pad_right_w > 0:
        right_seam_start = max(0, pad_left_w + norm_W - int(overlap_pixels))
        mask[:, :, right_seam_start:] = 1.0
        if feather_pixels > 0 and right_seam_start > 0:
            f_len = min(int(feather_pixels), right_seam_start)
            if f_len > 0:
                t = torch.linspace(0.0, 1.0, steps=f_len, dtype=dtype, device=device)
                ramp_up = 0.5 * (1.0 - torch.cos(math.pi * t))  # 0.0 -> 1.0
                cur = mask[:, :, right_seam_start - f_len : right_seam_start]
                mask[:, :, right_seam_start - f_len : right_seam_start] = torch.maximum(cur, ramp_up.view(1, 1, f_len))

    return extended_image, mask, norm_image, new_W, norm_H


def extend_image_left_tensor(
    image: torch.Tensor,
    frames_left: float = 1.0,
    extra_pixels_left: int = 0,
    fill_mode: str = "mirror_blur",
    feather_pixels: int = 48,
    overlap_pixels: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Nhận 1 ảnh đầu vào [B, H, W, C], lấy chiều cao H làm chuẩn và mở rộng qua TRÁI
    (mặc định +1.0 lần chiều rộng W -> kích thước [B, H, 2*W, C]).
    """
    ext_img, mask, _, _, _ = extend_image_horizontal_tensor(
        image=image,
        add_left_widths=frames_left,
        add_right_widths=0.0,
        extra_pixels_left=extra_pixels_left,
        fill_mode=fill_mode,
        feather_pixels=feather_pixels,
        overlap_pixels=overlap_pixels,
    )
    return ext_img, mask, int(image.shape[2])


def extend_image_right_tensor(
    image: torch.Tensor,
    frames_right: float = 1.0,
    extra_pixels_right: int = 0,
    fill_mode: str = "mirror_blur",
    feather_pixels: int = 48,
    overlap_pixels: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Nhận 1 ảnh đầu vào [B, H, W, C], lấy chiều cao H làm chuẩn và mở rộng qua PHẢI
    (mặc định +1.0 lần chiều rộng W -> kích thước [B, H, 2*W, C]).
    """
    ext_img, mask, _, _, _ = extend_image_horizontal_tensor(
        image=image,
        add_left_widths=0.0,
        add_right_widths=frames_right,
        extra_pixels_right=extra_pixels_right,
        fill_mode=fill_mode,
        feather_pixels=feather_pixels,
        overlap_pixels=overlap_pixels,
    )
    return ext_img, mask, int(image.shape[2])


def blend_restore_original_tensor(
    reference_or_original_image: torch.Tensor,
    generated_extended_image: torch.Tensor,
    outpaint_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Ghép khôi phục 100% độ nét của ảnh gốc vào ảnh đã mở rộng từ Qwen-Image-2.1.
    Tự động hỗ trợ cả 3 trường hợp:
    - Mở rộng qua Trái (+Left)
    - Mở rộng qua Phải (+Right)
    - Mở rộng cả 2 bên (Screen X)
    Người dùng có thể truyền `extended_image` (khung đã pad) hoặc `original_image` vào `reference_or_original_image`.
    """
    B, ref_H, ref_W, ref_C = reference_or_original_image.shape
    _, gen_H, gen_W, gen_C = generated_extended_image.shape

    # Chuẩn hóa ảnh AI sinh ra về đúng chiều cao chuẩn ref_H
    if gen_H != ref_H:
        scale = float(ref_H) / float(gen_H)
        target_W = int(round(gen_W * scale))
        gen_nchw = generated_extended_image.movedim(-1, 1)
        gen_resized = F.interpolate(gen_nchw, size=(ref_H, target_W), mode="bicubic", align_corners=False).movedim(1, -1).clamp(0.0, 1.0)
    else:
        gen_resized = generated_extended_image
        target_W = gen_W

    # Chuẩn hóa mặt nạ về (ref_H, target_W)
    if outpaint_mask.shape[-2:] != (ref_H, target_W):
        m = outpaint_mask.unsqueeze(1) if outpaint_mask.ndim == 3 else outpaint_mask.unsqueeze(0).unsqueeze(0)
        m = F.interpolate(m.float(), size=(ref_H, target_W), mode="bilinear", align_corners=False).squeeze(1)
    else:
        m = outpaint_mask

    out = gen_resized.clone()

    # Trường hợp 1: Đưa vào `extended_image` có cùng chiều rộng target_W (hoặc xấp xỉ)
    if abs(ref_W - target_W) <= 4:
        if ref_W != target_W:
            ref_resized = F.interpolate(reference_or_original_image.movedim(-1, 1), size=(ref_H, target_W), mode="bicubic", align_corners=False).movedim(1, -1)
        else:
            ref_resized = reference_or_original_image
        alpha = m.unsqueeze(-1).to(out.device, out.dtype)
        ref_slice = ref_resized[:, :, :, :gen_C].to(out.device, out.dtype)
        out[:, :, :, :gen_C] = ref_slice * (1.0 - alpha) + out[:, :, :, :gen_C] * alpha
        return out

    # Trường hợp 2: Đưa vào `original_image` có chiều rộng nhỏ hơn target_W -> Tự tìm vị trí vùng mask = 0.0
    copy_W = min(ref_W, target_W)
    left_mask_val = float(m[:, :, : max(1, copy_W // 4)].mean().item())
    right_mask_val = float(m[:, :, -max(1, copy_W // 4) :].mean().item())

    if left_mask_val <= right_mask_val and left_mask_val < 0.25:
        # Ảnh gốc nằm ở bên TRÁI (do mở rộng qua Phải)
        start_x = 0
    elif right_mask_val < left_mask_val and right_mask_val < 0.25:
        # Ảnh gốc nằm ở bên PHẢI (do mở rộng qua Trái)
        start_x = target_W - copy_W
    else:
        # Ảnh gốc nằm ở GIỮA (do mở rộng cả Trái & Phải)
        start_x = (target_W - copy_W) // 2

    end_x = start_x + copy_W
    alpha = m[:, :, start_x:end_x].unsqueeze(-1).to(out.device, out.dtype)
    orig_slice = reference_or_original_image[:, :, :copy_W, :gen_C].to(out.device, out.dtype)
    out[:, :, start_x:end_x, :gen_C] = orig_slice * (1.0 - alpha) + out[:, :, start_x:end_x, :gen_C] * alpha
    return out


# Giữ alias tương thích ngược
blend_original_left_tensor = blend_restore_original_tensor


# ==============================================================================
# NODE 1: Mở rộng + Bù qua TRÁI 1 lần chiều rộng (Lấy chiều cao tấm ảnh làm chuẩn)
# ==============================================================================
class VStone_ExtendLeft_1xWidth:
    """
    Node 1 (Screen X - +Qua Trái):
    Đưa 1 tấm ảnh vào -> Lấy chiều cao tấm ảnh (H) làm chuẩn -> Kéo khung rộng qua TRÁI
    bằng đúng +1 lần chiều rộng của nó (1.0x Width) và tự động bù nền + tạo Mask cho Qwen-Image-2.1.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width_multiplier_left": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 4.0,
                    "step": 0.1,
                    "tooltip": "1.0 = Cộng qua TRÁI đúng 1 lần chiều rộng của tấm ảnh (giữ nguyên chiều cao làm chuẩn)."
                }),
                "standard_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "0 = Giữ nguyên chiều cao gốc của tấm ảnh làm chuẩn. Nếu > 0 sẽ chuẩn hóa chiều cao về số này."
                }),
                "fill_mode": ([
                    "mirror_blur (Mượt màu & ánh sáng cho Qwen Edit - Khuyên dùng)",
                    "gray_0.5 (Xám trung tính chuẩn Qwen Fun ControlNet Inpaint)",
                    "edge_stretch (Kéo giãn viền trái)",
                    "mirror_sharp (Lật gương sắc nét)",
                    "white (Trắng)",
                    "black (Đen)"
                ], {
                    "default": "mirror_blur (Mượt màu & ánh sáng cho Qwen Edit - Khuyên dùng)"
                }),
                "feather_pixels": ("INT", {
                    "default": 48,
                    "min": 0,
                    "max": 512,
                    "step": 4,
                    "tooltip": "Độ mượt vùng chuyển tiếp giữa khung mới bù bên trái và ảnh gốc bên phải."
                }),
                "overlap_pixels": ("INT", {
                    "default": 16,
                    "min": 0,
                    "max": 256,
                    "step": 4,
                    "tooltip": "Số pixel lấn nhẹ vào mép trái ảnh gốc để xóa đường gờ ghép nối."
                }),
                "scene_description": ("STRING", {
                    "multiline": True,
                    "default": "seamlessly extend and continue the background environment, lighting, and perspective into the left half of the image."
                }),
            },
            "optional": {
                "base_image_for_width": ("IMAGE", {
                    "tooltip": "Tùy chọn: Nối ảnh gốc ban đầu vào đây nếu bạn nối tiếp Node +Qua Trái và Node +Qua Phải để cả 2 cùng cộng đúng 1x chiều rộng ảnh gốc."
                }),
                "previous_mask": ("MASK", {
                    "tooltip": "Tùy chọn: Nối mask từ node mở rộng trước đó nếu ghép nối tiếp."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("extended_image", "outpaint_mask", "original_image", "qwen_extend_prompt", "new_width", "locked_height")
    FUNCTION = "extend_left_1x"
    CATEGORY = "vstone/qwen_extend"

    def extend_left_1x(
        self,
        image: torch.Tensor,
        width_multiplier_left: float,
        standard_height: int,
        fill_mode: str,
        feather_pixels: int,
        overlap_pixels: int,
        scene_description: str,
        base_image_for_width: torch.Tensor | None = None,
        previous_mask: torch.Tensor | None = None,
    ):
        extended_image, outpaint_mask, norm_orig, new_w, locked_h = extend_image_horizontal_tensor(
            image=image,
            add_left_widths=width_multiplier_left,
            add_right_widths=0.0,
            extra_pixels_left=0,
            extra_pixels_right=0,
            standard_height=standard_height,
            reference_width_image=base_image_for_width,
            previous_mask=previous_mask,
            fill_mode=fill_mode,
            feather_pixels=feather_pixels,
            overlap_pixels=overlap_pixels,
        )

        qwen_prompt = (
            f"Keep the right side of <image1> completely unchanged and identical. "
            f"Outpaint and fill the extended left side of <image1> to {scene_description.strip()} "
            f"Ensure a seamless, natural continuation with no visible vertical seam or border."
        )
        # Trả về extended_image ở cả cổng thứ 3 để nối vào VStone_SeamlessRestore luôn khớp mọi vị trí
        return (extended_image, outpaint_mask, norm_orig, qwen_prompt, new_w, locked_h)


# ==============================================================================
# NODE 2: Mở rộng + Bù qua PHẢI 1 lần chiều rộng (Lấy chiều cao tấm ảnh làm chuẩn)
# ==============================================================================
class VStone_ExtendRight_1xWidth:
    """
    Node 2 (Screen X - +Qua Phải):
    Đưa 1 tấm ảnh vào -> Lấy chiều cao tấm ảnh (H) làm chuẩn -> Kéo khung rộng qua PHẢI
    bằng đúng +1 lần chiều rộng của nó (1.0x Width) và tự động bù nền + tạo Mask cho Qwen-Image-2.1.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width_multiplier_right": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 4.0,
                    "step": 0.1,
                    "tooltip": "1.0 = Cộng qua PHẢI đúng 1 lần chiều rộng của tấm ảnh (giữ nguyên chiều cao làm chuẩn)."
                }),
                "standard_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "0 = Giữ nguyên chiều cao gốc của tấm ảnh làm chuẩn. Nếu > 0 sẽ chuẩn hóa chiều cao về số này."
                }),
                "fill_mode": ([
                    "mirror_blur (Mượt màu & ánh sáng cho Qwen Edit - Khuyên dùng)",
                    "gray_0.5 (Xám trung tính chuẩn Qwen Fun ControlNet Inpaint)",
                    "edge_stretch (Kéo giãn viền phải)",
                    "mirror_sharp (Lật gương sắc nét)",
                    "white (Trắng)",
                    "black (Đen)"
                ], {
                    "default": "mirror_blur (Mượt màu & ánh sáng cho Qwen Edit - Khuyên dùng)"
                }),
                "feather_pixels": ("INT", {
                    "default": 48,
                    "min": 0,
                    "max": 512,
                    "step": 4,
                    "tooltip": "Độ mượt vùng chuyển tiếp giữa ảnh gốc bên trái và khung mới bù bên phải."
                }),
                "overlap_pixels": ("INT", {
                    "default": 16,
                    "min": 0,
                    "max": 256,
                    "step": 4,
                    "tooltip": "Số pixel lấn nhẹ vào mép phải ảnh gốc để xóa đường gờ ghép nối."
                }),
                "scene_description": ("STRING", {
                    "multiline": True,
                    "default": "seamlessly extend and continue the background environment, lighting, and perspective into the right half of the image."
                }),
            },
            "optional": {
                "base_image_for_width": ("IMAGE", {
                    "tooltip": "Tùy chọn: Nối ảnh gốc ban đầu vào đây nếu bạn nối tiếp Node +Qua Trái và Node +Qua Phải để cả 2 cùng cộng đúng 1x chiều rộng ảnh gốc."
                }),
                "previous_mask": ("MASK", {
                    "tooltip": "Tùy chọn: Nối mask từ node mở rộng trước đó nếu ghép nối tiếp."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("extended_image", "outpaint_mask", "original_image", "qwen_extend_prompt", "new_width", "locked_height")
    FUNCTION = "extend_right_1x"
    CATEGORY = "vstone/qwen_extend"

    def extend_right_1x(
        self,
        image: torch.Tensor,
        width_multiplier_right: float,
        standard_height: int,
        fill_mode: str,
        feather_pixels: int,
        overlap_pixels: int,
        scene_description: str,
        base_image_for_width: torch.Tensor | None = None,
        previous_mask: torch.Tensor | None = None,
    ):
        extended_image, outpaint_mask, norm_orig, new_w, locked_h = extend_image_horizontal_tensor(
            image=image,
            add_left_widths=0.0,
            add_right_widths=width_multiplier_right,
            extra_pixels_left=0,
            extra_pixels_right=0,
            standard_height=standard_height,
            reference_width_image=base_image_for_width,
            previous_mask=previous_mask,
            fill_mode=fill_mode,
            feather_pixels=feather_pixels,
            overlap_pixels=overlap_pixels,
        )

        qwen_prompt = (
            f"Keep the left side of <image1> completely unchanged and identical. "
            f"Outpaint and fill the extended right side of <image1> to {scene_description.strip()} "
            f"Ensure a seamless, natural continuation with no visible vertical seam or border."
        )
        return (extended_image, outpaint_mask, norm_orig, qwen_prompt, new_w, locked_h)


# ==============================================================================
# NODE 3: Mở rộng Screen X Cả 2 Bên Trái & Phải Cùng Lúc
# ==============================================================================
class VStone_ScreenX_ExtendCanvas:
    """
    Node 3 (Screen X - Mở rộng khung ngang Trái + Phải):
    Đưa 1 tấm ảnh vào, lấy chiều cao tấm ảnh làm chuẩn, kéo rộng khung qua Trái (+N lần chiều rộng)
    và/hoặc qua Phải (+M lần chiều rộng) và tự động bù vào tạo khung Screen X.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "add_left_widths": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 4.0,
                    "step": 0.1,
                    "tooltip": "Số lần chiều rộng cộng thêm qua bên TRÁI (1.0 = +1 lần chiều rộng ảnh gốc)."
                }),
                "add_right_widths": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 4.0,
                    "step": 0.1,
                    "tooltip": "Số lần chiều rộng cộng thêm qua bên PHẢI (1.0 = +1 lần chiều rộng ảnh gốc)."
                }),
                "standard_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "0 = Lấy đúng chiều cao tấm ảnh gốc làm chuẩn."
                }),
                "fill_mode": ([
                    "mirror_blur (Mượt màu & ánh sáng cho Qwen Edit - Khuyên dùng)",
                    "gray_0.5 (Xám trung tính chuẩn Qwen Fun ControlNet Inpaint)",
                    "edge_stretch (Kéo giãn viền 2 bên)",
                    "mirror_sharp (Lật gương sắc nét)",
                    "white (Trắng)",
                    "black (Đen)"
                ], {
                    "default": "mirror_blur (Mượt màu & ánh sáng cho Qwen Edit - Khuyên dùng)"
                }),
                "feather_pixels": ("INT", {"default": 48, "min": 0, "max": 512, "step": 4}),
                "overlap_pixels": ("INT", {"default": 16, "min": 0, "max": 256, "step": 4}),
                "scene_description": ("STRING", {
                    "multiline": True,
                    "default": "seamlessly extend and continue the wide panoramic environment, lighting, and perspective on both sides."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("extended_image", "outpaint_mask", "original_image", "qwen_extend_prompt", "new_width", "locked_height")
    FUNCTION = "extend_screen_x"
    CATEGORY = "vstone/qwen_extend"

    def extend_screen_x(
        self,
        image: torch.Tensor,
        add_left_widths: float,
        add_right_widths: float,
        standard_height: int,
        fill_mode: str,
        feather_pixels: int,
        overlap_pixels: int,
        scene_description: str,
    ):
        extended_image, outpaint_mask, norm_orig, new_w, locked_h = extend_image_horizontal_tensor(
            image=image,
            add_left_widths=add_left_widths,
            add_right_widths=add_right_widths,
            standard_height=standard_height,
            fill_mode=fill_mode,
            feather_pixels=feather_pixels,
            overlap_pixels=overlap_pixels,
        )
        qwen_prompt = (
            f"Keep the original region of <image1> completely unchanged. "
            f"Outpaint and fill the extended horizontal canvas to {scene_description.strip()} "
            f"Ensure a seamless panoramic continuation with no visible seams."
        )
        return (extended_image, outpaint_mask, norm_orig, qwen_prompt, new_w, locked_h)


# ==============================================================================
# NODE 4: Ghép khôi phục 100% ảnh gốc (Hỗ trợ cả +Qua Trái, +Qua Phải, và Screen X)
# ==============================================================================
class VStone_SeamlessRestore:
    """
    Ghép khôi phục 100% pixel gốc sắc nét với vùng mới do Qwen-Image-2.1 vẽ bù vào
    (Tự động hoạt động cho cả mở rộng qua Trái, mở rộng qua Phải, hoặc mở rộng 2 bên).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE", {
                    "tooltip": "Nối `original_image` hoặc `extended_image` từ Node VStone Extend vào đây."
                }),
                "generated_extended_image": ("IMAGE", {
                    "tooltip": "Ảnh kết quả sau khi chạy qua Qwen-Image-2.1 / VAE Decode."
                }),
                "outpaint_mask": ("MASK", {
                    "tooltip": "Mặt nạ `outpaint_mask` xuất ra từ Node VStone Extend."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("final_extended_image",)
    FUNCTION = "restore_seamless"
    CATEGORY = "vstone/qwen_extend"

    def restore_seamless(
        self,
        original_image: torch.Tensor,
        generated_extended_image: torch.Tensor,
        outpaint_mask: torch.Tensor,
    ):
        blended = blend_restore_original_tensor(
            reference_or_original_image=original_image,
            generated_extended_image=generated_extended_image,
            outpaint_mask=outpaint_mask,
        )
        return (blended,)


# Giữ class cũ tương thích ngược với workflow đã lưu
class VStone_ImageExtendRight(VStone_ExtendRight_1xWidth):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "frames_right": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.1}),
                "extra_pixels_right": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "fill_mode": ([
                    "mirror_blur (Mượt màu & ánh sáng cho Qwen Edit - Khuyên dùng)",
                    "gray_0.5 (Xám trung tính chuẩn Qwen Fun ControlNet Inpaint)",
                    "edge_stretch (Kéo giãn viền phải)",
                    "mirror_sharp (Lật gương sắc nét)",
                    "white (Trắng)",
                    "black (Đen)"
                ], {"default": "mirror_blur (Mượt màu & ánh sáng cho Qwen Edit - Khuyên dùng)"}),
                "feather_pixels": ("INT", {"default": 48, "min": 0, "max": 512, "step": 4}),
                "overlap_pixels": ("INT", {"default": 16, "min": 0, "max": 256, "step": 4}),
                "scene_description": ("STRING", {
                    "multiline": True,
                    "default": "seamlessly extend and continue the background environment, lighting, and perspective into the right half of the image."
                }),
            }
        }

    FUNCTION = "extend_right"

    def extend_right(
        self,
        image: torch.Tensor,
        frames_right: float,
        extra_pixels_right: int,
        fill_mode: str,
        feather_pixels: int,
        overlap_pixels: int,
        scene_description: str,
    ):
        extended_image, outpaint_mask, norm_orig, new_w, locked_h = extend_image_horizontal_tensor(
            image=image,
            add_left_widths=0.0,
            add_right_widths=frames_right,
            extra_pixels_right=extra_pixels_right,
            fill_mode=fill_mode,
            feather_pixels=feather_pixels,
            overlap_pixels=overlap_pixels,
        )
        qwen_prompt = (
            f"Keep the left half of <image1> completely unchanged and identical. "
            f"Outpaint and replace the right half of <image1> to {scene_description.strip()} "
            f"Ensure a seamless, natural continuation with no visible vertical seam or border."
        )
        return (extended_image, outpaint_mask, norm_orig, qwen_prompt, new_w, locked_h)


VStone_SeamlessRestoreLeft = VStone_SeamlessRestore


# ==============================================================================
# NODE 5: VStone Qwen 2.1 Extend All-in-One (Hỗ trợ +Qua Trái / +Qua Phải / Screen X)
# ==============================================================================
class VStone_Qwen21_ExtendRight_AIO:
    """
    Node All-in-One của VStone:
    Nhận 1 ảnh -> Lấy chiều cao làm chuẩn -> Tự mở rộng +1 lần chiều rộng qua Trái hoặc qua Phải
    -> Chạy Qwen-Image-2.1 (kèm Fun ControlNet Union Inpaint) -> Hòa trộn giữ nguyên 100% ảnh gốc.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "direction": ([
                    "right (+Qua Phải 1 lần chiều rộng)",
                    "left (+Qua Trái 1 lần chiều rộng)",
                    "both_screen_x (+Qua Trái & Phải cùng lúc)"
                ], {
                    "default": "right (+Qua Phải 1 lần chiều rộng)"
                }),
                "frames_right": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 3.0, "step": 0.25}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Seamlessly extend the scene horizontally, matching the exact lighting, camera angle, and environment."
                }),
                "use_fun_controlnet_inpaint": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Bật Qwen-Image-2.1 Fun ControlNet Union ở chế độ Inpaint/Outpaint để khóa chặt vùng ảnh gốc."
                }),
                "turbo_8step": ("BOOLEAN", {"default": True}),
                "resolution": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 50}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 5.0, "step": 0.1}),
                "feather_pixels": ("INT", {"default": 48, "min": 0, "max": 256, "step": 4}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("final_extended_image", "padded_preview", "outpaint_mask")
    FUNCTION = "run_extend_right_aio"
    CATEGORY = "vstone/qwen_extend"

    def run_extend_right_aio(
        self,
        image: torch.Tensor,
        frames_right: float,
        prompt: str,
        use_fun_controlnet_inpaint: bool,
        turbo_8step: bool,
        resolution: int,
        steps: int,
        cfg: float,
        feather_pixels: int,
        seed: int,
        direction: str = "right (+Qua Phải 1 lần chiều rộng)",
    ):
        import folder_paths
        import nodes
        import comfy.sd
        import comfy.utils
        from comfy_extras.nodes_qwen import TextEncodeQwenImage21

        try:
            from ..aio_nodes import ensure_model_exists
        except ImportError:
            from aio_nodes import ensure_model_exists

        add_left = float(frames_right) if direction.startswith("left") or direction.startswith("both") else 0.0
        add_right = float(frames_right) if direction.startswith("right") or direction.startswith("both") else 0.0

        fill_mode = "gray_0.5" if use_fun_controlnet_inpaint else "mirror_blur"
        padded_image, outpaint_mask, _, _, _ = extend_image_horizontal_tensor(
            image=image,
            add_left_widths=add_left,
            add_right_widths=add_right,
            fill_mode=fill_mode,
            feather_pixels=feather_pixels,
            overlap_pixels=16,
        )

        unet_path = ensure_model_exists("qwen_image_2.1_int8_convrot.safetensors")
        clip_path = ensure_model_exists("qwen3vl_8b_int8_convrot.safetensors")
        vae_path = ensure_model_exists("qwen_image_2.1_vae_bf16.safetensors")

        model = comfy.sd.load_diffusion_model(unet_path)
        if turbo_8step:
            lora_path = ensure_model_exists("qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors")
            lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model, _ = comfy.sd.load_lora_for_models(model, None, lora_sd, 1.0, 0.0)

        model = model.clone()
        model.model_options.setdefault("transformer_options", {})["qwen_image21_cache"] = {
            "device": "gpu",
            "dtype": "default"
        }

        clip_type = getattr(comfy.sd.CLIPType, "QWEN_IMAGE")
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type
        )
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))

        if use_fun_controlnet_inpaint:
            from comfy_extras.nodes_model_patch import ModelPatchLoader, ZImageFunControlnet
            cnet_name = "qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors"
            cnet_path = folder_paths.get_full_path("model_patches", cnet_name)
            if not cnet_path:
                from ..aio_nodes import HF_QWEN_BASE, MODEL_CATALOG
                MODEL_CATALOG[cnet_name] = ("model_patches", f"{HF_QWEN_BASE}/model_patches/{cnet_name}")
                ensure_model_exists(cnet_name)
            model_patch = ModelPatchLoader().load_model_patch(cnet_name)[0]
            model = ZImageFunControlnet().diffsynth_controlnet(
                model=model,
                model_patch=model_patch,
                vae=vae,
                image=None,
                strength=1.0,
                inpaint_image=padded_image,
                mask=outpaint_mask,
                start_percent=0.0,
                end_percent=1.0,
            )[0]

        if direction.startswith("left"):
            full_prompt = f"Keep the right side of <image1> unchanged. Seamlessly extend and outpaint the left side: {prompt}"
        elif direction.startswith("both"):
            full_prompt = f"Keep the center of <image1> unchanged. Seamlessly extend and outpaint both left and right sides: {prompt}"
        else:
            full_prompt = f"Keep the left side of <image1> unchanged. Seamlessly extend and outpaint the right side: {prompt}"

        enc_out = TextEncodeQwenImage21.execute(
            clip=clip,
            prompt=full_prompt,
            negative_prompt="",
            vae=vae,
            resolution=resolution,
            images={"image_1": padded_image},
        )
        positive, negative, latent_dict = enc_out[0], enc_out[1], enc_out[2]

        sampled_latent = nodes.common_ksampler(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name="euler",
            scheduler="simple",
            positive=positive,
            negative=negative,
            latent=latent_dict,
            denoise=1.0,
        )[0]

        decoded = vae.decode(sampled_latent["samples"])

        final_image = blend_restore_original_tensor(
            reference_or_original_image=padded_image,
            generated_extended_image=decoded,
            outpaint_mask=outpaint_mask,
        )
        return (final_image, padded_image, outpaint_mask)


VSTONE_NODE_CLASS_MAPPINGS = {
    "VStone_ExtendLeft_1xWidth": VStone_ExtendLeft_1xWidth,
    "VStone_ExtendRight_1xWidth": VStone_ExtendRight_1xWidth,
    "VStone_ScreenX_ExtendCanvas": VStone_ScreenX_ExtendCanvas,
    "VStone_SeamlessRestore": VStone_SeamlessRestore,
    "VStone_ImageExtendRight": VStone_ImageExtendRight,
    "VStone_SeamlessRestoreLeft": VStone_SeamlessRestoreLeft,
    "VStone_Qwen21_ExtendRight_AIO": VStone_Qwen21_ExtendRight_AIO,
}

VSTONE_NODE_DISPLAY_NAME_MAPPINGS = {
    "VStone_ExtendLeft_1xWidth": "VStone: +Qua Trái 1 Lần Chiều Rộng (Lock Height - Screen X)",
    "VStone_ExtendRight_1xWidth": "VStone: +Qua Phải 1 Lần Chiều Rộng (Lock Height - Screen X)",
    "VStone_ScreenX_ExtendCanvas": "VStone: Screen X Mở Rộng Khung Ngang (Trái + Phải)",
    "VStone_SeamlessRestore": "VStone: Ghép Khôi Phục Ảnh Gốc (Trái / Phải / Screen X)",
    "VStone_ImageExtendRight": "VStone: Extend Image Right (1 Frame Outpaint Pad)",
    "VStone_SeamlessRestoreLeft": "VStone: Seamless Restore Left (Blend Original)",
    "VStone_Qwen21_ExtendRight_AIO": "VStone: Qwen 2.1 Extend All-in-One (Trái / Phải / Screen X)",
}
