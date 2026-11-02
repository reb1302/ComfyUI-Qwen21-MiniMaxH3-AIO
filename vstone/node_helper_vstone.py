"""
VStone Node Helper Module (node_helper_vstone.py)
Cách cài đặt cực kỳ đơn giản (chọn 1 trong 2 cách đều chạy 100%):
- Cách 1: Chép trực tiếp file `node_helper_vstone.py` này thả thẳng vào thư mục `ComfyUI/custom_nodes/`
- Cách 2: Chép thư mục `vstone` (gồm `__init__.py` và `node_helper_vstone.py`) vào `ComfyUI/custom_nodes/vstone/`
Sau khi chép xong, TẮT và BẬT LẠI ComfyUI (`./start_h200.sh`) để ComfyUI nạp node mới.
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


COLOR_CHOICES = [
    "gray (#808080 - Tô Xám chuẩn Outpaint LoRA)",
    "red (#FF0000 - Tô Đỏ)",
    "green (#00FF00 - Tô Xanh Lá)",
    "blue (#0000FF - Tô Xanh Dương)",
    "white (#FFFFFF - Tô Trắng)",
    "black (#000000 - Tô Đen)",
    "custom_hex (Dùng mã màu Hex bên dưới)",
]


def _parse_fill_rgb(fill_color: str, custom_hex: str = "#808080") -> tuple[float, float, float]:
    """Chuyển lựa chọn màu tô sang bộ 3 giá trị RGB trong khoảng [0.0, 1.0]."""
    fc = (fill_color or "").lower()
    if fc.startswith("gray") or "#808080" in fc:
        return (128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0)  # Chuẩn #808080
    if fc.startswith("red") or "#ff0000" in fc:
        return (1.0, 0.0, 0.0)  # #FF0000
    if fc.startswith("green") or "#00ff00" in fc:
        return (0.0, 1.0, 0.0)  # #00FF00
    if fc.startswith("blue") or "#0000ff" in fc:
        return (0.0, 0.0, 1.0)  # #0000FF
    if fc.startswith("white") or "#ffffff" in fc:
        return (1.0, 1.0, 1.0)  # #FFFFFF
    if fc.startswith("black") or "#000000" in fc:
        return (0.0, 0.0, 0.0)  # #000000

    hx = (custom_hex or "#808080").strip().lstrip("#")
    if len(hx) == 3:
        hx = "".join(c * 2 for c in hx)
    if len(hx) == 6:
        try:
            r = int(hx[0:2], 16) / 255.0
            g = int(hx[2:4], 16) / 255.0
            b = int(hx[4:6], 16) / 255.0
            return (r, g, b)
        except ValueError:
            pass
    return (128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0)


def create_solid_pad_tensor(
    batch: int,
    height: int,
    width: int,
    channels: int,
    rgb: tuple[float, float, float],
    device: Any,
    dtype: Any,
) -> torch.Tensor:
    """Tạo vùng ảnh bù tô màu đặc sắc nét 100%, không làm mờ."""
    if height <= 0 or width <= 0:
        return torch.empty((batch, max(0, height), max(0, width), channels), dtype=dtype, device=device)

    pad = torch.zeros((batch, height, width, channels), dtype=dtype, device=device)
    pad[:, :, :, 0] = rgb[0]
    if channels >= 2:
        pad[:, :, :, 1] = rgb[1]
    if channels >= 3:
        pad[:, :, :, 2] = rgb[2]
    if channels >= 4:
        pad[:, :, :, 3] = 1.0
    return pad


def vstone_outpaint_pad_tensor(
    image: torch.Tensor,
    add_left_ratio: float = 0.0,
    add_right_ratio: float = 0.0,
    extra_left_px: int = 0,
    extra_right_px: int = 0,
    extra_top_px: int = 0,
    extra_bottom_px: int = 0,
    fill_color: str = COLOR_CHOICES[0],
    custom_hex: str = "#808080",
    feather_mask_px: int = 0,
    base_image_for_width: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Nhận 1 tấm ảnh [B, H, W, C]:
    - Giữ nguyên 100% từng pixel của ảnh gốc (không resize, không blur).
    - Mở rộng tạo vùng bù bên Trái / bên Phải (theo tỷ lệ chiều rộng W hoặc số pixel) và tô màu đã chọn.
    - Trả về (padded_image [B, new_H, new_W, 3], outpaint_mask [B, new_H, new_W], new_W, new_H).
    """
    if image.ndim != 4:
        raise ValueError(f"Expected 4D IMAGE tensor [B, H, W, C], got shape {tuple(image.shape)}")

    orig_rgb = image[:, :, :, :3]
    B, H, W, C = orig_rgb.shape
    device = orig_rgb.device
    dtype = orig_rgb.dtype

    unit_W = int(base_image_for_width.shape[2]) if (base_image_for_width is not None and base_image_for_width.ndim == 4) else W

    pad_left = max(0, int(round(unit_W * float(add_left_ratio))) + int(extra_left_px))
    pad_right = max(0, int(round(unit_W * float(add_right_ratio))) + int(extra_right_px))
    pad_top = max(0, int(extra_top_px))
    pad_bottom = max(0, int(extra_bottom_px))

    rgb = _parse_fill_rgb(fill_color, custom_hex)

    # 1. Ghép mở rộng chiều ngang [Vùng bù Trái | Ảnh gốc | Vùng bù Phải] cùng chiều cao H
    left_block = create_solid_pad_tensor(B, H, pad_left, C, rgb, device, dtype)
    right_block = create_solid_pad_tensor(B, H, pad_right, C, rgb, device, dtype)
    row_img = torch.cat([left_block, orig_rgb, right_block], dim=2)
    new_W = pad_left + W + pad_right

    # 2. Ghép mở rộng Trên / Dưới (nếu có)
    top_block = create_solid_pad_tensor(B, pad_top, new_W, C, rgb, device, dtype)
    bottom_block = create_solid_pad_tensor(B, pad_bottom, new_W, C, rgb, device, dtype)
    padded_image = torch.cat([top_block, row_img, bottom_block], dim=1)
    new_H = pad_top + H + pad_bottom

    # 3. Tạo Mask: 1.0 tại vùng vừa bù thêm, 0.0 tại vùng ảnh gốc
    mask = torch.ones((B, new_H, new_W), dtype=dtype, device=device)
    mask[:, pad_top : pad_top + H, pad_left : pad_left + W] = 0.0

    if feather_mask_px > 0:
        f = int(feather_mask_px)
        if pad_left > 0 and f > 0:
            f_len = min(f, W)
            t = torch.linspace(0.0, 1.0, steps=f_len, dtype=dtype, device=device)
            ramp = 0.5 * (1.0 + torch.cos(math.pi * t))
            mask[:, pad_top : pad_top + H, pad_left : pad_left + f_len] = torch.maximum(
                mask[:, pad_top : pad_top + H, pad_left : pad_left + f_len],
                ramp.view(1, 1, f_len),
            )
        if pad_right > 0 and f > 0:
            f_len = min(f, W)
            t = torch.linspace(0.0, 1.0, steps=f_len, dtype=dtype, device=device)
            ramp = 0.5 * (1.0 - torch.cos(math.pi * t))
            mask[:, pad_top : pad_top + H, pad_left + W - f_len : pad_left + W] = torch.maximum(
                mask[:, pad_top : pad_top + H, pad_left + W - f_len : pad_left + W],
                ramp.view(1, 1, f_len),
            )

    return padded_image, mask, new_W, new_H


def blend_restore_original_tensor(
    reference_or_original_image: torch.Tensor,
    generated_extended_image: torch.Tensor,
    outpaint_mask: torch.Tensor,
) -> torch.Tensor:
    """Ghép khôi phục 100% độ nét của ảnh gốc vào ảnh đã mở rộng."""
    if reference_or_original_image.shape[-1] == 4:
        reference_or_original_image = reference_or_original_image[:, :, :, :3]
    if generated_extended_image.shape[-1] == 4:
        generated_extended_image = generated_extended_image[:, :, :, :3]

    B, ref_H, ref_W, ref_C = reference_or_original_image.shape
    _, gen_H, gen_W, gen_C = generated_extended_image.shape

    if gen_H != ref_H:
        scale = float(ref_H) / float(gen_H)
        target_W = int(round(gen_W * scale))
        gen_nchw = generated_extended_image.movedim(-1, 1)
        gen_resized = F.interpolate(gen_nchw, size=(ref_H, target_W), mode="bicubic", align_corners=False).movedim(1, -1).clamp(0.0, 1.0)
    else:
        gen_resized = generated_extended_image
        target_W = gen_W

    if outpaint_mask.shape[-2:] != (ref_H, target_W):
        m = outpaint_mask.unsqueeze(1) if outpaint_mask.ndim == 3 else outpaint_mask.unsqueeze(0).unsqueeze(0)
        m = F.interpolate(m.float(), size=(ref_H, target_W), mode="bilinear", align_corners=False).squeeze(1)
    else:
        m = outpaint_mask

    out = gen_resized.clone()

    if abs(ref_W - target_W) <= 32:
        if ref_W != target_W:
            ref_resized = F.interpolate(reference_or_original_image.movedim(-1, 1), size=(ref_H, target_W), mode="bicubic", align_corners=False).movedim(1, -1)
        else:
            ref_resized = reference_or_original_image
        alpha = m.unsqueeze(-1).to(out.device, out.dtype)
        ref_slice = ref_resized[:, :, :, :gen_C].to(out.device, out.dtype)
        out[:, :, :, :gen_C] = ref_slice * (1.0 - alpha) + out[:, :, :, :gen_C] * alpha
        return out

    copy_W = min(ref_W, target_W)
    left_mask_val = float(m[:, :, : max(1, copy_W // 4)].mean().item())
    right_mask_val = float(m[:, :, -max(1, copy_W // 4) :].mean().item())

    if left_mask_val <= right_mask_val and left_mask_val < 0.25:
        start_x = 0
    elif right_mask_val < left_mask_val and right_mask_val < 0.25:
        start_x = target_W - copy_W
    else:
        start_x = (target_W - copy_W) // 2

    end_x = start_x + copy_W
    alpha = m[:, :, start_x:end_x].unsqueeze(-1).to(out.device, out.dtype)
    orig_slice = reference_or_original_image[:, :, :copy_W, :gen_C].to(out.device, out.dtype)
    out[:, :, start_x:end_x, :gen_C] = orig_slice * (1.0 - alpha) + out[:, :, start_x:end_x, :gen_C] * alpha
    return out


# ==============================================================================
# NODE 1: VStone Outpaint (Tạo vùng bù Trái / Phải & Chọn màu tô Xám / Đỏ / Xanh)
# ==============================================================================
class VStone_Outpaint_Pad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "add_left_widths": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Số lần chiều rộng cộng thêm qua bên TRÁI (1.0 = +1 lần chiều rộng)."
                }),
                "add_right_widths": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Số lần chiều rộng cộng thêm qua bên PHẢI (1.0 = +1 lần chiều rộng)."
                }),
                "fill_color": (COLOR_CHOICES, {
                    "default": COLOR_CHOICES[0],
                }),
                "custom_hex": ("STRING", {
                    "default": "#808080",
                }),
                "extra_left_px": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "extra_right_px": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "extra_top_px": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "extra_bottom_px": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "feather_mask_px": ("INT", {"default": 0, "min": 0, "max": 512, "step": 4}),
            },
            "optional": {
                "base_image_for_width": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    FUNCTION = "outpaint_pad"
    CATEGORY = "vstone"

    def outpaint_pad(
        self,
        image: torch.Tensor,
        add_left_widths: float = 0.0,
        add_right_widths: float = 1.0,
        fill_color: str = COLOR_CHOICES[0],
        custom_hex: str = "#808080",
        extra_left_px: int = 0,
        extra_right_px: int = 0,
        extra_top_px: int = 0,
        extra_bottom_px: int = 0,
        feather_mask_px: int = 0,
        base_image_for_width: torch.Tensor | None = None,
    ):
        padded_img, mask, new_w, new_h = vstone_outpaint_pad_tensor(
            image=image,
            add_left_ratio=add_left_widths,
            add_right_ratio=add_right_widths,
            extra_left_px=extra_left_px,
            extra_right_px=extra_right_px,
            extra_top_px=extra_top_px,
            extra_bottom_px=extra_bottom_px,
            fill_color=fill_color,
            custom_hex=custom_hex,
            feather_mask_px=feather_mask_px,
            base_image_for_width=base_image_for_width,
        )
        return (padded_img, mask, new_w, new_h)


# ==============================================================================
# NODE 2: VStone +Qua Trái 1 Lần Chiều Rộng (Chọn Màu Tô Xám / Đỏ / Xanh)
# ==============================================================================
class VStone_ExtendLeft_1xWidth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width_multiplier_left": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "fill_color": (COLOR_CHOICES, {"default": COLOR_CHOICES[0]}),
                "custom_hex": ("STRING", {"default": "#808080"}),
                "extra_left_px": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "feather_mask_px": ("INT", {"default": 0, "min": 0, "max": 512, "step": 4}),
            },
            "optional": {
                "base_image_for_width": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    FUNCTION = "extend_left"
    CATEGORY = "vstone"

    def extend_left(
        self,
        image: torch.Tensor,
        width_multiplier_left: float = 1.0,
        fill_color: str = COLOR_CHOICES[0],
        custom_hex: str = "#808080",
        extra_left_px: int = 0,
        feather_mask_px: int = 0,
        base_image_for_width: torch.Tensor | None = None,
        **kwargs,
    ):
        padded_img, mask, new_w, new_h = vstone_outpaint_pad_tensor(
            image=image,
            add_left_ratio=width_multiplier_left,
            add_right_ratio=0.0,
            extra_left_px=extra_left_px,
            fill_color=fill_color,
            custom_hex=custom_hex,
            feather_mask_px=feather_mask_px,
            base_image_for_width=base_image_for_width,
        )
        return (padded_img, mask, new_w, new_h)


# ==============================================================================
# NODE 3: VStone +Qua Phải 1 Lần Chiều Rộng (Chọn Màu Tô Xám / Đỏ / Xanh)
# ==============================================================================
class VStone_ExtendRight_1xWidth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width_multiplier_right": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "fill_color": (COLOR_CHOICES, {"default": COLOR_CHOICES[0]}),
                "custom_hex": ("STRING", {"default": "#808080"}),
                "extra_right_px": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "feather_mask_px": ("INT", {"default": 0, "min": 0, "max": 512, "step": 4}),
            },
            "optional": {
                "base_image_for_width": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    FUNCTION = "extend_right"
    CATEGORY = "vstone"

    def extend_right(
        self,
        image: torch.Tensor,
        width_multiplier_right: float = 1.0,
        fill_color: str = COLOR_CHOICES[0],
        custom_hex: str = "#808080",
        extra_right_px: int = 0,
        feather_mask_px: int = 0,
        base_image_for_width: torch.Tensor | None = None,
        **kwargs,
    ):
        padded_img, mask, new_w, new_h = vstone_outpaint_pad_tensor(
            image=image,
            add_left_ratio=0.0,
            add_right_ratio=width_multiplier_right,
            extra_right_px=extra_right_px,
            fill_color=fill_color,
            custom_hex=custom_hex,
            feather_mask_px=feather_mask_px,
            base_image_for_width=base_image_for_width,
        )
        return (padded_img, mask, new_w, new_h)


# ==============================================================================
# NODE 4: Ghép khôi phục ảnh gốc (Nếu cần Stitch Inpaint sau khi qua KSampler)
# ==============================================================================
class VStone_SeamlessRestore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "generated_extended_image": ("IMAGE",),
                "outpaint_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "restore_seamless"
    CATEGORY = "vstone"

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


# Giữ đầy đủ tên class cũ làm alias để mọi workflow cũ không bao giờ báo Missing Node Type
VStone_ScreenX_ExtendCanvas = VStone_Outpaint_Pad
VStone_ImageExtendRight = VStone_ExtendRight_1xWidth
VStone_SeamlessRestoreLeft = VStone_SeamlessRestore


VSTONE_NODE_CLASS_MAPPINGS = {
    "VStone_Outpaint_Pad": VStone_Outpaint_Pad,
    "VStone_ExtendLeft_1xWidth": VStone_ExtendLeft_1xWidth,
    "VStone_ExtendRight_1xWidth": VStone_ExtendRight_1xWidth,
    "VStone_SeamlessRestore": VStone_SeamlessRestore,
    "VStone_ScreenX_ExtendCanvas": VStone_ScreenX_ExtendCanvas,
    "VStone_ImageExtendRight": VStone_ImageExtendRight,
    "VStone_SeamlessRestoreLeft": VStone_SeamlessRestoreLeft,
}

VSTONE_NODE_DISPLAY_NAME_MAPPINGS = {
    "VStone_Outpaint_Pad": "VStone: Outpaint Tạo Vùng Bù (Chọn Màu Xám / Đỏ / Xanh)",
    "VStone_ExtendLeft_1xWidth": "VStone: Outpaint +Qua Trái 1 Lần Chiều Rộng (Chọn Màu Tô)",
    "VStone_ExtendRight_1xWidth": "VStone: Outpaint +Qua Phải 1 Lần Chiều Rộng (Chọn Màu Tô)",
    "VStone_SeamlessRestore": "VStone: Ghép Khôi Phục Ảnh Gốc (Seamless Restore)",
    "VStone_ScreenX_ExtendCanvas": "VStone: Screen X Mở Rộng Khung Ngang (Trái + Phải)",
    "VStone_ImageExtendRight": "VStone: Extend Image Right (1 Frame Outpaint Pad)",
    "VStone_SeamlessRestoreLeft": "VStone: Seamless Restore Left (Blend Original)",
}

# Cho phép ComfyUI load trực tiếp file `node_helper_vstone.py` nếu bỏ thẳng vào `ComfyUI/custom_nodes/`
NODE_CLASS_MAPPINGS = VSTONE_NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = VSTONE_NODE_DISPLAY_NAME_MAPPINGS
