"""
VStone Node Helper Module (node_helper_vstone.py)
Chỉ làm đúng 1 nhiệm vụ:
Nhận 1 tấm ảnh đầu vào -> Lấy chiều cao tấm ảnh làm chuẩn -> Mở rộng vùng bù (qua Trái, qua Phải, hoặc 4 hướng)
-> Tô màu phẳng sắc nét vào vùng bù (Xám #808080, Đỏ #FF0000, Xanh lá #00FF00, Đen, Trắng, hoặc mã Hex tùy chọn)
-> Xuất ra `IMAGE` và `MASK` để đi tiếp vào workflow ComfyUI (phục vụ các LoRA Outpaint / Inpaint).
Hoàn toàn KHÔNG resize ảnh gốc và KHÔNG làm mờ (blur) ảnh.
"""
from __future__ import annotations

import math
from typing import Any

try:
    import torch
except Exception:
    torch = Any  # type: ignore


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
    if fill_color.startswith("gray"):
        return (128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0)  # Chuẩn #808080
    if fill_color.startswith("red"):
        return (1.0, 0.0, 0.0)  # #FF0000
    if fill_color.startswith("green"):
        return (0.0, 1.0, 0.0)  # #00FF00
    if fill_color.startswith("blue"):
        return (0.0, 0.0, 1.0)  # #0000FF
    if fill_color.startswith("white"):
        return (1.0, 1.0, 1.0)  # #FFFFFF
    if fill_color.startswith("black"):
        return (0.0, 0.0, 0.0)  # #000000

    # Parse custom hex (#RRGGBB)
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

    # Giữ 3 kênh RGB chuẩn
    orig_rgb = image[:, :, :, :3]
    B, H, W, C = orig_rgb.shape
    device = orig_rgb.device
    dtype = orig_rgb.dtype

    # Nếu có nối ảnh gốc chuẩn vào `base_image_for_width` thì lấy chiều rộng của ảnh đó làm 1 đơn vị chiều rộng
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

    # 2. Ghép mở rộng Trên / Dưới (nếu extra_top_px hoặc extra_bottom_px > 0)
    top_block = create_solid_pad_tensor(B, pad_top, new_W, C, rgb, device, dtype)
    bottom_block = create_solid_pad_tensor(B, pad_bottom, new_W, C, rgb, device, dtype)
    padded_image = torch.cat([top_block, row_img, bottom_block], dim=1)
    new_H = pad_top + H + pad_bottom

    # 3. Tạo Mask: 1.0 tại vùng vừa bù thêm, 0.0 tại vùng ảnh gốc
    mask = torch.ones((B, new_H, new_W), dtype=dtype, device=device)
    mask[:, pad_top : pad_top + H, pad_left : pad_left + W] = 0.0

    # Nếu người dùng muốn làm mềm biên của MASK (không ảnh hưởng đến độ nét của IMAGE)
    if feather_mask_px > 0:
        f = int(feather_mask_px)
        if pad_left > 0 and f > 0:
            f_len = min(f, W)
            t = torch.linspace(0.0, 1.0, steps=f_len, dtype=dtype, device=device)
            ramp = 0.5 * (1.0 + torch.cos(math.pi * t))  # 1.0 -> 0.0
            mask[:, pad_top : pad_top + H, pad_left : pad_left + f_len] = torch.maximum(
                mask[:, pad_top : pad_top + H, pad_left : pad_left + f_len],
                ramp.view(1, 1, f_len),
            )
        if pad_right > 0 and f > 0:
            f_len = min(f, W)
            t = torch.linspace(0.0, 1.0, steps=f_len, dtype=dtype, device=device)
            ramp = 0.5 * (1.0 - torch.cos(math.pi * t))  # 0.0 -> 1.0
            mask[:, pad_top : pad_top + H, pad_left + W - f_len : pad_left + W] = torch.maximum(
                mask[:, pad_top : pad_top + H, pad_left + W - f_len : pad_left + W],
                ramp.view(1, 1, f_len),
            )

    return padded_image, mask, new_W, new_H


# ==============================================================================
# NODE 1: VStone Outpaint (Node tổng hợp: Tạo vùng bù Trái / Phải & Chọn màu tô)
# ==============================================================================
class VStone_Outpaint_Pad:
    """
    Node tạo vùng mở rộng (Outpaint Pad) cho tấm ảnh:
    - Lấy chiều cao tấm ảnh làm chuẩn, giữ nguyên 100% độ nét ảnh gốc.
    - Kéo rộng khung qua Trái / qua Phải theo tỷ lệ chiều rộng (hoặc pixel).
    - Tô màu vùng bù do bạn chọn (Xám #808080, Đỏ #FF0000, Xanh lá #00FF00, Trắng, Đen, hoặc mã Hex).
    - Xuất ra `image` và `mask` để nối đi tiếp trong workflow.
    """

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
                    "tooltip": "Số lần chiều rộng cộng thêm qua bên TRÁI (ví dụ: 1.0 = cộng qua trái 1 lần chiều rộng)."
                }),
                "add_right_widths": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Số lần chiều rộng cộng thêm qua bên PHẢI (ví dụ: 1.0 = cộng qua phải 1 lần chiều rộng)."
                }),
                "fill_color": (COLOR_CHOICES, {
                    "default": COLOR_CHOICES[0],
                    "tooltip": "Chọn màu tô vào vùng bù mở rộng (Xám #808080, Đỏ #FF0000, Xanh lá #00FF00...)."
                }),
                "custom_hex": ("STRING", {
                    "default": "#808080",
                    "tooltip": "Mã màu Hex nếu bạn chọn `custom_hex` ở trên (ví dụ: #808080, #FF0000, #00FF00)."
                }),
                "extra_left_px": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "extra_right_px": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "extra_top_px": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "extra_bottom_px": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "feather_mask_px": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 512,
                    "step": 4,
                    "tooltip": "Làm mềm biên của MASK xuất ra (0 = biên sắc nét; chỉ tác động lên MASK, không làm mờ IMAGE)."
                }),
            },
            "optional": {
                "base_image_for_width": ("IMAGE", {
                    "tooltip": "Tùy chọn: Nối ảnh gốc ban đầu vào đây nếu bạn mắc nối tiếp nhiều node mở rộng."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    FUNCTION = "outpaint_pad"
    CATEGORY = "vstone/outpaint"

    def outpaint_pad(
        self,
        image: torch.Tensor,
        add_left_widths: float,
        add_right_widths: float,
        fill_color: str,
        custom_hex: str,
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
# NODE 2: VStone +Qua Trái 1 Lần Chiều Rộng (Lấy chiều cao làm chuẩn & Tô màu vùng bù)
# ==============================================================================
class VStone_ExtendLeft_1xWidth:
    """
    Lấy chiều cao tấm ảnh làm chuẩn, cộng qua TRÁI 1 lần chiều rộng,
    tô màu vùng bù (Xám / Đỏ / Xanh...) và xuất `image` + `mask` để đi tiếp workflow.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width_multiplier_left": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "1.0 = Cộng qua TRÁI đúng 1 lần chiều rộng tấm ảnh."
                }),
                "fill_color": (COLOR_CHOICES, {
                    "default": COLOR_CHOICES[0],
                }),
                "custom_hex": ("STRING", {
                    "default": "#808080",
                }),
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
    CATEGORY = "vstone/outpaint"

    def extend_left(
        self,
        image: torch.Tensor,
        width_multiplier_left: float,
        fill_color: str,
        custom_hex: str,
        extra_left_px: int = 0,
        feather_mask_px: int = 0,
        base_image_for_width: torch.Tensor | None = None,
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
# NODE 3: VStone +Qua Phải 1 Lần Chiều Rộng (Lấy chiều cao làm chuẩn & Tô màu vùng bù)
# ==============================================================================
class VStone_ExtendRight_1xWidth:
    """
    Lấy chiều cao tấm ảnh làm chuẩn, cộng qua PHẢI 1 lần chiều rộng,
    tô màu vùng bù (Xám / Đỏ / Xanh...) và xuất `image` + `mask` để đi tiếp workflow.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width_multiplier_right": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "1.0 = Cộng qua PHẢI đúng 1 lần chiều rộng tấm ảnh."
                }),
                "fill_color": (COLOR_CHOICES, {
                    "default": COLOR_CHOICES[0],
                }),
                "custom_hex": ("STRING", {
                    "default": "#808080",
                }),
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
    CATEGORY = "vstone/outpaint"

    def extend_right(
        self,
        image: torch.Tensor,
        width_multiplier_right: float,
        fill_color: str,
        custom_hex: str,
        extra_right_px: int = 0,
        feather_mask_px: int = 0,
        base_image_for_width: torch.Tensor | None = None,
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


VSTONE_NODE_CLASS_MAPPINGS = {
    "VStone_Outpaint_Pad": VStone_Outpaint_Pad,
    "VStone_ExtendLeft_1xWidth": VStone_ExtendLeft_1xWidth,
    "VStone_ExtendRight_1xWidth": VStone_ExtendRight_1xWidth,
}

VSTONE_NODE_DISPLAY_NAME_MAPPINGS = {
    "VStone_Outpaint_Pad": "VStone: Outpaint Tạo Vùng Bù (Chọn Màu Xám / Đỏ / Xanh)",
    "VStone_ExtendLeft_1xWidth": "VStone: Outpaint +Qua Trái 1 Lần Chiều Rộng (Chọn Màu Tô)",
    "VStone_ExtendRight_1xWidth": "VStone: Outpaint +Qua Phải 1 Lần Chiều Rộng (Chọn Màu Tô)",
}
