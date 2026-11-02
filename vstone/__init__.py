"""
VStone Custom Node Package (`vstone`)
Chứa các node hỗ trợ mở rộng khung ảnh Screen X (Image-to-Image Extend / Outpaint Trái & Phải) cho Qwen-Image-2.1.
Có thể dùng độc lập trong `ComfyUI/custom_nodes/vstone` hoặc tích hợp chung trong `ComfyUI_Qwen21_MiniMaxH3_AIO`.
"""

from .node_helper_vstone import (
    VSTONE_NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS,
    VSTONE_NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS,
    extend_image_horizontal_tensor,
    extend_image_left_tensor,
    extend_image_right_tensor,
    blend_restore_original_tensor,
    blend_original_left_tensor,
)

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "extend_image_horizontal_tensor",
    "extend_image_left_tensor",
    "extend_image_right_tensor",
    "blend_restore_original_tensor",
    "blend_original_left_tensor",
]
