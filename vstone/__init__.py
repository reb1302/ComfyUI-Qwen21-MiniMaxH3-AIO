"""
VStone Custom Node Package (`vstone`)
Node chuyên tạo vùng bù Outpaint (tô màu Xám #808080, Đỏ #FF0000, Xanh lá #00FF00, hoặc mã Hex tùy chọn)
và xuất ra `image` + `mask` sắc nét 100% để nối đi tiếp trong workflow ComfyUI.
"""

from .node_helper_vstone import (
    VSTONE_NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS,
    VSTONE_NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS,
    vstone_outpaint_pad_tensor,
)

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "vstone_outpaint_pad_tensor",
]
