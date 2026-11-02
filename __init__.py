"""
ComfyUI All-in-One Custom Node Pack for Qwen-Image-2.1 & MiniMax-H3 (Optimized for NVIDIA H200 Linux)
Automatically installs bundled workflows & embeddings on startup and registers All-in-One nodes.
"""

import os
import shutil
from .aio_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

def _bootstrap_bundled_assets():
    """Copy bundled official workflows and MiniMax-H3 embeddings into ComfyUI on first load."""
    import folder_paths
    pkg_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Copy MiniMax-H3 embeddings into ComfyUI/models/embeddings
    src_emb_dir = os.path.join(pkg_dir, "embeddings")
    emb_paths = folder_paths.get_folder_paths("embeddings")
    if os.path.isdir(src_emb_dir) and emb_paths:
        dst_emb_dir = emb_paths[0]
        os.makedirs(dst_emb_dir, exist_ok=True)
        for fname in os.listdir(src_emb_dir):
            if fname.endswith(".safetensors"):
                dst_file = os.path.join(dst_emb_dir, fname)
                if not os.path.exists(dst_file):
                    try:
                        shutil.copy2(os.path.join(src_emb_dir, fname), dst_file)
                    except Exception as e:
                        print(f"[AIO-H200] Warning copying embedding {fname}: {e}")

    # 2. Copy official workflows into ComfyUI/user/default/workflows
    src_wf_dir = os.path.join(pkg_dir, "workflows")
    comfy_root = folder_paths.base_path
    dst_wf_dir = os.path.join(comfy_root, "user", "default", "workflows")
    if os.path.isdir(src_wf_dir):
        for root, _, files in os.walk(src_wf_dir):
            rel = os.path.relpath(root, src_wf_dir)
            target_dir = os.path.join(dst_wf_dir, rel) if rel != "." else dst_wf_dir
            os.makedirs(target_dir, exist_ok=True)
            for f in files:
                if f.endswith(".json"):
                    dst_f = os.path.join(target_dir, f)
                    if not os.path.exists(dst_f):
                        try:
                            shutil.copy2(os.path.join(root, f), dst_f)
                        except Exception as e:
                            print(f"[AIO-H200] Warning copying workflow {f}: {e}")

try:
    _bootstrap_bundled_assets()
except Exception:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
