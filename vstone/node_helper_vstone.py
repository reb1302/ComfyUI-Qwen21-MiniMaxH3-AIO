"""
VStone Node Helper Module (node_helper_vstone.py)
Chuyên dụng cho bài toán Image-to-Image Extend (Outpainting mở rộng khung ảnh qua phải) với Qwen-Image-2.1.
"""

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
def extend_image_right_tensor(
    image: torch.Tensor,
    frames_right: float = 1.0,
    extra_pixels_right: int = 0,
    fill_mode: str = "gray_0.5",
    feather_pixels: int = 32,
    overlap_pixels: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Nhận 1 ảnh đầu vào [B, H, W, C] và mở rộng qua bên phải (mặc định 1 khung = gấp đôi chiều ngang).

    Args:
        image: Tensor ảnh chuẩn ComfyUI [B, H, W, C] trong miền [0.0, 1.0].
        frames_right: Hệ số khung mở rộng qua phải (1.0 = mở rộng thêm đúng 1 khung bằng chiều rộng ảnh gốc W).
        extra_pixels_right: Số pixel cộng thêm tùy chỉnh (nếu > 0 sẽ cộng thêm vào phần mở rộng).
        fill_mode: Cách lấp vùng khung mới bên phải:
            - "gray_0.5": Màu xám trung tính 0.5 (điểm 0 chuẩn của VAE Qwen-Image-2.1).
            - "mirror_blur": Lật gương + làm mờ nhẹ để giữ đúng dải màu/ánh sáng môi trường bên phải.
            - "mirror_sharp": Lật gương nguyên bản qua phải.
            - "edge_stretch": Kéo giãn cột pixel ngoài cùng bên phải.
            - "white": Màu trắng (1.0).
            - "black": Màu đen (0.0).
        feather_pixels: Độ rộng vùng chuyển tiếp mượt (feather) tại biên ghép nối.
        overlap_pixels: Số pixel lấn nhẹ vào mép phải của ảnh gốc để AI vẽ nối liền mạch không bị đường kẻ dọc.

    Returns:
        (extended_image [B, H, new_W, C], outpaint_mask [B, H, new_W], orig_W)
    """
    if image.ndim != 4:
        raise ValueError(f"Expected 4D IMAGE tensor [B, H, W, C], got shape {tuple(image.shape)}")

    B, H, W, C = image.shape
    device = image.device
    dtype = image.dtype

    # Tính chiều rộng phần mở rộng qua phải
    ext_w = int(round(W * float(frames_right))) + int(extra_pixels_right)
    ext_w = max(32, ext_w)

    # Tạo phần khung mở rộng bên phải [B, H, ext_w, C]
    if fill_mode.startswith("gray_0.5"):
        right_pad = torch.full((B, H, ext_w, C), 0.5, dtype=dtype, device=device)
    elif fill_mode.startswith("mirror_blur") or fill_mode.startswith("mirror_sharp"):
        flipped = torch.flip(image, dims=[2])
        repeats = math.ceil(ext_w / W)
        right_pad = flipped.repeat(1, 1, repeats, 1)[:, :, :ext_w, :].clone()
        if fill_mode.startswith("mirror_blur"):
            # Làm mờ mạnh vùng gương để Qwen-Image-2.1 lấy ánh sáng/bối cảnh mà không lặp lại vật thể
            k = max(3, (min(H, ext_w) // 8) | 1)
            pad_nchw = right_pad.movedim(-1, 1)
            low_res = F.avg_pool2d(pad_nchw, kernel_size=k, stride=max(1, k // 2), padding=k // 2)
            right_pad = F.interpolate(low_res, size=(H, ext_w), mode="bilinear", align_corners=False).movedim(1, -1)
    elif fill_mode.startswith("edge_stretch"):
        right_edge = image[:, :, -1:, :]
        right_pad = right_edge.expand(B, H, ext_w, C).clone()
    elif fill_mode.startswith("white"):
        right_pad = torch.ones((B, H, ext_w, C), dtype=dtype, device=device)
    elif fill_mode.startswith("black"):
        right_pad = torch.zeros((B, H, ext_w, C), dtype=dtype, device=device)
    else:
        right_pad = torch.full((B, H, ext_w, C), 0.5, dtype=dtype, device=device)

    # Nếu ảnh là RGBA (4 kênh), đặt alpha vùng mở rộng chuẩn
    if C == 4 and fill_mode.startswith("gray_0.5"):
        right_pad[:, :, :, 3] = 1.0

    # Ghép ảnh gốc (trái) + khung mở rộng (phải) -> [B, H, W + ext_w, C]
    extended_image = torch.cat([image, right_pad], dim=2)
    new_W = W + ext_w

    # Tạo mặt nạ Outpaint [B, H, new_W]:
    # 0.0 = Giữ nguyên khung gốc bên trái
    # 1.0 = Vùng cần Qwen-Image-2.1 vẽ mở rộng bên phải
    mask = torch.zeros((B, H, new_W), dtype=dtype, device=device)
    seam_start = max(0, W - int(overlap_pixels))
    mask[:, :, seam_start:] = 1.0

    # Làm mềm biên (Cosine Feathering) từ [seam_start - feather_pixels] đến [seam_start]
    if feather_pixels > 0 and seam_start > 0:
        f_len = min(int(feather_pixels), seam_start)
        if f_len > 0:
            t = torch.linspace(0.0, 1.0, steps=f_len, dtype=dtype, device=device)
            ramp = 0.5 * (1.0 - torch.cos(math.pi * t))  # Mượt theo hàm Cosine
            mask[:, :, seam_start - f_len : seam_start] = ramp.view(1, 1, f_len)

    return extended_image, mask, W


def blend_original_left_tensor(
    original_image: torch.Tensor,
    generated_extended_image: torch.Tensor,
    outpaint_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Ghép khôi phục 100% độ nét của ảnh gốc ở khung bên trái vào ảnh đã mở rộng từ Qwen-Image-2.1,
    sử dụng mặt nạ feather để không có vết cắt ở giữa.
    """
    B, orig_H, orig_W, orig_C = original_image.shape
    _, gen_H, gen_W, gen_C = generated_extended_image.shape

    # Nếu Qwen-Image-2.1 resize ảnh theo bội số của 32, chuẩn hóa lại kích thước
    if gen_H != orig_H:
        scale = orig_H / float(gen_H)
        target_W = int(round(gen_W * scale))
        gen_nchw = generated_extended_image.movedim(-1, 1)
        gen_resized = F.interpolate(gen_nchw, size=(orig_H, target_W), mode="bicubic", align_corners=False).movedim(1, -1).clamp(0.0, 1.0)
    else:
        gen_resized = generated_extended_image
        target_W = gen_W

    # Căn chỉnh mặt nạ về đúng (orig_H, target_W)
    if outpaint_mask.shape[-2:] != (orig_H, target_W):
        m = outpaint_mask.unsqueeze(1)
        m = F.interpolate(m, size=(orig_H, target_W), mode="bilinear", align_corners=False).squeeze(1)
    else:
        m = outpaint_mask

    out = gen_resized.clone()
    copy_W = min(orig_W, target_W)
    alpha = m[:, :, :copy_W].unsqueeze(-1).to(out.device, out.dtype)
    orig_slice = original_image[:, :, :copy_W, :gen_C].to(out.device, out.dtype)

    # alpha = 0 -> giữ 100% original_image; alpha = 1 -> lấy 100% ảnh AI mới vẽ
    out[:, :, :copy_W, :gen_C] = orig_slice * (1.0 - alpha) + out[:, :, :copy_W, :gen_C] * alpha
    return out


# ==============================================================================
# COMFYUI NODE 1: VStone Extend Image Right (Mở rộng 1 khung qua phải)
# ==============================================================================
class VStone_ImageExtendRight:
    """
    Node nhận 1 ảnh đầu vào và mở rộng thêm 1 khung qua bên phải (mặc định gấp đôi chiều rộng)
    kèm mặt nạ Outpaint và gợi ý Prompt chuẩn cho Qwen-Image-2.1.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "frames_right": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 4.0,
                    "step": 0.1,
                    "tooltip": "1.0 = Mở rộng thêm đúng 1 khung qua bên phải (chiều rộng gấp đôi ảnh gốc)."
                }),
                "extra_pixels_right": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Số pixel cộng thêm vào bên phải (nếu muốn chỉnh theo số pixel cố định)."
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
                    "tooltip": "Độ mượt vùng chuyển tiếp giữa khung gốc bên trái và khung mới bên phải."
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
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("extended_image", "outpaint_mask", "original_image", "qwen_extend_prompt", "new_width", "new_height")
    FUNCTION = "extend_right"
    CATEGORY = "vstone/qwen_extend"

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
        extended_image, outpaint_mask, _ = extend_image_right_tensor(
            image=image,
            frames_right=frames_right,
            extra_pixels_right=extra_pixels_right,
            fill_mode=fill_mode,
            feather_pixels=feather_pixels,
            overlap_pixels=overlap_pixels,
        )
        new_h = int(extended_image.shape[1])
        new_w = int(extended_image.shape[2])

        qwen_prompt = (
            f"Keep the left half of <image1> completely unchanged and identical. "
            f"Outpaint and replace the right half of <image1> to {scene_description.strip()} "
            f"Ensure a seamless, natural continuation with no visible vertical seam or border."
        )
        return (extended_image, outpaint_mask, image, qwen_prompt, new_w, new_h)


# ==============================================================================
# COMFYUI NODE 2: VStone Seamless Restore Left (Khôi phục 100% khung gốc bên trái)
# ==============================================================================
class VStone_SeamlessRestoreLeft:
    """
    Ghép khôi phục 100% pixel gốc sắc nét ở khung bên trái với khung mới do Qwen-Image-2.1
    vẽ ở bên phải thông qua outpaint_mask.
    """

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
    RETURN_NAMES = ("final_extended_image",)
    FUNCTION = "restore_left"
    CATEGORY = "vstone/qwen_extend"

    def restore_left(
        self,
        original_image: torch.Tensor,
        generated_extended_image: torch.Tensor,
        outpaint_mask: torch.Tensor,
    ):
        blended = blend_original_left_tensor(
            original_image=original_image,
            generated_extended_image=generated_extended_image,
            outpaint_mask=outpaint_mask,
        )
        return (blended,)


# ==============================================================================
# COMFYUI NODE 3: VStone Qwen 2.1 Extend Right All-in-One (1-Node Outpaint)
# ==============================================================================
class VStone_Qwen21_ExtendRight_AIO:
    """
    Node All-in-One của VStone:
    Nhận 1 ảnh -> Tự mở rộng 1 khung qua phải -> Chạy Qwen-Image-2.1 (kèm Fun ControlNet Union Inpaint
    hoặc Inpaint Latent Mask) -> Tự hòa trộn giữ nguyên 100% ảnh gốc bên trái -> Xuất ra ảnh hoàn chỉnh.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "frames_right": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 3.0, "step": 0.25}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Seamlessly extend the scene to the right, matching the exact lighting, camera angle, and environment."
                }),
                "use_fun_controlnet_inpaint": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Bật Qwen-Image-2.1 Fun ControlNet Union ở chế độ Inpaint/Outpaint để khóa chặt khung bên trái."
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
    ):
        import folder_paths
        import nodes
        import comfy.sd
        import comfy.utils
        from comfy_extras.nodes_qwen import TextEncodeQwenImage21

        # Import model downloader helper
        try:
            from ..aio_nodes import ensure_model_exists
        except ImportError:
            from aio_nodes import ensure_model_exists

        # 1. Mở rộng ảnh 1 khung qua bên phải
        fill_mode = "gray_0.5" if use_fun_controlnet_inpaint else "mirror_blur"
        padded_image, outpaint_mask, _ = extend_image_right_tensor(
            image=image,
            frames_right=frames_right,
            extra_pixels_right=0,
            fill_mode=fill_mode,
            feather_pixels=feather_pixels,
            overlap_pixels=16,
        )

        # 2. Load Qwen-Image-2.1 Models
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

        # 3. Áp dụng Qwen-Image-2.1 Fun ControlNet Union (chế độ Inpaint/Outpaint) nếu bật
        if use_fun_controlnet_inpaint:
            from comfy_extras.nodes_model_patch import ModelPatchLoader, ZImageFunControlnet
            cnet_name = "qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors"
            # Đảm bảo file tồn tại trong model_patches
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

        # 4. Encode Prompt & Reference Image (<image1>)
        full_prompt = (
            f"Keep the left side of <image1> unchanged. Seamlessly extend and outpaint the right side: {prompt}"
        )
        enc_out = TextEncodeQwenImage21.execute(
            clip=clip,
            prompt=full_prompt,
            negative_prompt="",
            vae=vae,
            resolution=resolution,
            images={"image_1": padded_image},
        )
        positive, negative, latent_dict = enc_out[0], enc_out[1], enc_out[2]

        # 5. KSampler
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

        # 6. Hòa trộn giữ nguyên 100% ảnh gốc bên trái
        final_image = blend_original_left_tensor(
            original_image=image,
            generated_extended_image=decoded,
            outpaint_mask=outpaint_mask,
        )
        return (final_image, padded_image, outpaint_mask)


VSTONE_NODE_CLASS_MAPPINGS = {
    "VStone_ImageExtendRight": VStone_ImageExtendRight,
    "VStone_SeamlessRestoreLeft": VStone_SeamlessRestoreLeft,
    "VStone_Qwen21_ExtendRight_AIO": VStone_Qwen21_ExtendRight_AIO,
}

VSTONE_NODE_DISPLAY_NAME_MAPPINGS = {
    "VStone_ImageExtendRight": "VStone: Extend Image Right (1 Frame Outpaint Pad)",
    "VStone_SeamlessRestoreLeft": "VStone: Seamless Restore Left (Blend Original)",
    "VStone_Qwen21_ExtendRight_AIO": "VStone: Qwen 2.1 Extend Right All-in-One",
}
