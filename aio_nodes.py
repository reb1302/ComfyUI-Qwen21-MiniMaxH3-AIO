import os
import shutil
import subprocess
import urllib.request

# ==============================================================================
# MODEL REGISTRY (Official Comfy-Org HuggingFace Direct Links)
# ==============================================================================
HF_QWEN_BASE = "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main"
HF_MINIMAX_BASE = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"

MODEL_CATALOG = {
    # Qwen-Image-2.1 Core Models
    "qwen_image_2.1_int8_convrot.safetensors": ("diffusion_models", f"{HF_QWEN_BASE}/diffusion_models/qwen_image_2.1_int8_convrot.safetensors"),
    "qwen_image_2.1_turbo_int8_convrot.safetensors": ("diffusion_models", f"{HF_QWEN_BASE}/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors"),
    "qwen_image_2.1_bf16.safetensors": ("diffusion_models", f"{HF_QWEN_BASE}/diffusion_models/qwen_image_2.1_bf16.safetensors"),
    "qwen_image_2.1_turbo_bf16.safetensors": ("diffusion_models", f"{HF_QWEN_BASE}/diffusion_models/qwen_image_2.1_turbo_bf16.safetensors"),
    "qwen3vl_8b_int8_convrot.safetensors": ("text_encoders", f"{HF_QWEN_BASE}/text_encoders/qwen3vl_8b_int8_convrot.safetensors"),
    "qwen3vl_8b_bf16.safetensors": ("text_encoders", f"{HF_QWEN_BASE}/text_encoders/qwen3vl_8b_bf16.safetensors"),
    "qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors": ("text_encoders", f"{HF_QWEN_BASE}/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors"),
    "qwen_image_2.1_vae_bf16.safetensors": ("vae", f"{HF_QWEN_BASE}/vae/qwen_image_2.1_vae_bf16.safetensors"),
    "qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors": ("loras", f"{HF_QWEN_BASE}/loras/qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors"),
    "p_q_image_2.1_8_steps_v0.1_comfyui_bf16.safetensors": ("loras", f"{HF_QWEN_BASE}/loras/p_q_image_2.1_8_steps_v0.1_comfyui_bf16.safetensors"),

    # MiniMax-H3 Core Models
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors": ("diffusion_models", f"{HF_MINIMAX_BASE}/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors": ("diffusion_models", f"{HF_MINIMAX_BASE}/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
    "minimax_h3_fl2va_pruned_bf16.safetensors": ("diffusion_models", f"{HF_MINIMAX_BASE}/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors"),
    "minimax_h3_ref2va_pruned_bf16.safetensors": ("diffusion_models", f"{HF_MINIMAX_BASE}/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors"),
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": ("text_encoders", f"{HF_MINIMAX_BASE}/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
    "qwen3vl_32b_minimax_h3_int8_convrot.safetensors": ("text_encoders", f"{HF_MINIMAX_BASE}/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
    "qwen3vl_32b_minimax_h3_bf16.safetensors": ("text_encoders", f"{HF_MINIMAX_BASE}/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors"),
    "minimax_h3_video_vae_int8_convrot.safetensors": ("vae", f"{HF_MINIMAX_BASE}/vae/minimax_h3_video_vae_int8_convrot.safetensors"),
    "minimax_h3_video_vae_fp16.safetensors": ("vae", f"{HF_MINIMAX_BASE}/vae/minimax_h3_video_vae_fp16.safetensors"),
    "minimax_h3_audio_vae_fp32.safetensors": ("vae", f"{HF_MINIMAX_BASE}/vae/minimax_h3_audio_vae_fp32.safetensors"),
    "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors": ("loras", f"{HF_MINIMAX_BASE}/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"),
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors": ("loras", f"{HF_MINIMAX_BASE}/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"),
    "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors": ("loras", f"{HF_MINIMAX_BASE}/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"),
}


def ensure_model_exists(filename: str) -> str:
    """Resolve model path in ComfyUI or automatically download it via aria2c / curl / urllib."""
    import folder_paths
    if filename not in MODEL_CATALOG:
        raise ValueError(f"Unknown model filename: {filename}")
    folder_type, url = MODEL_CATALOG[filename]

    existing_path = folder_paths.get_full_path(folder_type, filename)
    if existing_path and os.path.exists(existing_path) and os.path.getsize(existing_path) > 1024 * 1024:
        return existing_path

    target_dirs = folder_paths.get_folder_paths(folder_type)
    target_dir = target_dirs[0] if target_dirs else os.path.join(folder_paths.models_dir, folder_type)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, filename)

    print(f"\n[AIO-H200] Auto-downloading missing model: {filename} -> {target_path}")
    aria2c_bin = shutil.which("aria2c")
    if aria2c_bin:
        cmd = [
            aria2c_bin, "--check-certificate=false", "--disable-ipv6=true",
            "--file-allocation=none", "-c", "-x", "16", "-s", "16", "-k", "1M",
            "-d", target_dir, "-o", filename, url
        ]
        subprocess.run(cmd, check=True)
    else:
        curl_bin = shutil.which("curl")
        if curl_bin:
            cmd = [curl_bin, "-4", "-L", "-C", "-", url, "-o", target_path]
            subprocess.run(cmd, check=True)
        else:
            urllib.request.urlretrieve(url, target_path)

    return target_path


# ==============================================================================
# NODE 1: All-in-One H200 Model Downloader & Environment Verifier
# ==============================================================================
class AIO_H200_ModelManager:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bundle_to_download": ([
                    "All_Core_12_Models (86GB - Recommended)",
                    "Qwen_Image_2.1_Only (25.8GB)",
                    "MiniMax_H3_Only (60.5GB)",
                    "H200_Max_Precision_BF16_Extras (+120GB)"
                ], {"default": "All_Core_12_Models (86GB - Recommended)"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status_report",)
    FUNCTION = "download_bundle"
    CATEGORY = "AIO H200 (Qwen 2.1 & MiniMax H3)"

    def download_bundle(self, bundle_to_download: str):
        qwen_core = [
            "qwen_image_2.1_int8_convrot.safetensors",
            "qwen3vl_8b_int8_convrot.safetensors",
            "qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors",
            "qwen_image_2.1_vae_bf16.safetensors",
            "qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors",
        ]
        minimax_core = [
            "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "minimax_h3_video_vae_int8_convrot.safetensors",
            "minimax_h3_audio_vae_fp32.safetensors",
            "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
            "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        ]
        bf16_extras = [
            "qwen_image_2.1_bf16.safetensors",
            "qwen3vl_8b_bf16.safetensors",
            "minimax_h3_fl2va_pruned_bf16.safetensors",
            "minimax_h3_ref2va_pruned_bf16.safetensors",
            "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        ]

        targets = []
        if bundle_to_download.startswith("All_Core"):
            targets = qwen_core + minimax_core
        elif bundle_to_download.startswith("Qwen"):
            targets = qwen_core
        elif bundle_to_download.startswith("MiniMax"):
            targets = minimax_core
        elif bundle_to_download.startswith("H200_Max"):
            targets = qwen_core + minimax_core + bf16_extras

        verified = []
        for fname in targets:
            path = ensure_model_exists(fname)
            size_gb = os.path.getsize(path) / (1024 ** 3)
            verified.append(f"[OK] {fname} ({size_gb:.2f} GB) -> {path}")

        report = "\n".join(verified)
        return (report,)


# ==============================================================================
# NODE 2: All-in-One Qwen-Image-2.1 Image-to-Image / Multi-Image Edit Node
# ==============================================================================
class Qwen21_AllInOne_I2I:
    """
    Single unified node for Qwen-Image-2.1 Image-to-Image & Multi-Image Editing.
    Automatically downloads/loads Checkpoint, Qwen3-VL Text Encoder, RGBA VAE, and Turbo LoRA,
    encodes up to 3 reference images (<image1>, <image2>, <image3>), runs sampling, and decodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Keep the character in <image1> consistent, change the background to a futuristic neon studio with soft cinematic lighting."
                }),
                "precision": (["int8_convrot (Fast & 99.5% Quality)", "bf16 (H200 Max Precision)"], {
                    "default": "int8_convrot (Fast & 99.5% Quality)"
                }),
                "turbo_mode": (["Off (Standard 28 Steps)", "Turbo_LoRA (8 Steps)", "Turbo_Checkpoint (8 Steps)"], {
                    "default": "Turbo_LoRA (8 Steps)"
                }),
                "resolution": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "run_qwen21_i2i"
    CATEGORY = "AIO H200 (Qwen 2.1 & MiniMax H3)"

    def run_qwen21_i2i(self, prompt, precision, turbo_mode, resolution, steps, cfg, seed,
                       image_1=None, image_2=None, image_3=None, negative_prompt=""):
        import folder_paths
        import nodes
        import comfy.sd
        import comfy.utils
        from comfy_extras.nodes_qwen import TextEncodeQwenImage21

        use_bf16 = precision.startswith("bf16")
        if turbo_mode == "Turbo_Checkpoint (8 Steps)":
            unet_name = "qwen_image_2.1_turbo_bf16.safetensors" if use_bf16 else "qwen_image_2.1_turbo_int8_convrot.safetensors"
        else:
            unet_name = "qwen_image_2.1_bf16.safetensors" if use_bf16 else "qwen_image_2.1_int8_convrot.safetensors"

        clip_name = "qwen3vl_8b_bf16.safetensors" if use_bf16 else "qwen3vl_8b_int8_convrot.safetensors"
        vae_name = "qwen_image_2.1_vae_bf16.safetensors"

        unet_path = ensure_model_exists(unet_name)
        clip_path = ensure_model_exists(clip_name)
        vae_path = ensure_model_exists(vae_name)

        # 1. Load Diffusion Model, CLIP (qwen_image), and VAE
        model = comfy.sd.load_diffusion_model(unet_path)
        if turbo_mode == "Turbo_LoRA (8 Steps)":
            lora_path = ensure_model_exists("qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors")
            lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model, _ = comfy.sd.load_lora_for_models(model, None, lora_sd, 1.0, 0.0)

        # Enable GPU KV Cache for H200
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

        # 2. Conditioning via TextEncodeQwenImage21 logic
        images_dict = {}
        if image_1 is not None:
            images_dict["image_1"] = image_1
        if image_2 is not None:
            images_dict["image_2"] = image_2
        if image_3 is not None:
            images_dict["image_3"] = image_3

        enc_out = TextEncodeQwenImage21.execute(
            clip=clip,
            prompt=prompt,
            negative_prompt=negative_prompt,
            vae=vae,
            resolution=resolution,
            images=images_dict
        )
        positive, negative, latent_dict = enc_out[0], enc_out[1], enc_out[2]

        # 3. KSampler Execution
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
            denoise=1.0
        )[0]

        # 4. Decode RGBA / RGB output
        decoded = vae.decode(sampled_latent["samples"])
        return (decoded,)


# ==============================================================================
# NODE 3: All-in-One MiniMax-H3 Loader & Conditioner (I2V & R2V + Audio)
# ==============================================================================
class MiniMaxH3_AllInOne_Pipeline:
    """
    Unified node for MiniMax-H3 (Image-to-Video FL2VA & Reference-to-Video R2V).
    Automatically downloads/loads Checkpoint, Qwen3-VL-32B Text Encoder, Video VAE, Audio VAE,
    applies Turbo LoRA & SigmaShift (12.0 video / 3.0 audio), and outputs ready-to-sample
    (MODEL, POSITIVE_COND, AV_LATENT, VIDEO_VAE, AUDIO_VAE).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_mode": ([
                    "Image_to_Video (FL2VA - First/Last Frame)",
                    "Reference_to_Video (R2V - Character/Style Lock)"
                ], {"default": "Image_to_Video (FL2VA - First/Last Frame)"}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Cinematic shot, natural fluid motion, realistic ambient sound and footsteps."
                }),
                "precision": (["pruned_int8_convrot (Standard)", "pruned_bf16 (H200 Max Precision)"], {
                    "default": "pruned_int8_convrot (Standard)"
                }),
                "turbo_lora": ([
                    "Turbo_8_Steps (Recommended)",
                    "Turbo_4_Steps (Ultra Fast)",
                    "Off (Full Steps)"
                ], {"default": "Turbo_8_Steps (Recommended)"}),
                "width": ("INT", {"default": 1344, "min": 256, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 32}),
                "length_frames": ("INT", {"default": 124, "min": 22, "max": 362, "step": 17}),
            },
            "optional": {
                "first_or_ref_image_1": ("IMAGE",),
                "last_or_ref_image_2": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT", "VAE", "VAE")
    RETURN_NAMES = ("model", "positive", "av_latent", "video_vae", "audio_vae")
    FUNCTION = "setup_minimax_h3"
    CATEGORY = "AIO H200 (Qwen 2.1 & MiniMax H3)"

    def setup_minimax_h3(self, task_mode, prompt, precision, turbo_lora, width, height, length_frames,
                         first_or_ref_image_1=None, last_or_ref_image_2=None):
        import folder_paths
        import comfy.sd
        import comfy.utils
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift, MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo

        is_r2v = task_mode.startswith("Reference_to_Video")
        use_bf16 = precision.startswith("pruned_bf16")

        if is_r2v:
            unet_name = "minimax_h3_ref2va_pruned_bf16.safetensors" if use_bf16 else "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
        else:
            unet_name = "minimax_h3_fl2va_pruned_bf16.safetensors" if use_bf16 else "minimax_h3_fl2va_pruned_int8_convrot.safetensors"

        clip_name = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
        v_vae_name = "minimax_h3_video_vae_int8_convrot.safetensors"
        a_vae_name = "minimax_h3_audio_vae_fp32.safetensors"

        unet_path = ensure_model_exists(unet_name)
        clip_path = ensure_model_exists(clip_name)
        v_vae_path = ensure_model_exists(v_vae_name)
        a_vae_path = ensure_model_exists(a_vae_name)

        # 1. Load Diffusion Model + Optional Turbo LoRA
        model = comfy.sd.load_diffusion_model(unet_path)
        if turbo_lora != "Off (Full Steps)":
            if is_r2v:
                lora_name = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
            elif turbo_lora.startswith("Turbo_4"):
                lora_name = "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
            else:
                lora_name = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
            lora_path = ensure_model_exists(lora_name)
            lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model, _ = comfy.sd.load_lora_for_models(model, None, lora_sd, 1.0, 0.0)

        # 2. Apply MiniMaxH3SigmaShift (shift_video=12.0, shift_audio=3.0)
        model = MiniMaxH3SigmaShift.execute(model, shift_video=12.0, shift_audio=3.0)[0]

        # 3. Load CLIP (type: minimax) & VAEs
        clip_type = getattr(comfy.sd.CLIPType, "MINIMAX")
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type
        )
        video_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(v_vae_path))
        audio_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(a_vae_path))

        # 4. Build Conditioning & Joint AV Latent
        if is_r2v:
            ref_images = {}
            if first_or_ref_image_1 is not None:
                ref_images["ref_image_1"] = first_or_ref_image_1
            if last_or_ref_image_2 is not None:
                ref_images["ref_image_2"] = last_or_ref_image_2
            cond_out = MiniMaxH3ReferenceToVideo.execute(
                clip=clip,
                prompt=prompt,
                width=width,
                height=height,
                length=length_frames,
                ref_image_size="match",
                vae=video_vae,
                audio_vae=audio_vae,
                ref_images=ref_images
            )
        else:
            cond_out = MiniMaxH3ImageToVideo.execute(
                clip=clip,
                vae=video_vae,
                prompt=prompt,
                width=width,
                height=height,
                length=length_frames,
                first_frame=first_or_ref_image_1,
                last_frame=last_or_ref_image_2
            )

        positive, av_latent = cond_out[0], cond_out[1]
        return (model, positive, av_latent, video_vae, audio_vae)


NODE_CLASS_MAPPINGS = {
    "AIO_H200_ModelManager": AIO_H200_ModelManager,
    "Qwen21_AllInOne_I2I": Qwen21_AllInOne_I2I,
    "MiniMaxH3_AllInOne_Pipeline": MiniMaxH3_AllInOne_Pipeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIO_H200_ModelManager": "AIO H200 Model Downloader & Verifier",
    "Qwen21_AllInOne_I2I": "Qwen-Image 2.1 All-in-One (Image Edit / I2I)",
    "MiniMaxH3_AllInOne_Pipeline": "MiniMax H3 All-in-One (I2V & R2V + Audio)",
}
