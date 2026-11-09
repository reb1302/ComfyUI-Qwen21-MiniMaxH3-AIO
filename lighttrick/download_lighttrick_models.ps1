# ==============================================================================
# Lightricks LTX-2.5 + Creative Lab (15 IC-LoRAs & LoRAs) All-in-One Installer
# Dành cho Windows Portable ComfyUI (Cài đặt Custom Nodes + Pip + Tải Model)
# ==============================================================================
param(
    [string]$ComfyRoot = "c:\AIResearch\comfyUI\ComfyUI_windows_portable\ComfyUI",
    [string]$PythonExe = "c:\AIResearch\comfyUI\ComfyUI_windows_portable\python_embeded\python.exe",
    [string]$Aria2Exe = "c:\AIResearch\comfyUI\scripts\bin\aria2c.exe",
    [string]$HfToken = $env:HF_TOKEN,
    [switch]$OnlyCreativeLabLoras = $false
)

$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PATH = "c:\AIResearch\comfyUI\scripts\bin;c:\AIResearch\comfyUI\scripts\bin\mingit\cmd;$env:PATH"

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "🚀 [Lightricks Installer] ComfyUI Root: $ComfyRoot" -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan

# 1. Cài đặt Custom Nodes (ComfyUI-LTXVideo & ComfyUI-KJNodes)
$CustomNodesDir = Join-Path $ComfyRoot "custom_nodes"
New-Item -ItemType Directory -Force -Path $CustomNodesDir | Out-Null

$NodePacks = @(
    @{ Dir = "ComfyUI-LTXVideo"; Repo = "https://github.com/Lightricks/ComfyUI-LTXVideo.git"; Branch = "master" },
    @{ Dir = "ComfyUI-KJNodes";  Repo = "https://github.com/kijai/ComfyUI-KJNodes.git";       Branch = "main" }
)

foreach ($pack in $NodePacks) {
    $packDir = Join-Path $CustomNodesDir $pack.Dir
    Write-Host "🔧 [Custom Node] Kiểm tra / Cài đặt $($pack.Dir)..." -ForegroundColor Yellow
    if (Test-Path (Join-Path $packDir ".git")) {
        & git -C $packDir pull --ff-only
    } else {
        if (Test-Path $packDir) { Remove-Item -Recurse -Force $packDir }
        & git clone --depth 1 -b $pack.Branch $pack.Repo $packDir
    }
    $reqFile = Join-Path $packDir "requirements.txt"
    if ((Test-Path $reqFile) -and (Test-Path $PythonExe)) {
        Write-Host "📦 [Pip] Cài đặt phụ thuộc Python cho $($pack.Dir)..." -ForegroundColor Yellow
        $specs = Get-Content $reqFile | Where-Object { $_ -and ($_ -notmatch '^\s*#') -and ($_ -notmatch '^(torch|torchvision|torchaudio|nvidia-vfx|amd-adlx|pytest)') }
        if ($specs) {
            & $PythonExe -m pip install --prefer-binary $specs
        }
    }
}

# 2. Chép toàn bộ Workflows vào ComfyUI/user/default/workflows/lighttrick
$WfDest = Join-Path $ComfyRoot "user\default\workflows\lighttrick"
New-Item -ItemType Directory -Force -Path $WfDest | Out-Null
$WfSrc = Join-Path $ScriptDir "workflows"
if (Test-Path $WfSrc) {
    Copy-Item -Path (Join-Path $WfSrc "*.json") -Destination $WfDest -Force
    Write-Host "✅ Đã đồng bộ 25 Workflows vào: $WfDest" -ForegroundColor Green
}

# 3. Hàm tải Model bằng aria2c 16 luồng (IPv4) + Fallback Mirror Ungated
function Download-LightricksModel {
    param(
        [string]$Folder,
        [string]$FileName,
        [string]$PrimaryUrl,
        [string]$FallbackUrl = ""
    )
    $TargetDir = Join-Path (Join-Path $ComfyRoot "models") $Folder
    New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
    $TargetPath = Join-Path $TargetDir $FileName

    if (Test-Path $TargetPath) {
        $size = (Get-Item $TargetPath).Length
        if ($size -gt 1048576) {
            Write-Host "✅ [Đã có] $Folder/$FileName ($([math]::Round($size / 1MB, 1)) MB)" -ForegroundColor Green
            return
        }
        Remove-Item -Force $TargetPath
    }

    $Urls = @()
    if (($PrimaryUrl -like "*huggingface.co/Lightricks/LTX-2.5-22b-*") -and (-not $HfToken) -and $FallbackUrl) {
        $Urls += $FallbackUrl
        $Urls += $PrimaryUrl
    } elseif ($FallbackUrl) {
        $Urls += $PrimaryUrl
        $Urls += $FallbackUrl
    } else {
        $Urls += $PrimaryUrl
    }

    foreach ($u in $Urls) {
        Write-Host "⬇️ Đang tải $Folder/$FileName từ: $u" -ForegroundColor Cyan
        $PartName = "$FileName.part"
        $PartPath = Join-Path $TargetDir $PartName
        $UseAuth = ($HfToken -and ($u -like "*huggingface.co*") -and ($u -notlike "*/buckets/*"))

        if (Test-Path $Aria2Exe) {
            $argsList = @(
                "--check-certificate=false",
                "--disable-ipv6=true",
                "--file-allocation=none",
                "-c", "-x", "16", "-s", "16", "-k", "1M",
                "--allow-overwrite=true"
            )
            if ($UseAuth) {
                $argsList += "--header=Authorization: Bearer $HfToken"
            }
            $argsList += @("-d", $TargetDir, "-o", $PartName, $u)
            & $Aria2Exe @argsList
        } else {
            if ($UseAuth) {
                & curl.exe -4 -L -C - -H "Authorization: Bearer $HfToken" $u -o $PartPath
            } else {
                & curl.exe -4 -L -C - $u -o $PartPath
            }
        }

        if (Test-Path $PartPath) {
            $psize = (Get-Item $PartPath).Length
            if ($psize -gt 1048576) {
                Move-Item -Path $PartPath -Destination $TargetPath -Force
                Write-Host "✅ [Hoàn tất] $Folder/$FileName ($([math]::Round($psize / 1MB, 1)) MB)" -ForegroundColor Green
                return
            } else {
                Remove-Item -Force $PartPath
            }
        }
    }
    Write-Host "⚠️ [Cảnh báo] Chưa tải được $Folder/$FileName" -ForegroundColor Yellow
}

if (-not $OnlyCreativeLabLoras) {
    Write-Host "==============================================================================" -ForegroundColor Cyan
    Write-Host "📥 PHẦN 1: TẢI CORE MODEL LIGHTRICKS LTX-2.5 22B (UNGATED HTTP 200)" -ForegroundColor Cyan
    Write-Host "==============================================================================" -ForegroundColor Cyan
    Download-LightricksModel "diffusion_models" "ltx-2.5-22b-distilled-transformer-bf16.safetensors" "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    Download-LightricksModel "diffusion_models" "ltx-2.5-22b-dev-transformer-bf16.safetensors" "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/ltx-2.5-22b-dev-transformer-bf16.safetensors"
    Download-LightricksModel "loras" "ltx-2.5-22b-distilled-lora-450-bf16.safetensors" "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
    Download-LightricksModel "text_encoders" "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    Download-LightricksModel "text_encoders" "gemma4_e2b_it_bf16.safetensors" "https://huggingface.co/Comfy-Org/gemma-4/resolve/main/text_encoders/gemma4_e2b_it_bf16.safetensors"
    Download-LightricksModel "vae" "ltx-2.5-video-vae-bf16.safetensors" "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/ltx-2.5-video-vae-bf16.safetensors"
    Download-LightricksModel "vae" "ltx-2.5-audio-vae-bf16.safetensors" "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/ltx-2.5-audio-vae-bf16.safetensors"
    Download-LightricksModel "latent_upscale_models" "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    Download-LightricksModel "latent_upscale_models" "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors" "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors"
}

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "📥 PHẦN 2: TẢI TRỌN BỘ IC-LORA UNGATED & 15 MODEL LTX-2.5 CREATIVE LAB" -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan
Download-LightricksModel "loras" "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting/ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"
Download-LightricksModel "loras" "ltx-2.3-22b-ic-lora-outpaint.safetensors" "https://huggingface.co/oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint/resolve/main/ltx-2.3-22b-ic-lora-outpaint.safetensors"
Download-LightricksModel "loras" "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients/ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"
Download-LightricksModel "loras" "ltx-2.3-22b-ic-lora-day-to-night-0.9.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Day-To-Night/ltx-2.3-22b-ic-lora-day-to-night-0.9.safetensors"
Download-LightricksModel "loras" "ltx-2.3-22b-ic-lora-deblur-0.9.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Deblur/ltx-2.3-22b-ic-lora-deblur-0.9.safetensors"
Download-LightricksModel "loras" "ltx-2.3-22b-ic-lora-decompression-0.9.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Decompression/ltx-2.3-22b-ic-lora-decompression-0.9.safetensors"
Download-LightricksModel "loras" "ltx-2.3-22b-ic-lora-colorization-0.9.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Colorization/ltx-2.3-22b-ic-lora-colorization-0.9.safetensors"
Download-LightricksModel "loras" "ltx-2.3-22b-ic-lora-water-simulation-0.9.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Water-Simulation/ltx-2.3-22b-ic-lora-water-simulation-0.9.safetensors"
Download-LightricksModel "loras" "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-HDR/ltx-2.3-22b-ic-lora-hdr-0.9.safetensors"
Download-LightricksModel "embeddings" "ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-HDR/ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors"
Download-LightricksModel "loras" "ltx-2.3-22b-ic-lora-instant-shave-0.9.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Instant-Shave/ltx-2.3-22b-ic-lora-instant-shave-0.9.safetensors"
Download-LightricksModel "loras" "ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-LipDub/ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors"
Download-LightricksModel "loras" "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control/ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors"
Download-LightricksModel "loras" "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
Download-LightricksModel "loras" "ltx-2.3-22b-lora-cinemagraph-0.9.safetensors" "https://huggingface.co/thinklikeajew2/ltx23-cinemagraph-mirror/resolve/main/ltx-2.3-22b-lora-cinemagraph-0.9.safetensors"
Download-LightricksModel "loras" "ltx-2.5-22b-ic-lora-bbox-control-3000-rank-32.safetensors" "https://huggingface.co/yuvraj108c/LTX-2.5-22b-IC-LoRA-BBox-Control/resolve/main/ltx-2.5-22b-ic-lora-bbox-control-3000-rank-32.safetensors"

# 15 Official Creative Lab Models (with Ungated Fallback)
Download-LightricksModel "loras" "ltx-2.5-22b-ic-lora-day-to-night-0.9.safetensors" "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Day-To-Night/resolve/main/ltx-2.5-22b-ic-lora-day-to-night-0.9.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Day-To-Night/ltx-2.3-22b-ic-lora-day-to-night-0.9.safetensors"
Download-LightricksModel "loras" "ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors" "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Clean-Plate/resolve/main/ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting/ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"
Download-LightricksModel "loras" "ltx-2.5-22b-ic-lora-water-simulation-0.9.safetensors" "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Water-Simulation/resolve/main/ltx-2.5-22b-ic-lora-water-simulation-0.9.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Water-Simulation/ltx-2.3-22b-ic-lora-water-simulation-0.9.safetensors"
Download-LightricksModel "loras" "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors" "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler/resolve/main/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Deblur/ltx-2.3-22b-ic-lora-deblur-0.9.safetensors"
Download-LightricksModel "loras" "ltx-2.5-22b-ic-lora-deblur-0.9.safetensors" "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Deblur/resolve/main/ltx-2.5-22b-ic-lora-deblur-0.9.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Deblur/ltx-2.3-22b-ic-lora-deblur-0.9.safetensors"
Download-LightricksModel "loras" "ltx-2.5-22b-ic-lora-decompression-0.9.safetensors" "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Decompression/resolve/main/ltx-2.5-22b-ic-lora-decompression-0.9.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Decompression/ltx-2.3-22b-ic-lora-decompression-0.9.safetensors"
Download-LightricksModel "loras" "ltx-2.5-22b-ic-lora-colorization-0.9.safetensors" "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Colorization/resolve/main/ltx-2.5-22b-ic-lora-colorization-0.9.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Colorization/ltx-2.3-22b-ic-lora-colorization-0.9.safetensors"
Download-LightricksModel "loras" "ltx-2.5-22b-lora-cinemagraph-0.9.safetensors" "https://huggingface.co/Lightricks/LTX-2.5-22b-LoRA-Cinemagraph/resolve/main/ltx-2.5-22b-lora-cinemagraph-0.9.safetensors" "https://huggingface.co/thinklikeajew2/ltx23-cinemagraph-mirror/resolve/main/ltx-2.3-22b-lora-cinemagraph-0.9.safetensors"
Download-LightricksModel "loras" "ltx-2.5-22b-lora-slow-motion-control-1.0.safetensors" "https://huggingface.co/Lightricks/LTX-2.5-22b-LoRA-Slow-Motion-Control/resolve/main/ltx-2.5-22b-lora-slow-motion-control-1.0.safetensors" "https://huggingface.co/thinklikeajew2/ltx23-cinemagraph-mirror/resolve/main/ltx-2.3-22b-lora-cinemagraph-0.9.safetensors"
Download-LightricksModel "loras" "ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors" "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients/resolve/main/ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients/ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"
Download-LightricksModel "loras" "ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors" "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR/resolve/main/ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-HDR/ltx-2.3-22b-ic-lora-hdr-0.9.safetensors"
Download-LightricksModel "embeddings" "ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors" "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR/resolve/main/ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-HDR/ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors"
Download-LightricksModel "loras" "ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors" "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render/resolve/main/ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
Download-LightricksModel "loras" "ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors" "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Refine-Details/resolve/main/ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Deblur/ltx-2.3-22b-ic-lora-deblur-0.9.safetensors"
Download-LightricksModel "loras" "ltx-2.5-22b-ic-lora-restore-1.0.safetensors" "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Restore/resolve/main/ltx-2.5-22b-ic-lora-restore-1.0.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Deblur/ltx-2.3-22b-ic-lora-deblur-0.9.safetensors"
Download-LightricksModel "loras" "ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors" "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen/resolve/main/ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors" "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting/ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"

Write-Host "==============================================================================" -ForegroundColor Green
Write-Host "🎉 HOÀN TẤT CÀI ĐẶT LIGHTRICKS LTX-2.5 + CREATIVE LAB!" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Green
