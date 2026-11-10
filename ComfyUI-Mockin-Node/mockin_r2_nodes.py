import asyncio
import base64
import concurrent.futures
import http.client
import io
import json
import mimetypes
import os
import shutil
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from aiohttp import web

try:
    import folder_paths
    COMFY_ROOT = os.path.abspath(folder_paths.base_path)
except Exception:
    COMFY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import server
except Exception:
    server = None

try:
    import numpy as np
    import torch
    from PIL import Image
    HAS_TORCH_PIL = True
except Exception:
    HAS_TORCH_PIL = False

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mockin_config.json")
DEFAULT_R2_API_BASE = "https://api.mockin.app/api/room"
DEFAULT_R2_CODE = "9999"
DEFAULT_ROOM_ID = "R-6ZZP"
BRIDGE_ROOM_ID = "R-LAPTOP-BRIDGE"
BRIDGE_NOTE_CMD = "laptop_cmd"
BRIDGE_NOTE_RES = "laptop_res"


class _IPv4HTTPConnection(http.client.HTTPConnection):
    def connect(self):
        last_err = None
        try:
            infos = socket.getaddrinfo(self.host, self.port, socket.AF_INET, socket.SOCK_STREAM)
        except Exception:
            infos = []
        for res in infos:
            af, socktype, proto, _, sa = res
            sock = None
            try:
                sock = socket.socket(af, socktype, proto)
                if self.timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                    sock.settimeout(self.timeout)
                sock.connect(sa)
                self.sock = sock
                return
            except OSError as e:
                last_err = e
                if sock is not None:
                    sock.close()
        if last_err:
            raise last_err
        super().connect()


class _IPv4HTTPSConnection(http.client.HTTPSConnection):
    def connect(self):
        _IPv4HTTPConnection.connect(self)
        server_hostname = self._tunnel_host if self._tunnel_host else self.host
        self.sock = self._context.wrap_socket(self.sock, server_hostname=server_hostname)


class _IPv4HTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_IPv4HTTPConnection, req)


class _IPv4HTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_IPv4HTTPSConnection, req, context=self._context)


_R2_IPV4_OPENER = urllib.request.build_opener(_IPv4HTTPHandler(), _IPv4HTTPSHandler())


def _r2_urlopen(req, timeout: int = 60):
    return _R2_IPV4_OPENER.open(req, timeout=timeout)


LOCAL_DIR_PRESETS = [
    "ComfyUI/",
    "ComfyUI/input",
    "ComfyUI/output",
    "ComfyUI/models/loras",
    "ComfyUI/models/checkpoints",
    "ComfyUI/models/diffusion_models",
    "ComfyUI/models/vae",
    "ComfyUI/models/text_encoders",
    "ComfyUI/models/clip",
    "ComfyUI/models/controlnet",
    "ComfyUI/models/upscale_models",
    "ComfyUI/user/default/workflows",
    "ComfyUI/custom_nodes",
    "Custom Path",
]

INVALID_ROOM_IDS = {
    "",
    "9999",
    "read_file_or_note",
    "sync_pull_to_local",
    "sync_push_from_local",
    "copy_on_r2",
    "move_on_r2",
    "mkdir_on_r2",
    "list_fs",
    "manifest_folder",
    "sync_download_to_local",
    "sync_upload_to_r2",
    "copy_folder_on_r2",
    "move_or_rename_on_r2",
    "delete_on_r2",
    "raw_api_request",
}


def load_r2_config() -> dict:
    cfg = {
        "r2_api_base": DEFAULT_R2_API_BASE,
        "r2_code": DEFAULT_R2_CODE,
        "default_room_id": DEFAULT_ROOM_ID,
        "project_path": COMFY_ROOT,
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
                b = str(saved.get("r2_api_base") or "").strip().rstrip("/")
                if b.startswith("http://") or b.startswith("https://"):
                    cfg["r2_api_base"] = b
                c = str(saved.get("r2_code") or "").strip()
                if c and not c.upper().startswith("R-") and c not in INVALID_ROOM_IDS and not c.startswith("http"):
                    cfg["r2_code"] = c
                r = str(saved.get("default_room_id") or "").strip()
                if r and r not in INVALID_ROOM_IDS and r != BRIDGE_ROOM_ID and not r.startswith("http") and "/" not in r:
                    cfg["default_room_id"] = r
                p = str(saved.get("project_path") or "").strip()
                if p:
                    cfg["project_path"] = p
        except Exception:
            pass
    return cfg


def save_r2_config(updates: dict):
    full_cfg = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                full_cfg = json.load(f)
        except Exception:
            full_cfg = {}
    for k, v in updates.items():
        if v is None or not str(v).strip():
            continue
        val = str(v).strip()
        if k == "r2_api_base" and not (val.startswith("http://") or val.startswith("https://")):
            continue
        if k == "r2_code" and (val.upper().startswith("R-") or val in INVALID_ROOM_IDS or val.startswith("http")):
            continue
        if k == "default_room_id" and (val in INVALID_ROOM_IDS or val == BRIDGE_ROOM_ID or val.startswith("http") or "/" in val):
            continue
        full_cfg[k] = val
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(full_cfg, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def resolve_local_dir(preset: str, custom_path: str = "", remote_subpath: str = "") -> str:
    cfg = load_r2_config()
    proj_root = cfg.get("project_path") or COMFY_ROOT
    custom = (custom_path or "").strip()
    if preset in ("ComfyUI/", "Project Root (Auto 1:1)", "", None):
        base = os.path.abspath(custom) if (custom and os.path.isabs(custom)) else os.path.abspath(proj_root)
        if remote_subpath:
            clean_rem = remote_subpath.strip().strip("/\\").replace("/", os.sep)
            if clean_rem:
                return os.path.abspath(os.path.join(base, clean_rem))
        return base
    if preset == "Custom Path" and custom:
        return os.path.abspath(custom)
    if custom and (os.path.isabs(custom) or preset == "Custom Path"):
        return os.path.abspath(custom)
    rel = (preset or "ComfyUI/input").strip()
    if rel.startswith("ComfyUI/"):
        rel = rel[len("ComfyUI/"):]
    base_dir = os.path.join(COMFY_ROOT, rel.replace("/", os.sep))
    if custom:
        return os.path.abspath(os.path.join(base_dir, custom.lstrip("/\\")))
    return os.path.abspath(base_dir)


def sanitize_r2_params(api_base: str = "", code: str = "", room_id: str = "") -> tuple:
    cfg = load_r2_config()
    b = str(api_base or "").strip().rstrip("/")
    if not (b.startswith("http://") or b.startswith("https://")):
        b = cfg.get("r2_api_base", DEFAULT_R2_API_BASE)
    c = str(code or "").strip()
    if not c or c.upper().startswith("R-") or c in INVALID_ROOM_IDS or c.startswith("http"):
        c = cfg.get("r2_code", DEFAULT_R2_CODE)
    r = str(room_id or "").strip()
    if r in INVALID_ROOM_IDS or r.startswith("http") or r.startswith("ComfyUI/") or "/" in r:
        raw_c = str(code or "").strip()
        if raw_c.upper().startswith("R-") and raw_c not in INVALID_ROOM_IDS:
            r = raw_c
        else:
            r = cfg.get("default_room_id", DEFAULT_ROOM_ID)
    return b, c, r


def build_r2_url(api_base: str, endpoint_path: str, query: dict = None) -> str:
    base, _, _ = sanitize_r2_params(api_base, "", "")
    ep = (endpoint_path or "").strip()
    if ep.startswith("http://") or ep.startswith("https://"):
        url = ep
    else:
        if ep.startswith("/api/room"):
            ep = ep[len("/api/room"):]
        if not ep.startswith("/"):
            ep = "/" + ep
        parts = ep.lstrip("/").split("/")
        if parts and parts[0] in INVALID_ROOM_IDS:
            _, _, fallback_room = sanitize_r2_params(api_base, "", "")
            parts[0] = fallback_room
            ep = "/" + "/".join(parts)
        url = base + ep
    if query:
        clean_q = {k: v for k, v in query.items() if v is not None and str(v) != ""}
        if clean_q:
            qs = urllib.parse.urlencode(clean_q)
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{qs}"
    return url


def r2_api_request(
    method: str,
    endpoint_path: str,
    api_base: str = DEFAULT_R2_API_BASE,
    code: str = DEFAULT_R2_CODE,
    query: dict = None,
    json_body: dict = None,
    raw_body: bytes = None,
    content_type: str = None,
    timeout: int = 45,
) -> dict:
    api_base, code, _ = sanitize_r2_params(api_base, code, "")
    url = build_r2_url(api_base, endpoint_path, query)
    headers = {
        "X-Room-Code": code,
        "User-Agent": "ComfyUI-Mockin-R2/2.0",
    }
    data_bytes = None
    if raw_body is not None:
        data_bytes = raw_body
        if content_type:
            headers["Content-Type"] = content_type
    elif json_body is not None:
        data_bytes = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    t0 = time.time()
    try:
        req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method.upper())
        with _r2_urlopen(req, timeout=timeout) as resp:
            elapsed_ms = round((time.time() - t0) * 1000, 1)
            raw = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            text = raw.decode("utf-8", errors="replace")
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and isinstance(parsed.get("rooms"), list):
                    parsed["rooms"] = [rm for rm in parsed["rooms"] if rm.get("room_id") != BRIDGE_ROOM_ID]
            except Exception:
                parsed = None
            return {
                "ok": True,
                "status": resp.status,
                "elapsed_ms": elapsed_ms,
                "url": url,
                "method": method.upper(),
                "content_type": ctype,
                "data": parsed if parsed is not None else text,
                "raw_text": text,
                "raw_bytes": raw,
            }
    except urllib.error.HTTPError as e:
        elapsed_ms = round((time.time() - t0) * 1000, 1)
        err_raw = e.read().decode("utf-8", errors="replace")
        try:
            err_parsed = json.loads(err_raw)
        except Exception:
            err_parsed = {"error": err_raw}
        return {
            "ok": False,
            "status": e.code,
            "elapsed_ms": elapsed_ms,
            "url": url,
            "method": method.upper(),
            "error": err_raw,
            "data": err_parsed,
            "raw_text": err_raw,
        }
    except Exception as e:
        elapsed_ms = round((time.time() - t0) * 1000, 1)
        return {
            "ok": False,
            "status": 0,
            "elapsed_ms": elapsed_ms,
            "url": url,
            "method": method.upper(),
            "error": str(e),
            "data": {"error": str(e)},
            "raw_text": str(e),
        }


def r2_fetch_file_bytes(
    room_id: str,
    remote_path: str,
    api_base: str = DEFAULT_R2_API_BASE,
    code: str = DEFAULT_R2_CODE,
) -> tuple:
    api_base, code, room_id = sanitize_r2_params(api_base, code, room_id)
    dl_url = build_r2_url(api_base, f"/{room_id}/fs/download", {"path": remote_path.lstrip("/")})
    req = urllib.request.Request(
        dl_url,
        headers={
            "X-Room-Code": str(code or DEFAULT_R2_CODE).strip(),
            "User-Agent": "ComfyUI-Mockin-R2/2.0",
        },
        method="GET",
    )
    with _r2_urlopen(req, timeout=300) as resp:
        data = resp.read()
        ctype = resp.headers.get("Content-Type") or mimetypes.guess_type(remote_path)[0] or "application/octet-stream"
        return data, ctype


def r2_download_single_file(
    room_id: str,
    remote_path: str,
    local_dest_path: str,
    api_base: str = DEFAULT_R2_API_BASE,
    code: str = DEFAULT_R2_CODE,
    direct_url: str = "",
) -> dict:
    os.makedirs(os.path.dirname(os.path.abspath(local_dest_path)), exist_ok=True)
    t0 = time.time()
    if direct_url:
        req = urllib.request.Request(direct_url, headers={"User-Agent": "ComfyUI-Mockin-R2/2.0"}, method="GET")
    else:
        dl_url = build_r2_url(api_base, f"/{room_id}/fs/download", {"path": remote_path.lstrip("/")})
        req = urllib.request.Request(
            dl_url,
            headers={
                "X-Room-Code": str(code or DEFAULT_R2_CODE).strip(),
                "User-Agent": "ComfyUI-Mockin-R2/2.0",
            },
            method="GET",
        )
    part_path = local_dest_path + ".part"
    with _r2_urlopen(req, timeout=300) as resp, open(part_path, "wb") as out_f:
        shutil.copyfileobj(resp, out_f, length=1024 * 1024)
    os.replace(part_path, local_dest_path)
    size_bytes = os.path.getsize(local_dest_path)
    return {
        "ok": True,
        "remote_path": remote_path,
        "local_path": os.path.abspath(local_dest_path),
        "size": size_bytes,
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
    }


def trigger_laptop_bridge_async(payload: dict):
    """Non-blocking relay to Laptop Bridge so workflow runs on H200 can also auto-sync to Laptop."""
    def _send():
        try:
            cmd_id = f"cmd_{int(time.time() * 1000)}"
            cmd_obj = {**payload, "cmd_id": cmd_id}
            r2_api_request(
                "POST",
                f"/{BRIDGE_ROOM_ID}/notes",
                json_body={
                    "id": BRIDGE_NOTE_CMD,
                    "title": "laptop_cmd",
                    "text": json.dumps(cmd_obj, ensure_ascii=False),
                },
                timeout=15,
            )
        except Exception:
            pass
    threading.Thread(target=_send, daemon=True).start()


def r2_sync_download(
    room_id: str,
    remote_path: str,
    local_target_dir: str,
    mode: str = "auto",
    skip_existing: bool = True,
    preserve_subpath: bool = False,
    api_base: str = DEFAULT_R2_API_BASE,
    code: str = DEFAULT_R2_CODE,
) -> dict:
    room_id = (room_id or "").strip()
    remote_clean = (remote_path or "").strip().strip("/")
    os.makedirs(local_target_dir, exist_ok=True)

    is_folder = mode == "folder"
    manifest_files = []
    if mode in ("auto", "folder"):
        manifest_check = r2_api_request(
            "GET",
            f"/{room_id}/fs/manifest",
            api_base=api_base,
            code=code,
            query={"path": remote_clean},
        )
        if manifest_check["ok"] and isinstance(manifest_check["data"], dict):
            manifest_files = manifest_check["data"].get("files", [])
            if not remote_clean:
                is_folder = True
            elif len(manifest_files) > 1 or (len(manifest_files) == 1 and manifest_files[0].get("path") != remote_clean):
                is_folder = True
            elif len(manifest_files) == 1 and manifest_files[0].get("path") == remote_clean:
                is_folder = False

    downloaded = []
    skipped = []
    errors = []

    if not is_folder and remote_clean:
        rel_dest = remote_clean if preserve_subpath else os.path.basename(remote_clean)
        dest_file = os.path.join(local_target_dir, rel_dest.replace("/", os.sep))
        direct_url = manifest_files[0].get("direct_download_url", "") if manifest_files else ""
        if skip_existing and os.path.exists(dest_file) and os.path.getsize(dest_file) > 0:
            skipped.append({"remote_path": remote_clean, "local_path": dest_file, "size": os.path.getsize(dest_file)})
        else:
            try:
                info = r2_download_single_file(room_id, remote_clean, dest_file, api_base=api_base, code=code, direct_url=direct_url)
                downloaded.append(info)
            except Exception as e:
                errors.append({"remote_path": remote_clean, "error": str(e)})
        return {
            "ok": len(errors) == 0,
            "mode": "file",
            "room_id": room_id,
            "remote_path": remote_clean,
            "local_path": dest_file if os.path.exists(dest_file) else local_target_dir,
            "downloaded_count": len(downloaded),
            "skipped_count": len(skipped),
            "downloaded": downloaded,
            "skipped": skipped,
            "errors": errors,
        }

    if not manifest_files:
        manifest_res = r2_api_request(
            "GET",
            f"/{room_id}/fs/manifest",
            api_base=api_base,
            code=code,
            query={"path": remote_clean},
        )
        if not manifest_res["ok"] or not isinstance(manifest_res["data"], dict):
            return {
                "ok": False,
                "mode": "folder",
                "room_id": room_id,
                "remote_path": remote_clean,
                "local_path": local_target_dir,
                "error": manifest_res.get("error", "Failed to fetch folder manifest"),
                "downloaded_count": 0,
                "skipped_count": 0,
                "downloaded": [],
                "skipped": [],
                "errors": [manifest_res.get("error", "Manifest error")],
            }
        manifest_files = manifest_res["data"].get("files", [])

    to_download = []
    for f_item in manifest_files:
        f_remote = f_item.get("path", "")
        f_rel = f_item.get("relative_path") or f_item.get("name") or os.path.basename(f_remote)
        if preserve_subpath and f_remote:
            f_rel = f_remote
        dest_file = os.path.join(local_target_dir, f_rel.replace("/", os.sep))
        expected_size = int(f_item.get("size", 0) or 0)
        if skip_existing and os.path.exists(dest_file):
            local_sz = os.path.getsize(dest_file)
            if (expected_size > 0 and local_sz == expected_size) or (expected_size == 0 and local_sz > 0):
                skipped.append({"remote_path": f_remote, "local_path": dest_file, "size": local_sz})
                continue
        to_download.append((f_remote, dest_file, f_item.get("direct_download_url", "")))

    if to_download:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            f_map = {
                pool.submit(
                    r2_download_single_file,
                    room_id,
                    f_rem,
                    dst_f,
                    api_base,
                    code,
                    d_url,
                ): f_rem
                for (f_rem, dst_f, d_url) in to_download
            }
            for fut in concurrent.futures.as_completed(f_map):
                f_rem = f_map[fut]
                try:
                    downloaded.append(fut.result())
                except Exception as e:
                    errors.append({"remote_path": f_rem, "error": str(e)})

    return {
        "ok": len(errors) == 0,
        "mode": "folder",
        "room_id": room_id,
        "remote_path": remote_clean,
        "local_path": local_target_dir,
        "manifest_total": len(manifest_files),
        "downloaded_count": len(downloaded),
        "skipped_count": len(skipped),
        "downloaded": downloaded,
        "skipped": skipped,
        "errors": errors,
    }


def r2_build_folder_zip_bytes(
    room_id: str,
    remote_path: str = "",
    api_base: str = DEFAULT_R2_API_BASE,
    code: str = DEFAULT_R2_CODE,
    include_root: bool = True,
) -> tuple:
    room_id = (room_id or DEFAULT_ROOM_ID).strip()
    remote_clean = (remote_path or "").strip().strip("/")
    manifest_res = r2_api_request(
        "GET",
        f"/{room_id}/fs/manifest",
        api_base=api_base,
        code=code,
        query={"path": remote_clean},
    )
    if not manifest_res["ok"] or not isinstance(manifest_res["data"], dict):
        raise RuntimeError(manifest_res.get("error") or "Failed to fetch folder manifest")

    manifest_files = manifest_res["data"].get("files") or []
    cache_root = os.path.join(COMFY_ROOT, "temp", "mockin_r2_cache", room_id)
    os.makedirs(cache_root, exist_ok=True)

    to_download = []
    cached_items = []
    folder_base = os.path.basename(remote_clean) if remote_clean else room_id

    for f_item in manifest_files:
        f_remote = (f_item.get("path") or "").strip().lstrip("/")
        if not f_remote:
            continue
        f_rel = (f_item.get("relative_path") or "").strip().lstrip("/")
        if not f_rel:
            if remote_clean and f_remote.startswith(remote_clean + "/"):
                f_rel = f_remote[len(remote_clean) + 1 :]
            else:
                f_rel = f_item.get("name") or os.path.basename(f_remote)

        arcname = f"{folder_base}/{f_rel}" if (include_root and remote_clean and f_rel != os.path.basename(remote_clean)) else f_rel
        local_cache_path = os.path.join(cache_root, f_remote.replace("/", os.sep))
        expected_size = int(f_item.get("size", 0) or 0)
        cached_items.append((arcname, local_cache_path, f_remote))

        if os.path.exists(local_cache_path):
            local_sz = os.path.getsize(local_cache_path)
            if (expected_size > 0 and local_sz == expected_size) or (expected_size == 0 and local_sz > 0):
                continue
        to_download.append((f_remote, local_cache_path, f_item.get("direct_download_url", "")))

    if to_download:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            futs = [
                pool.submit(
                    r2_download_single_file,
                    room_id,
                    f_rem,
                    dst_f,
                    api_base,
                    code,
                    d_url,
                )
                for (f_rem, dst_f, d_url) in to_download
            ]
            for fut in concurrent.futures.as_completed(futs):
                try:
                    fut.result()
                except Exception:
                    pass

    bio = io.BytesIO()
    packed_count = 0
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for arcname, local_cache_path, _ in cached_items:
            if os.path.exists(local_cache_path):
                zf.write(local_cache_path, arcname=arcname)
                packed_count += 1

    zip_fname = f"{folder_base or room_id}.zip"
    return bio.getvalue(), zip_fname, packed_count


def build_laptop_sync_agent_ps1_content(api_base: str = DEFAULT_R2_API_BASE, code: str = DEFAULT_R2_CODE) -> str:
    ps_script = r"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$StateFile = Join-Path $env:USERPROFILE ".mockin_laptop_sync.json"
$DefaultDir = Join-Path ([Environment]::GetFolderPath("Desktop")) "Mockin_Sync"
if (-not (Test-Path $DefaultDir)) { New-Item -ItemType Directory -Path $DefaultDir -Force | Out-Null }
$CurrentPath = $DefaultDir

if (Test-Path $StateFile) {
    try {
        $st = Get-Content $StateFile -Raw | ConvertFrom-Json
        if ($st.laptop_path -and (Test-Path $st.laptop_path)) { $CurrentPath = $st.laptop_path }
    } catch {}
}

function Save-State($p) {
    $script:CurrentPath = $p
    if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p -Force | Out-Null }
    @{ laptop_path = $p; label = (Split-Path $p -Leaf) } | ConvertTo-Json | Set-Content $StateFile -Encoding UTF8
}

function Pick-Folder-Dialog($initial) {
    $dlg = New-Object System.Windows.Forms.FolderBrowserDialog
    $dlg.Description = "Chon thu muc tren Laptop de Sync 1:1 voi ComfyUI Mockin R2"
    $dlg.ShowNewFolderButton = $true
    if ($initial -and (Test-Path $initial)) { $dlg.SelectedPath = $initial }
    $form = New-Object System.Windows.Forms.Form
    $form.TopMost = $true
    $res = $dlg.ShowDialog($form)
    $form.Dispose()
    if ($res -eq [System.Windows.Forms.DialogResult]::OK -and $dlg.SelectedPath) {
        Save-State $dlg.SelectedPath
        return $dlg.SelectedPath
    }
    return $null
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  MOCKIN R2 LAPTOP 1:1 SYNC AGENT (Chay tren Laptop)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Thu muc Sync hien tai: $CurrentPath" -ForegroundColor Green
Write-Host "Dang mo bang chon thu muc tren Laptop (bam Cancel neu giu nguyen)..." -ForegroundColor Yellow
$picked = Pick-Folder-Dialog $CurrentPath
if ($picked) {
    Write-Host "-> Da chon thu muc: $CurrentPath" -ForegroundColor Green
}

$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add("http://127.0.0.1:18189/")
try {
    $listener.Start()
    Write-Host "San sang! Quay lai trinh duyet ComfyUI va bam Sync / Set Project." -ForegroundColor Green
} catch {
    Write-Host "Cong 18189 dang chay san tren Laptop." -ForegroundColor Yellow
    exit
}

while ($listener.IsListening) {
    $ctx = $listener.GetContext()
    $req = $ctx.Request
    $resp = $ctx.Response
    $resp.Headers.Add("Access-Control-Allow-Origin", "*")
    $resp.Headers.Add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    $resp.Headers.Add("Access-Control-Allow-Headers", "*")
    $resp.Headers.Add("Access-Control-Allow-Private-Network", "true")

    if ($req.HttpMethod -eq "OPTIONS") {
        $resp.StatusCode = 204
        $resp.Close()
        continue
    }

    $bodyObj = @{}
    if ($req.HasEntityBody) {
        $reader = New-Object System.IO.StreamReader($req.InputStream, [System.Text.Encoding]::UTF8)
        $raw = $reader.ReadToEnd()
        $reader.Close()
        if ($raw) { try { $bodyObj = $raw | ConvertFrom-Json } catch {} }
    }

    $action = $req.Url.AbsolutePath.Trim("/")
    if ($bodyObj.action) { $action = $bodyObj.action }
    $out = @{ ok = $true; laptop_path = $CurrentPath; label = (Split-Path $CurrentPath -Leaf) }

    try {
        if ($action -eq "pick_folder") {
            $p = Pick-Folder-Dialog $CurrentPath
            if ($p) {
                $out = @{ ok = $true; laptop_path = $CurrentPath; label = (Split-Path $CurrentPath -Leaf) }
            } else {
                $out = @{ ok = $false; cancelled = $true; laptop_path = $CurrentPath; label = (Split-Path $CurrentPath -Leaf) }
            }
        } elseif ($action -eq "set_folder") {
            if ($bodyObj.laptop_path) { Save-State $bodyObj.laptop_path }
            $out = @{ ok = $true; laptop_path = $CurrentPath; label = (Split-Path $CurrentPath -Leaf) }
        } elseif ($action -eq "open_explorer") {
            Start-Process explorer.exe -ArgumentList $CurrentPath
            $out = @{ ok = $true; laptop_path = $CurrentPath; label = (Split-Path $CurrentPath -Leaf) }
        } elseif ($action -eq "sync_pull") {
            $apiBase = if ($bodyObj.api_base) { $bodyObj.api_base.TrimEnd("/") } else { "__DEFAULT_API_BASE__" }
            $code = if ($bodyObj.code) { $bodyObj.code } else { "__DEFAULT_CODE__" }
            $roomId = if ($bodyObj.room_id) { $bodyObj.room_id } else { "R-6ZZP" }
            $remPath = if ($bodyObj.remote_path) { $bodyObj.remote_path.Trim("/") } else { "" }
            $targetRoot = if ($bodyObj.laptop_path) { $bodyObj.laptop_path } else { $CurrentPath }
            Save-State $targetRoot

            Write-Host "[Sync] Dang tai '$remPath' (Room $roomId) -> $targetRoot ..." -ForegroundColor Cyan
            $dlCount = 0
            $skCount = 0
            $unpackedViaZip = $false

            if ($bodyObj.h200_zip_url -and $bodyObj.mode -ne "file") {
                try {
                    $tmpZip = Join-Path $env:TEMP ("mockin_sync_" + [guid]::NewGuid().ToString("N") + ".zip")
                    Invoke-WebRequest -Uri $bodyObj.h200_zip_url -OutFile $tmpZip -UseBasicParsing -TimeoutSec 180
                    $za = [System.IO.Compression.ZipFile]::OpenRead($tmpZip)
                    foreach ($entry in $za.Entries) {
                        if (-not $entry.Name) { continue }
                        $rel = $entry.FullName -replace "/", "\"
                        $destFile = Join-Path $targetRoot $rel
                        $destDir = Split-Path $destFile -Parent
                        if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir -Force | Out-Null }
                        if ($bodyObj.skip_existing -and (Test-Path $destFile)) {
                            $localSz = (Get-Item $destFile).Length
                            if ($entry.Length -gt 0 -and $localSz -eq $entry.Length) {
                                $skCount++
                                continue
                            }
                        }
                        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $destFile, $true)
                        $dlCount++
                    }
                    $za.Dispose()
                    Remove-Item $tmpZip -Force -ErrorAction SilentlyContinue
                    $unpackedViaZip = $true
                } catch {
                    if ($tmpZip -and (Test-Path $tmpZip)) { Remove-Item $tmpZip -Force -ErrorAction SilentlyContinue }
                }
            }

            if (-not $unpackedViaZip) {
                $mfUrl = "$apiBase/$roomId/fs/manifest?path=$([uri]::EscapeDataString($remPath))"
                $mfResp = Invoke-RestMethod -Uri $mfUrl -Headers @{ "X-Room-Code" = $code } -Method Get -TimeoutSec 30
                $files = $mfResp.files
                $folderBase = if ($remPath) { Split-Path $remPath -Leaf } else { "" }

                foreach ($f in $files) {
                    $fPath = $f.path.TrimStart("/")
                    $rel = if ($f.relative_path) { $f.relative_path.TrimStart("/") } else { Split-Path $fPath -Leaf }
                    if ($files.Count -gt 1 -and $folderBase -and (-not $rel.StartsWith("$folderBase/"))) {
                        $rel = "$folderBase/$rel"
                    }
                    $destFile = Join-Path $targetRoot ($rel -replace "/", "\")
                    $destDir = Split-Path $destFile -Parent
                    if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir -Force | Out-Null }
                    if ($bodyObj.skip_existing -and (Test-Path $destFile)) {
                        $localSz = (Get-Item $destFile).Length
                        if ($f.size -gt 0 -and $localSz -eq $f.size) {
                            $skCount++
                            continue
                        }
                    }
                    $dlUrl = if ($f.direct_download_url) { $f.direct_download_url } else { "$apiBase/$roomId/fs/download?path=$([uri]::EscapeDataString($fPath))" }
                    $hdrs = if ($f.direct_download_url) { @{} } else { @{ "X-Room-Code" = $code } }
                    Invoke-WebRequest -Uri $dlUrl -Headers $hdrs -OutFile $destFile -UseBasicParsing -TimeoutSec 180
                    $dlCount++
                    Write-Host "  + $rel" -ForegroundColor Gray
                }
            }
            Write-Host "[Hoan tat] Da luu $dlCount file moi ($skCount file da co) vao: $targetRoot" -ForegroundColor Green
            Start-Process explorer.exe -ArgumentList $targetRoot
            $out = @{
                ok = $true
                laptop_path = $targetRoot
                label = (Split-Path $targetRoot -Leaf)
                downloaded_count = $dlCount
                skipped_count = $skCount
            }
        }
    } catch {
        $out = @{ ok = $false; error = $_.Exception.Message; laptop_path = $CurrentPath }
    }

    $jsonBytes = [System.Text.Encoding]::UTF8.GetBytes(($out | ConvertTo-Json -Compress))
    $resp.ContentType = "application/json; charset=utf-8"
    $resp.ContentLength64 = $jsonBytes.Length
    $resp.OutputStream.Write($jsonBytes, 0, $jsonBytes.Length)
    $resp.Close()
}
""".replace("__DEFAULT_API_BASE__", api_base).replace("__DEFAULT_CODE__", code).strip()
    return ps_script


def build_laptop_sync_agent_bat_content(api_base: str = DEFAULT_R2_API_BASE, code: str = DEFAULT_R2_CODE) -> str:
    ps_script = build_laptop_sync_agent_ps1_content(api_base, code)
    b64_ps = base64.b64encode(ps_script.encode("utf-16le")).decode("ascii")
    return (
        "@echo off\r\n"
        "title Mockin R2 Laptop 1:1 Sync Agent\r\n"
        f"powershell.exe -NoProfile -ExecutionPolicy Bypass -STA -EncodedCommand {b64_ps}\r\n"
    )


def r2_sync_upload(
    room_id: str,
    local_source_path: str,
    remote_base_dir: str = "",
    api_base: str = DEFAULT_R2_API_BASE,
    code: str = DEFAULT_R2_CODE,
) -> dict:
    room_id = (room_id or "").strip()
    src = os.path.abspath((local_source_path or "").strip())
    base_dir = (remote_base_dir or "").strip().strip("/")

    if not os.path.exists(src):
        return {"ok": False, "error": f"Local path does not exist: {src}", "uploaded_count": 0}

    local_files = []
    if os.path.isfile(src):
        rel_name = os.path.basename(src)
        ctype = mimetypes.guess_type(src)[0] or "application/octet-stream"
        local_files.append({
            "abs_path": src,
            "rel_path": rel_name,
            "size": os.path.getsize(src),
            "content_type": ctype,
        })
    else:
        for root, _, files in os.walk(src):
            for fn in files:
                if fn.endswith(".part"):
                    continue
                full_p = os.path.join(root, fn)
                rel_p = os.path.relpath(full_p, src).replace("\\", "/")
                ctype = mimetypes.guess_type(full_p)[0] or "application/octet-stream"
                local_files.append({
                    "abs_path": full_p,
                    "rel_path": rel_p,
                    "size": os.path.getsize(full_p),
                    "content_type": ctype,
                })

    if not local_files:
        return {"ok": False, "error": f"No files found in: {src}", "uploaded_count": 0}

    meta_list = [
        {"path": item["rel_path"], "size": item["size"], "content_type": item["content_type"]}
        for item in local_files
    ]

    presign_res = r2_api_request(
        "POST",
        f"/{room_id}/files/presign",
        api_base=api_base,
        code=code,
        json_body={"base_dir": base_dir, "files": meta_list},
    )
    if not presign_res["ok"] or not isinstance(presign_res["data"], dict):
        return {
            "ok": False,
            "error": f"Presign failed: {presign_res.get('error')}",
            "uploaded_count": 0,
            "presign_response": presign_res.get("data"),
        }

    uploads = presign_res["data"].get("uploads", [])
    uploaded_items = []
    errors = []

    for idx, f_meta in enumerate(local_files):
        if idx >= len(uploads):
            errors.append({"file": f_meta["rel_path"], "error": "Missing upload_url in presign response"})
            continue
        u_info = uploads[idx]
        put_url = u_info.get("upload_url", "")
        ctype = u_info.get("content_type") or f_meta["content_type"]
        try:
            with open(f_meta["abs_path"], "rb") as fp:
                file_bytes = fp.read()
            put_req = urllib.request.Request(
                put_url,
                data=file_bytes,
                headers={
                    "Content-Type": ctype,
                    "Content-Length": str(len(file_bytes)),
                    "User-Agent": "ComfyUI-Mockin-R2/2.0",
                },
                method="PUT",
            )
            with _r2_urlopen(put_req, timeout=600) as _:
                pass
            uploaded_items.append(u_info)
        except Exception as e:
            errors.append({"file": f_meta["rel_path"], "error": str(e)})

    confirm_res = None
    if uploaded_items:
        confirm_res = r2_api_request(
            "POST",
            f"/{room_id}/files/confirm",
            api_base=api_base,
            code=code,
            json_body={"base_dir": "", "files": uploaded_items},
        )

    return {
        "ok": len(errors) == 0 and len(uploaded_items) > 0,
        "room_id": room_id,
        "remote_base_dir": base_dir,
        "uploaded_count": len(uploaded_items),
        "uploaded_paths": [u.get("path", "") for u in uploaded_items],
        "errors": errors,
        "confirm_status": confirm_res.get("status") if confirm_res else None,
    }


def make_empty_image_tensor():
    if not HAS_TORCH_PIL:
        return None
    return torch.zeros((1, 64, 64, 3), dtype=torch.float32)


def load_image_tensor_from_path(path: str):
    if not HAS_TORCH_PIL or not path or not os.path.exists(path):
        return make_empty_image_tensor()
    try:
        img = Image.open(path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr)[None,]
    except Exception:
        return make_empty_image_tensor()


# Register Backend HTTP Routes on ComfyUI PromptServer
try:
    if server is not None and getattr(getattr(server, "PromptServer", None), "instance", None) is not None:
        routes = server.PromptServer.instance.routes
        # Clean up any prior /mockin/r2 routes in PromptServer.instance.routes before registering fresh handlers
        if hasattr(routes, "_items") and isinstance(routes._items, list):
            routes._items[:] = [
                it for it in routes._items
                if not (hasattr(it, "path") and str(getattr(it, "path", "")).startswith("/mockin/r2"))
            ]

        @routes.get("/mockin/r2/config")
        async def get_r2_cfg_route(request):
            cfg = load_r2_config()
            return web.json_response({"ok": True, **cfg, "comfy_root": COMFY_ROOT, "presets": LOCAL_DIR_PRESETS})

        @routes.post("/mockin/r2/config")
        async def post_r2_cfg_route(request):
            data = await request.json()
            save_r2_config(data)
            return web.json_response({"ok": True, **load_r2_config()})

        @routes.post("/mockin/r2/proxy")
        async def post_r2_proxy_route(request):
            data = await request.json()
            cfg = load_r2_config()
            api_base = data.get("api_base") or cfg["r2_api_base"]
            code = data.get("code") or cfg["r2_code"]
            method = data.get("method", "GET")
            endpoint_path = data.get("endpoint_path", "/list")
            query = data.get("query") or {}
            json_body = data.get("body")
            if data.get("save_config"):
                save_r2_config({
                    "r2_api_base": api_base,
                    "r2_code": code,
                    "default_room_id": data.get("room_id"),
                })
            res = await asyncio.to_thread(
                r2_api_request,
                method=method,
                endpoint_path=endpoint_path,
                api_base=api_base,
                code=code,
                query=query,
                json_body=json_body,
            )
            res.pop("raw_bytes", None)
            return web.json_response(res)

        @routes.get("/mockin/r2/stream_file")
        async def get_r2_stream_file_route(request):
            cfg = load_r2_config()
            api_base = request.query.get("api_base") or cfg["r2_api_base"]
            code = request.query.get("code") or cfg["r2_code"]
            room_id = (request.query.get("room_id") or cfg["default_room_id"]).strip()
            remote_path = (request.query.get("path") or "").strip().lstrip("/")
            inline = request.query.get("inline", "0") == "1"
            if not remote_path:
                return web.json_response({"ok": False, "error": "Missing path"}, status=400)
            try:
                file_bytes, ctype = await asyncio.to_thread(
                    r2_fetch_file_bytes,
                    room_id=room_id,
                    remote_path=remote_path,
                    api_base=api_base,
                    code=code,
                )
                fname = os.path.basename(remote_path) or "download.bin"
                disp_type = "inline" if inline else "attachment"
                headers = {
                    "Content-Disposition": f'{disp_type}; filename="{fname}"',
                    "Cache-Control": "no-cache",
                }
                return web.Response(body=file_bytes, content_type=ctype.split(";")[0], headers=headers)
            except Exception as e:
                return web.json_response({"ok": False, "error": str(e)}, status=400)

        @routes.get("/mockin/r2/stream_folder_zip")
        async def get_r2_stream_folder_zip_route(request):
            cfg = load_r2_config()
            api_base = request.query.get("api_base") or cfg["r2_api_base"]
            code = request.query.get("code") or cfg["r2_code"]
            room_id = (request.query.get("room_id") or cfg["default_room_id"]).strip()
            remote_path = (request.query.get("path") or request.query.get("remote_path") or "").strip().strip("/")
            include_root = request.query.get("include_root", "1") != "0"
            try:
                zip_bytes, zip_fname, packed_count = await asyncio.to_thread(
                    r2_build_folder_zip_bytes,
                    room_id=room_id,
                    remote_path=remote_path,
                    api_base=api_base,
                    code=code,
                    include_root=include_root,
                )
                headers = {
                    "Content-Disposition": f'attachment; filename="{zip_fname}"',
                    "X-File-Count": str(packed_count),
                    "Cache-Control": "no-cache",
                }
                return web.Response(body=zip_bytes, content_type="application/zip", headers=headers)
            except Exception as e:
                return web.json_response({"ok": False, "error": str(e)}, status=400)

        @routes.get("/mockin/r2/laptop_sync_agent.ps1")
        async def get_r2_laptop_sync_agent_ps1_route(request):
            cfg = load_r2_config()
            ps_content = build_laptop_sync_agent_ps1_content(cfg["r2_api_base"], cfg["r2_code"])
            return web.Response(body=ps_content.encode("utf-8"), content_type="text/plain; charset=utf-8", headers={"Cache-Control": "no-cache"})

        @routes.get("/mockin/r2/laptop_sync_agent.bat")
        async def get_r2_laptop_sync_agent_bat_route(request):
            cfg = load_r2_config()
            bat_content = build_laptop_sync_agent_bat_content(cfg["r2_api_base"], cfg["r2_code"])
            headers = {
                "Content-Disposition": 'attachment; filename="Mockin_Laptop_Sync.bat"',
                "Cache-Control": "no-cache",
            }
            return web.Response(body=bat_content.encode("utf-8"), content_type="application/octet-stream", headers=headers)

        @routes.post("/mockin/r2/upload_blob")
        async def post_r2_upload_blob_route(request):
            upload_url = request.headers.get("X-Upload-Url", "").strip()
            ctype = request.headers.get("X-Upload-Content-Type", "application/octet-stream").strip()
            raw_bytes = await request.read()
            if not upload_url:
                return web.json_response({"ok": False, "error": "Missing X-Upload-Url"}, status=400)

            def _do_put():
                put_req = urllib.request.Request(
                    upload_url,
                    data=raw_bytes,
                    headers={
                        "Content-Type": ctype,
                        "Content-Length": str(len(raw_bytes)),
                        "User-Agent": "ComfyUI-Mockin-R2/2.0",
                    },
                    method="PUT",
                )
                with _r2_urlopen(put_req, timeout=600) as _:
                    pass

            try:
                await asyncio.to_thread(_do_put)
                return web.json_response({"ok": True, "size": len(raw_bytes)})
            except Exception as e:
                return web.json_response({"ok": False, "error": str(e)}, status=400)

        @routes.post("/mockin/r2/laptop_cmd")
        async def post_r2_laptop_cmd_route(request):
            data = await request.json()
            cfg = load_r2_config()
            api_base = data.get("api_base") or cfg["r2_api_base"]
            code = data.get("code") or cfg["r2_code"]
            action = str(data.get("action") or "get_status").strip()

            rm_res = await asyncio.to_thread(
                r2_api_request,
                "GET",
                f"/{BRIDGE_ROOM_ID}",
                api_base=api_base,
                code=code,
                timeout=6,
            )
            agent_alive = False
            latest_parsed = None
            if rm_res.get("ok") and isinstance(rm_res.get("data"), dict):
                rm_data = rm_res["data"]
                notes = rm_data.get("notes") or []
                res_note = next((n for n in notes if isinstance(n, dict) and n.get("id") == BRIDGE_NOTE_RES), None)
                for candidate in [res_note.get("text") if res_note else "", rm_data.get("text") or ""]:
                    s = str(candidate or "").strip()
                    if s.startswith("{"):
                        try:
                            parsed = json.loads(s)
                            latest_parsed = parsed
                            ts_val = float(parsed.get("ts") or 0)
                            if ts_val > 0 and (time.time() - ts_val) < 45.0:
                                agent_alive = True
                            break
                        except Exception:
                            pass

            if action == "get_status":
                if agent_alive and latest_parsed and latest_parsed.get("laptop_path"):
                    return web.json_response({
                        "ok": True,
                        "laptop_path": latest_parsed["laptop_path"],
                        "label": latest_parsed.get("label") or os.path.basename(latest_parsed["laptop_path"].rstrip("\\/")),
                    })
                return web.json_response({"ok": False, "no_agent": True})

            if not agent_alive:
                return web.json_response({"ok": False, "no_agent": True, "error": "Laptop Agent not running"})

            cmd_id = f"cmd_{int(time.time() * 1000)}_{os.urandom(2).hex()}"
            cmd_payload = {**data, "action": action, "cmd_id": cmd_id}
            send_res = await asyncio.to_thread(
                r2_api_request,
                "POST",
                f"/{BRIDGE_ROOM_ID}/notes",
                api_base=api_base,
                code=code,
                json_body={
                    "id": BRIDGE_NOTE_CMD,
                    "title": "laptop_cmd",
                    "text": json.dumps(cmd_payload, ensure_ascii=False),
                },
                timeout=15,
            )
            if not send_res.get("ok"):
                return web.json_response({"ok": False, "error": send_res.get("error", "Bridge relay error")})

            max_wait = 180.0 if action in ("pick_folder", "sync_pull", "sync_push") else 8.0
            deadline = time.time() + max_wait
            while time.time() < deadline:
                await asyncio.sleep(0.45)
                poll_res = await asyncio.to_thread(
                    r2_api_request,
                    "GET",
                    f"/{BRIDGE_ROOM_ID}",
                    api_base=api_base,
                    code=code,
                    timeout=10,
                )
                if poll_res.get("ok") and isinstance(poll_res.get("data"), dict):
                    notes = poll_res["data"].get("notes") or []
                    res_note = next((n for n in notes if isinstance(n, dict) and n.get("id") == BRIDGE_NOTE_RES), None)
                    if res_note:
                        raw_txt = str(res_note.get("text") or "").strip()
                        if raw_txt.startswith("{"):
                            try:
                                parsed = json.loads(raw_txt)
                                if str(parsed.get("cmd_id") or "") == cmd_id:
                                    return web.json_response(parsed)
                            except Exception:
                                pass

            return web.json_response({"ok": False, "error": "Laptop Bridge timeout"})

        @routes.post("/mockin/r2/read_file")
        async def post_r2_read_file_route(request):
            data = await request.json()
            cfg = load_r2_config()
            api_base = data.get("api_base") or cfg["r2_api_base"]
            code = data.get("code") or cfg["r2_code"]
            room_id = (data.get("room_id") or cfg["default_room_id"]).strip()
            remote_path = (data.get("remote_path") or "").strip().lstrip("/")
            save_local = bool(data.get("save_local", True))
            preset = data.get("target_preset", "ComfyUI/input")
            custom_dir = data.get("custom_local_dir", "")

            if not remote_path:
                return web.json_response({"ok": False, "error": "remote_path is empty"}, status=400)

            target_dir = resolve_local_dir(preset, custom_dir) if save_local else os.path.join(COMFY_ROOT, "temp", "mockin_r2_cache")
            local_file = os.path.join(target_dir, os.path.basename(remote_path))

            try:
                dl_info = await asyncio.to_thread(
                    r2_download_single_file,
                    room_id,
                    remote_path,
                    local_file,
                    api_base=api_base,
                    code=code,
                )
                ext = os.path.splitext(local_file)[1].lower()
                text_exts = {
                    ".txt", ".md", ".json", ".yaml", ".yml", ".py", ".js", ".csv",
                    ".log", ".toml", ".ini", ".xml", ".html", ".css", ".sh", ".bat", ".ps1",
                }
                img_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

                preview_type = "binary"
                text_content = ""
                data_url = ""

                if ext in text_exts or dl_info["size"] < 512 * 1024:
                    try:
                        with open(local_file, "r", encoding="utf-8") as f:
                            text_content = f.read()
                        preview_type = "text"
                    except UnicodeDecodeError:
                        preview_type = "binary"

                if ext in img_exts and dl_info["size"] < 25 * 1024 * 1024:
                    preview_type = "image"
                    mime = mimetypes.guess_type(local_file)[0] or "image/png"
                    with open(local_file, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("ascii")
                    data_url = f"data:{mime};base64,{b64}"

                return web.json_response({
                    "ok": True,
                    "room_id": room_id,
                    "remote_path": remote_path,
                    "local_path": dl_info["local_path"],
                    "size": dl_info["size"],
                    "elapsed_ms": dl_info["elapsed_ms"],
                    "preview_type": preview_type,
                    "text_content": text_content,
                    "data_url": data_url,
                })
            except Exception as e:
                return web.json_response({"ok": False, "error": str(e)}, status=400)

        @routes.post("/mockin/r2/download_sync")
        async def post_r2_download_sync_route(request):
            data = await request.json()
            cfg = load_r2_config()
            api_base = data.get("api_base") or cfg["r2_api_base"]
            code = data.get("code") or cfg["r2_code"]
            room_id = (data.get("room_id") or cfg["default_room_id"]).strip()
            remote_path = (data.get("remote_path") or "").strip()
            mode = data.get("mode", "auto")
            skip_existing = bool(data.get("skip_existing", True))
            preserve_subpath = bool(data.get("preserve_subpath", False))
            preset = data.get("target_preset", "ComfyUI/")
            custom_dir = data.get("custom_local_dir", "")
            if preset in ("ComfyUI/", "Project Root (Auto 1:1)", "", None):
                preserve_subpath = True

            target_dir = resolve_local_dir(preset, custom_dir)
            res = await asyncio.to_thread(
                r2_sync_download,
                room_id=room_id,
                remote_path=remote_path,
                local_target_dir=target_dir,
                mode=mode,
                skip_existing=skip_existing,
                preserve_subpath=preserve_subpath,
                api_base=api_base,
                code=code,
            )
            return web.json_response(res)

        @routes.post("/mockin/r2/upload_local")
        async def post_r2_upload_local_route(request):
            data = await request.json()
            cfg = load_r2_config()
            api_base = data.get("api_base") or cfg["r2_api_base"]
            code = data.get("code") or cfg["r2_code"]
            room_id = (data.get("room_id") or cfg["default_room_id"]).strip()
            local_path = (data.get("local_path") or "").strip()
            remote_base_dir = (data.get("remote_base_dir") or "").strip()
            preset = data.get("source_preset", "Custom Path")
            if preset != "Custom Path":
                local_path = resolve_local_dir(preset, local_path)

            res = await asyncio.to_thread(
                r2_sync_upload,
                room_id=room_id,
                local_source_path=local_path,
                remote_base_dir=remote_base_dir,
                api_base=api_base,
                code=code,
            )
            return web.json_response(res)

        @routes.get("/mockin/r2/local_browse")
        async def get_r2_local_browse_route(request):
            preset = request.query.get("preset", "ComfyUI/input")
            subpath = request.query.get("subpath", "")
            target = resolve_local_dir(preset, subpath)
            if not os.path.exists(target):
                return web.json_response({"ok": False, "path": target, "error": "Directory does not exist", "items": []})
            items = []
            if os.path.isfile(target):
                return web.json_response({
                    "ok": True,
                    "path": target,
                    "is_file": True,
                    "size": os.path.getsize(target),
                    "items": [],
                })
            try:
                for entry in os.scandir(target):
                    try:
                        stat = entry.stat()
                        items.append({
                            "name": entry.name,
                            "path": entry.path,
                            "is_dir": entry.is_dir(),
                            "size": 0 if entry.is_dir() else stat.st_size,
                            "mtime": stat.st_mtime,
                        })
                    except Exception:
                        pass
                items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
                return web.json_response({"ok": True, "path": target, "is_file": False, "items": items[:300]})
            except Exception as e:
                return web.json_response({"ok": False, "path": target, "error": str(e), "items": []})

except Exception as _route_err:
    print(f"[MockinR2] Route registration note: {_route_err}")


def extract_active_note_text(raw_text: str, room_data: dict = None) -> str:
    if isinstance(room_data, dict):
        notes_arr = room_data.get("notes")
        if isinstance(notes_arr, list) and len(notes_arr) > 0:
            for n in notes_arr:
                if isinstance(n, dict) and str(n.get("text", "")).strip():
                    return str(n.get("text", ""))
            first_n = notes_arr[0]
            if isinstance(first_n, dict):
                return str(first_n.get("text", ""))
    if not raw_text:
        return ""
    s = raw_text.strip()
    if s.startswith("{") and '"_mockin_notes"' in s:
        try:
            obj = json.loads(s)
            notes = obj.get("_mockin_notes") or []
            idx = int(obj.get("active", 0) or 0)
            if 0 <= idx < len(notes):
                return str(notes[idx].get("text", ""))
            if notes:
                return str(notes[0].get("text", ""))
        except Exception:
            pass
    return raw_text


class MockinR2CloudDriveNode:
    """Interactive Cloudflare R2 Virtual Drive Explorer, Endpoint Reviewer, Folder Copy & Sync Hub."""

    @classmethod
    def INPUT_TYPES(cls):
        cfg = load_r2_config()
        return {
            "required": {
                "api_base": ("STRING", {"default": cfg["r2_api_base"], "multiline": False}),
                "room_code": ("STRING", {"default": cfg["r2_code"], "multiline": False}),
                "room_id": ("STRING", {"default": cfg["default_room_id"], "multiline": False}),
                "action_on_run": (
                    [
                        "read_file_or_note",
                        "sync_pull_to_local",
                        "sync_push_from_local",
                        "copy_on_r2",
                        "move_on_r2",
                        "mkdir_on_r2",
                        "list_fs",
                        "manifest_folder",
                    ],
                    {"default": "read_file_or_note"},
                ),
                "remote_path": ("STRING", {"default": "", "multiline": False}),
                "dest_or_target_path": ("STRING", {"default": "", "multiline": False}),
                "target_local_preset": (LOCAL_DIR_PRESETS, {"default": "ComfyUI/input"}),
                "selected_content": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("file_content", "quick_note", "local_path", "api_response_json")
    FUNCTION = "execute"
    CATEGORY = "Mockin/R2 Cloud Drive"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def execute(
        self,
        api_base,
        room_code,
        room_id,
        action_on_run,
        remote_path,
        dest_or_target_path,
        target_local_preset,
        selected_content,
    ):
        api_base, room_code, room_id = sanitize_r2_params(api_base, room_code, room_id)
        remote_clean = (remote_path or "").strip().lstrip("/")
        if remote_clean == "read_file_or_note":
            remote_clean = ""
        dest_clean = (dest_or_target_path or "").strip()

        room_info = r2_api_request("GET", f"/{room_id}", api_base=api_base, code=room_code)
        quick_note = ""
        if room_info["ok"] and isinstance(room_info["data"], dict):
            quick_note = extract_active_note_text(room_info["data"].get("text", "") or "", room_info["data"])

        file_content = selected_content or ""
        local_out_path = resolve_local_dir(target_local_preset, dest_clean if target_local_preset == "Custom Path" else "")
        api_res = room_info["data"]

        if action_on_run == "read_file_or_note":
            if remote_clean:
                target_file = os.path.join(local_out_path, os.path.basename(remote_clean))
                try:
                    dl = r2_download_single_file(room_id, remote_clean, target_file, api_base=api_base, code=room_code)
                    local_out_path = dl["local_path"]
                    api_res = dl
                    try:
                        with open(local_out_path, "r", encoding="utf-8") as f:
                            file_content = f.read()
                    except Exception:
                        file_content = selected_content or f"[Binary File Downloaded: {local_out_path} ({dl['size']} bytes)]"
                except Exception as e:
                    api_res = {"ok": False, "error": str(e)}
            else:
                file_content = selected_content or quick_note

        elif action_on_run == "sync_pull_to_local":
            trigger_laptop_bridge_async({
                "action": "sync_pull",
                "api_base": api_base,
                "code": room_code,
                "room_id": room_id,
                "remote_path": remote_clean,
                "mode": "auto",
                "skip_existing": True,
            })
            sync_res = r2_sync_download(
                room_id=room_id,
                remote_path=remote_clean,
                local_target_dir=local_out_path,
                mode="auto",
                skip_existing=True,
                api_base=api_base,
                code=room_code,
            )
            api_res = sync_res
            local_out_path = sync_res.get("local_path", local_out_path)
            if os.path.isfile(local_out_path):
                try:
                    with open(local_out_path, "r", encoding="utf-8") as f:
                        file_content = f.read()
                except Exception:
                    file_content = json.dumps(sync_res, indent=2, ensure_ascii=False)
            else:
                file_content = json.dumps(sync_res, indent=2, ensure_ascii=False)

        elif action_on_run == "sync_push_from_local":
            src_local = resolve_local_dir(target_local_preset, dest_clean)
            up_res = r2_sync_upload(
                room_id=room_id,
                local_source_path=src_local,
                remote_base_dir=remote_clean,
                api_base=api_base,
                code=room_code,
            )
            api_res = up_res
            local_out_path = src_local
            file_content = json.dumps(up_res, indent=2, ensure_ascii=False)

        elif action_on_run == "copy_on_r2":
            cp_res = r2_api_request(
                "POST",
                f"/{room_id}/fs/copy",
                api_base=api_base,
                code=room_code,
                json_body={"src_path": remote_clean, "dest_path": dest_clean.lstrip("/"), "dest_room_id": room_id},
            )
            api_res = cp_res["data"]
            file_content = json.dumps(api_res, indent=2, ensure_ascii=False)

        elif action_on_run == "move_on_r2":
            mv_res = r2_api_request(
                "POST",
                f"/{room_id}/fs/move",
                api_base=api_base,
                code=room_code,
                json_body={"src_path": remote_clean, "dest_path": dest_clean.lstrip("/")},
            )
            api_res = mv_res["data"]
            file_content = json.dumps(api_res, indent=2, ensure_ascii=False)

        elif action_on_run == "mkdir_on_r2":
            mk_res = r2_api_request(
                "POST",
                f"/{room_id}/fs/mkdir",
                api_base=api_base,
                code=room_code,
                json_body={"path": remote_clean},
            )
            api_res = mk_res["data"]
            file_content = json.dumps(api_res, indent=2, ensure_ascii=False)

        elif action_on_run == "list_fs":
            fs_res = r2_api_request(
                "GET",
                f"/{room_id}/fs",
                api_base=api_base,
                code=room_code,
                query={"path": remote_clean, "recursive": "true"},
            )
            api_res = fs_res["data"]
            file_content = json.dumps(api_res, indent=2, ensure_ascii=False)

        elif action_on_run == "manifest_folder":
            mf_res = r2_api_request(
                "GET",
                f"/{room_id}/fs/manifest",
                api_base=api_base,
                code=room_code,
                query={"path": remote_clean},
            )
            api_res = mf_res["data"]
            file_content = json.dumps(api_res, indent=2, ensure_ascii=False)

        api_json_str = json.dumps(api_res, indent=2, ensure_ascii=False) if not isinstance(api_res, str) else api_res
        return (file_content, quick_note, local_out_path, api_json_str)


class MockinR2ReadFileNode:
    """Reads text/JSON/image files or Quick Notes directly from Mockin R2 Cloud Drive into ComfyUI."""

    @classmethod
    def INPUT_TYPES(cls):
        cfg = load_r2_config()
        return {
            "required": {
                "room_id": ("STRING", {"default": cfg["default_room_id"], "multiline": False}),
                "remote_file_path": ("STRING", {"default": "", "multiline": False}),
                "update_quick_note": ("STRING", {"default": "", "multiline": False}),
                "push_quick_note_on_run": ("BOOLEAN", {"default": False}),
                "save_to_preset": (LOCAL_DIR_PRESETS, {"default": "ComfyUI/input"}),
                "custom_local_dir": ("STRING", {"default": "", "multiline": False}),
                "api_base": ("STRING", {"default": cfg["r2_api_base"], "multiline": False}),
                "room_code": ("STRING", {"default": cfg["r2_code"], "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("file_text", "quick_note", "image", "local_file_path")
    FUNCTION = "read_r2"
    CATEGORY = "Mockin/R2 Cloud Drive"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def read_r2(
        self,
        room_id,
        remote_file_path,
        update_quick_note,
        push_quick_note_on_run,
        save_to_preset,
        custom_local_dir,
        api_base,
        room_code,
    ):
        api_base, room_code, room_id = sanitize_r2_params(api_base, room_code, room_id)
        remote_clean = (remote_file_path or "").strip().lstrip("/")

        if push_quick_note_on_run and update_quick_note.strip():
            r2_api_request(
                "POST",
                f"/{room_id}/notes",
                api_base=api_base,
                code=room_code,
                json_body={"id": "note_init", "title": "note", "text": update_quick_note},
            )

        room_res = r2_api_request("GET", f"/{room_id}", api_base=api_base, code=room_code)
        quick_note = ""
        if room_res["ok"] and isinstance(room_res["data"], dict):
            quick_note = extract_active_note_text(room_res["data"].get("text", "") or "", room_res["data"])

        file_text = ""
        local_file = ""
        img_tensor = make_empty_image_tensor()

        if remote_clean:
            target_dir = resolve_local_dir(save_to_preset, custom_local_dir)
            local_file = os.path.join(target_dir, os.path.basename(remote_clean))
            dl = r2_download_single_file(room_id, remote_clean, local_file, api_base=api_base, code=room_code)
            local_file = dl["local_path"]
            ext = os.path.splitext(local_file)[1].lower()
            if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
                img_tensor = load_image_tensor_from_path(local_file)
            try:
                with open(local_file, "r", encoding="utf-8") as f:
                    file_text = f.read()
            except Exception:
                file_text = local_file
        else:
            file_text = quick_note

        return (file_text, quick_note, img_tensor, local_file)


class MockinR2SyncDownloadNode:
    """Pulls a single file or entire folder recursively from Mockin R2 Cloud Drive to local ComfyUI folders."""

    @classmethod
    def INPUT_TYPES(cls):
        cfg = load_r2_config()
        return {
            "required": {
                "room_id": ("STRING", {"default": cfg["default_room_id"], "multiline": False}),
                "remote_path": ("STRING", {"default": "models/loras", "multiline": False}),
                "download_mode": (["auto", "folder", "file"], {"default": "auto"}),
                "target_local_preset": (LOCAL_DIR_PRESETS, {"default": "ComfyUI/models/loras"}),
                "custom_local_dir": ("STRING", {"default": "", "multiline": False}),
                "skip_existing": ("BOOLEAN", {"default": True}),
                "preserve_remote_subpath": ("BOOLEAN", {"default": False}),
                "api_base": ("STRING", {"default": cfg["r2_api_base"], "multiline": False}),
                "room_code": ("STRING", {"default": cfg["r2_code"], "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("local_path", "sync_manifest_json", "downloaded_count")
    FUNCTION = "sync_pull"
    CATEGORY = "Mockin/R2 Cloud Drive"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def sync_pull(
        self,
        room_id,
        remote_path,
        download_mode,
        target_local_preset,
        custom_local_dir,
        skip_existing,
        preserve_remote_subpath,
        api_base,
        room_code,
    ):
        api_base, room_code, room_id = sanitize_r2_params(api_base, room_code, room_id)
        trigger_laptop_bridge_async({
            "action": "sync_pull",
            "api_base": api_base,
            "code": room_code,
            "room_id": room_id,
            "remote_path": remote_path,
            "mode": download_mode,
            "skip_existing": skip_existing,
            "preserve_subpath": preserve_remote_subpath,
        })
        target_dir = resolve_local_dir(target_local_preset, custom_local_dir)
        res = r2_sync_download(
            room_id=room_id,
            remote_path=remote_path,
            local_target_dir=target_dir,
            mode=download_mode,
            skip_existing=skip_existing,
            preserve_subpath=preserve_remote_subpath,
            api_base=api_base,
            code=room_code,
        )
        return (
            res.get("local_path", target_dir),
            json.dumps(res, indent=2, ensure_ascii=False),
            int(res.get("downloaded_count", 0)),
        )


class MockinR2SyncUploadNode:
    """Pushes a local file, entire local folder, or ComfyUI IMAGE tensor directly to Mockin R2 Cloud Drive."""

    @classmethod
    def INPUT_TYPES(cls):
        cfg = load_r2_config()
        return {
            "required": {
                "room_id": ("STRING", {"default": cfg["default_room_id"], "multiline": False}),
                "source_local_preset": (LOCAL_DIR_PRESETS, {"default": "ComfyUI/output"}),
                "local_file_or_folder_path": ("STRING", {"default": "", "multiline": False}),
                "remote_base_dir": ("STRING", {"default": "comfy_uploads", "multiline": False}),
                "api_base": ("STRING", {"default": cfg["r2_api_base"], "multiline": False}),
                "room_code": ("STRING", {"default": cfg["r2_code"], "multiline": False}),
            },
            "optional": {
                "images": ("IMAGE",),
                "image_filename_prefix": ("STRING", {"default": "r2_render"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("remote_paths_text", "upload_result_json", "uploaded_count")
    FUNCTION = "sync_push"
    CATEGORY = "Mockin/R2 Cloud Drive"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def sync_push(
        self,
        room_id,
        source_local_preset,
        local_file_or_folder_path,
        remote_base_dir,
        api_base,
        room_code,
        images=None,
        image_filename_prefix="r2_render",
    ):
        api_base, room_code, room_id = sanitize_r2_params(api_base, room_code, room_id)
        if images is not None and HAS_TORCH_PIL:
            temp_dir = os.path.join(COMFY_ROOT, "temp", "mockin_r2_push", str(int(time.time() * 1000)))
            os.makedirs(temp_dir, exist_ok=True)
            for idx, img_tensor in enumerate(images):
                arr = np.clip(255.0 * img_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
                pil_img = Image.fromarray(arr)
                fname = f"{image_filename_prefix}_{idx + 1:03d}.png"
                pil_img.save(os.path.join(temp_dir, fname))
            src_path = temp_dir
        else:
            src_path = resolve_local_dir(source_local_preset, local_file_or_folder_path)

        res = r2_sync_upload(
            room_id=room_id,
            local_source_path=src_path,
            remote_base_dir=remote_base_dir,
            api_base=api_base,
            code=room_code,
        )
        paths_str = "\n".join(res.get("uploaded_paths", []))
        return (
            paths_str,
            json.dumps(res, indent=2, ensure_ascii=False),
            int(res.get("uploaded_count", 0)),
        )


class MockinR2FolderOpsNode:
    """Server-side R2 Folder & File Operations: Copy Folder/File, Move/Rename, Mkdir, Delete, List FS."""

    @classmethod
    def INPUT_TYPES(cls):
        cfg = load_r2_config()
        return {
            "required": {
                "operation": (
                    [
                        "copy_folder_or_file",
                        "move_or_rename",
                        "mkdir",
                        "delete_path",
                        "list_fs_recursive",
                        "list_rooms",
                        "create_room",
                    ],
                    {"default": "copy_folder_or_file"},
                ),
                "room_id": ("STRING", {"default": cfg["default_room_id"], "multiline": False}),
                "src_path": ("STRING", {"default": "", "multiline": False}),
                "dest_path": ("STRING", {"default": "", "multiline": False}),
                "dest_room_id": ("STRING", {"default": "", "multiline": False}),
                "api_base": ("STRING", {"default": cfg["r2_api_base"], "multiline": False}),
                "room_code": ("STRING", {"default": cfg["r2_code"], "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("response_json", "ok")
    FUNCTION = "run_op"
    CATEGORY = "Mockin/R2 Cloud Drive"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run_op(self, operation, room_id, src_path, dest_path, dest_room_id, api_base, room_code):
        api_base, room_code, room_id = sanitize_r2_params(api_base, room_code, room_id)
        s_path = (src_path or "").strip().lstrip("/")
        d_path = (dest_path or "").strip().lstrip("/")
        d_room = (dest_room_id or room_id).strip()

        if operation == "copy_folder_or_file":
            res = r2_api_request(
                "POST",
                f"/{room_id}/fs/copy",
                api_base=api_base,
                code=room_code,
                json_body={"src_path": s_path, "dest_path": d_path, "dest_room_id": d_room},
            )
        elif operation == "move_or_rename":
            res = r2_api_request(
                "POST",
                f"/{room_id}/fs/move",
                api_base=api_base,
                code=room_code,
                json_body={"src_path": s_path, "dest_path": d_path},
            )
        elif operation == "mkdir":
            res = r2_api_request(
                "POST",
                f"/{room_id}/fs/mkdir",
                api_base=api_base,
                code=room_code,
                json_body={"path": s_path or d_path},
            )
        elif operation == "delete_path":
            res = r2_api_request(
                "DELETE",
                f"/{room_id}/fs",
                api_base=api_base,
                code=room_code,
                query={"path": s_path},
            )
        elif operation == "list_fs_recursive":
            res = r2_api_request(
                "GET",
                f"/{room_id}/fs",
                api_base=api_base,
                code=room_code,
                query={"path": s_path, "recursive": "true"},
            )
        elif operation == "list_rooms":
            res = r2_api_request("GET", "/list", api_base=api_base, code=room_code)
        elif operation == "create_room":
            res = r2_api_request(
                "POST",
                "/create",
                api_base=api_base,
                code=room_code,
                json_body={"room_id": room_id, "title": d_path or f"Room {room_id}", "text": s_path},
            )
        else:
            res = {"ok": False, "data": {"error": f"Unknown operation: {operation}"}}

        return (json.dumps(res.get("data", res), indent=2, ensure_ascii=False), bool(res.get("ok")))


R2_NODE_CLASS_MAPPINGS = {
    "MockinR2CloudDriveNode": MockinR2CloudDriveNode,
    "MockinR2ReadFileNode": MockinR2ReadFileNode,
    "MockinR2SyncDownloadNode": MockinR2SyncDownloadNode,
    "MockinR2SyncUploadNode": MockinR2SyncUploadNode,
    "MockinR2FolderOpsNode": MockinR2FolderOpsNode,
}

R2_NODE_DISPLAY_NAME_MAPPINGS = {
    "MockinR2CloudDriveNode": "☁️ Mockin R2 Cloud Drive & Endpoint Explorer",
    "MockinR2ReadFileNode": "📄 Mockin R2 Read File / Note",
    "MockinR2SyncDownloadNode": "⬇️ Mockin R2 Sync / Download Folder",
    "MockinR2SyncUploadNode": "⬆️ Mockin R2 Sync / Upload Folder",
    "MockinR2FolderOpsNode": "🗂️ Mockin R2 Copy / Move / Delete Folder",
}
