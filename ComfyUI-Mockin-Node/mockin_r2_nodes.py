import base64
import json
import mimetypes
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
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
                if r and r not in INVALID_ROOM_IDS and not r.startswith("http") and "/" not in r:
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
        if k == "default_room_id" and (val in INVALID_ROOM_IDS or val.startswith("http") or "/" in val):
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
}


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
        # Check if code accidentally held the room_id (e.g. "R-6ZZP") from a shifted widget array
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
        # Sanitize first path segment if it was a shifted action name like /read_file_or_note
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
    timeout: int = 60,
) -> dict:
    api_base, code, _ = sanitize_r2_params(api_base, code, "")
    url = build_r2_url(api_base, endpoint_path, query)
    headers = {
        "X-Room-Code": code,
        "User-Agent": "ComfyUI-Mockin-R2/1.0",
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
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed_ms = round((time.time() - t0) * 1000, 1)
            raw = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            text = raw.decode("utf-8", errors="replace")
            try:
                parsed = json.loads(text)
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
        req = urllib.request.Request(direct_url, headers={"User-Agent": "ComfyUI-Mockin-R2/1.0"}, method="GET")
    else:
        dl_url = build_r2_url(api_base, f"/{room_id}/fs/download", {"path": remote_path.lstrip("/")})
        req = urllib.request.Request(
            dl_url,
            headers={
                "X-Room-Code": str(code or DEFAULT_R2_CODE).strip(),
                "User-Agent": "ComfyUI-Mockin-R2/1.0",
            },
            method="GET",
        )
    with urllib.request.urlopen(req, timeout=300) as resp, open(local_dest_path, "wb") as out_f:
        shutil.copyfileobj(resp, out_f, length=1024 * 1024)
    size_bytes = os.path.getsize(local_dest_path)
    return {
        "ok": True,
        "remote_path": remote_path,
        "local_path": os.path.abspath(local_dest_path),
        "size": size_bytes,
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
    }


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
    if mode == "auto":
        if not remote_clean:
            is_folder = True
        else:
            manifest_check = r2_api_request(
                "GET",
                f"/{room_id}/fs/manifest",
                api_base=api_base,
                code=code,
                query={"path": remote_clean},
            )
            if manifest_check["ok"] and isinstance(manifest_check["data"], dict):
                files_list = manifest_check["data"].get("files", [])
                if len(files_list) > 1 or (len(files_list) == 1 and files_list[0].get("path") != remote_clean):
                    is_folder = True
                elif len(files_list) == 1 and files_list[0].get("path") == remote_clean:
                    is_folder = False

    downloaded = []
    skipped = []
    errors = []

    if not is_folder and remote_clean:
        rel_dest = remote_clean if preserve_subpath else os.path.basename(remote_clean)
        dest_file = os.path.join(local_target_dir, rel_dest.replace("/", os.sep))
        if skip_existing and os.path.exists(dest_file) and os.path.getsize(dest_file) > 0:
            skipped.append({"remote_path": remote_clean, "local_path": dest_file, "size": os.path.getsize(dest_file)})
        else:
            try:
                info = r2_download_single_file(room_id, remote_clean, dest_file, api_base=api_base, code=code)
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
        try:
            info = r2_download_single_file(
                room_id,
                f_remote,
                dest_file,
                api_base=api_base,
                code=code,
                direct_url=f_item.get("direct_download_url", ""),
            )
            downloaded.append(info)
        except Exception as e:
            errors.append({"remote_path": f_remote, "error": str(e)})

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
                    "User-Agent": "ComfyUI-Mockin-R2/1.0",
                },
                method="PUT",
            )
            with urllib.request.urlopen(put_req, timeout=600) as _:
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
            res = r2_api_request(
                method=method,
                endpoint_path=endpoint_path,
                api_base=api_base,
                code=code,
                query=query,
                json_body=json_body,
            )
            res.pop("raw_bytes", None)
            return web.json_response(res)

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
                dl_info = r2_download_single_file(room_id, remote_path, local_file, api_base=api_base, code=code)
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
            res = r2_sync_download(
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

            res = r2_sync_upload(
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
