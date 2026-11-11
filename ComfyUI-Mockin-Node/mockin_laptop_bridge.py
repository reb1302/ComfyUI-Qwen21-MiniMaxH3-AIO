import concurrent.futures
import http.client
import json
import mimetypes
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BRIDGE_PORT = 18189
BRIDGE_ROOM_ID = "R-LAPTOP-BRIDGE"
BRIDGE_NOTE_CMD = "laptop_cmd"
BRIDGE_NOTE_RES = "laptop_res"

DEFAULT_R2_API_BASE = "https://api.mockin.app/api/room"
DEFAULT_R2_CODE = "9999"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "laptop_project.json")
DEFAULT_LAPTOP_DIR = os.path.join(os.path.expanduser("~"), "Downloads", "Mockin_Sync")


class IPv4HTTPConnection(http.client.HTTPConnection):
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


class IPv4HTTPSConnection(http.client.HTTPSConnection):
    def connect(self):
        IPv4HTTPConnection.connect(self)
        server_hostname = self._tunnel_host if self._tunnel_host else self.host
        self.sock = self._context.wrap_socket(self.sock, server_hostname=server_hostname)


class IPv4HTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(IPv4HTTPConnection, req)


class IPv4HTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(IPv4HTTPSConnection, req, context=self._context)


_IPV4_OPENER = urllib.request.build_opener(IPv4HTTPHandler(), IPv4HTTPSHandler())


def urlopen_ipv4(req, timeout=60):
    return _IPV4_OPENER.open(req, timeout=timeout)


def load_laptop_state() -> dict:
    os.makedirs(DEFAULT_LAPTOP_DIR, exist_ok=True)
    state = {
        "laptop_path": DEFAULT_LAPTOP_DIR,
        "label": os.path.basename(DEFAULT_LAPTOP_DIR) or DEFAULT_LAPTOP_DIR,
    }
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                p = str(saved.get("laptop_path") or "").strip()
                if p:
                    os.makedirs(p, exist_ok=True)
                    state["laptop_path"] = os.path.abspath(p)
                    state["label"] = os.path.basename(state["laptop_path"].rstrip("\\/")) or state["laptop_path"]
        except Exception:
            pass
    return state


def save_laptop_state(laptop_path: str) -> dict:
    clean_p = os.path.abspath(str(laptop_path or DEFAULT_LAPTOP_DIR).strip())
    os.makedirs(clean_p, exist_ok=True)
    label = os.path.basename(clean_p.rstrip("\\/")) or clean_p
    state = {"laptop_path": clean_p, "label": label, "hostname": socket.gethostname()}
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    threading.Thread(
        target=lambda: r2_request(
            "POST",
            f"/{BRIDGE_ROOM_ID}/text",
            json_body={"text": json.dumps({"ok": True, **state}, ensure_ascii=False)},
        ),
        daemon=True,
    ).start()
    return state


def pick_folder_windows(initial_dir: str = "") -> str:
    ps_script = r"""
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

[ComImport, Guid("DC1C5A9C-E88A-4dde-A5A1-60F82A20AEF7")]
internal class FileOpenDialogRCW { }

[ComImport, Guid("42f85136-db7e-439c-85f1-e4075d135fc8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
internal interface IFileOpenDialog {
    [PreserveSig] int Show(IntPtr parent);
    void SetFileTypes(uint cFileTypes, IntPtr rgFilterSpec);
    void SetFileTypeIndex(uint iFileType);
    void GetFileTypeIndex(out uint piFileType);
    void Advise(IntPtr pfde, out uint pdwCookie);
    void Unadvise(uint dwCookie);
    void SetOptions(uint fos);
    void GetOptions(out uint pfos);
    void SetDefaultFolder(IShellItem psi);
    void SetFolder(IShellItem psi);
    void GetFolder(out IShellItem ppsi);
    void GetCurrentSelection(out IShellItem ppsi);
    void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string pszName);
    void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string pszName);
    void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string pszTitle);
    void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string pszText);
    void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string pszLabel);
    void GetResult(out IShellItem ppsi);
}

[ComImport, Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
internal interface IShellItem {
    void BindToHandler(IntPtr pbc, [MarshalAs(UnmanagedType.LPStruct)] Guid bhid, [MarshalAs(UnmanagedType.LPStruct)] Guid riid, out IntPtr ppv);
    void GetParent(out IShellItem ppsi);
    void GetDisplayName(uint sigdnName, [MarshalAs(UnmanagedType.LPWStr)] out string ppszName);
    void GetAttributes(uint sfgaoMask, out uint psfgaoAttribs);
    void Compare(IShellItem psi, uint hint, out int piOrder);
}

public class ModernFolderPicker {
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = false)]
    public static extern void SHCreateItemFromParsingName(
        [MarshalAs(UnmanagedType.LPWStr)] string pszPath,
        IntPtr pbc,
        [MarshalAs(UnmanagedType.LPStruct)] Guid riid,
        out IShellItem ppv
    );

    public static string Pick(string title, string initDir) {
        var dlg = (IFileOpenDialog)new FileOpenDialogRCW();
        uint opts;
        dlg.GetOptions(out opts);
        dlg.SetOptions(opts | 0x20 | 0x40);
        if (!string.IsNullOrEmpty(title)) dlg.SetTitle(title);
        if (!string.IsNullOrEmpty(initDir) && System.IO.Directory.Exists(initDir)) {
            try {
                IShellItem initItem;
                SHCreateItemFromParsingName(initDir, IntPtr.Zero, new Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE"), out initItem);
                dlg.SetFolder(initItem);
            } catch { }
        }
        int hr = dlg.Show(GetForegroundWindow());
        if (hr != 0) return "";
        IShellItem item;
        dlg.GetResult(out item);
        string path;
        item.GetDisplayName(0x80058000, out path);
        return path ?? "";
    }
}
"@
$init = $env:MOCKIN_INIT_DIR
$res = [ModernFolderPicker]::Pick("Mockin R2 - Chọn thư mục đồng bộ trên Laptop (Máy con)", $init)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Write-Output $res
"""
    env = os.environ.copy()
    env["MOCKIN_INIT_DIR"] = initial_dir or DEFAULT_LAPTOP_DIR
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-STA", "-Command", "-"],
            input=ps_script,
            text=True,
            encoding="utf-8",
            capture_output=True,
            env=env,
            timeout=300,
        )
        lines = [ln.strip() for ln in (res.stdout or "").splitlines() if ln.strip()]
        chosen = lines[-1] if lines else ""
        if chosen and os.path.isdir(chosen):
            return os.path.abspath(chosen)
    except Exception:
        pass
    return ""


def r2_request(method: str, endpoint_path: str, api_base: str = DEFAULT_R2_API_BASE, code: str = DEFAULT_R2_CODE, query: dict = None, json_body: dict = None, timeout: int = 30) -> dict:
    base = (api_base or DEFAULT_R2_API_BASE).strip().rstrip("/")
    ep = endpoint_path if endpoint_path.startswith("/") else "/" + endpoint_path
    url = base + ep
    if query:
        clean_q = {k: v for k, v in query.items() if v is not None and str(v) != ""}
        if clean_q:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(clean_q)
    headers = {
        "X-Room-Code": str(code or DEFAULT_R2_CODE).strip(),
        "User-Agent": "Mockin-Laptop-Bridge/2.0",
    }
    data_bytes = None
    if json_body is not None:
        data_bytes = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    try:
        req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method.upper())
        with urlopen_ipv4(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = raw
            return {"ok": True, "status": resp.status, "data": parsed}
    except urllib.error.HTTPError as e:
        err_raw = e.read().decode("utf-8", errors="replace")
        return {"ok": False, "status": e.code, "error": err_raw}
    except Exception as e:
        return {"ok": False, "status": 0, "error": str(e)}


def download_one_file(room_id: str, remote_path: str, local_dest: str, api_base: str, code: str, direct_url: str = "") -> dict:
    os.makedirs(os.path.dirname(os.path.abspath(local_dest)), exist_ok=True)
    t0 = time.time()
    if direct_url:
        req = urllib.request.Request(direct_url, headers={"User-Agent": "Mockin-Laptop-Bridge/2.0"}, method="GET")
    else:
        dl_url = f"{api_base.rstrip('/')}/{room_id}/fs/download?path={urllib.parse.quote(remote_path.lstrip('/'))}"
        req = urllib.request.Request(
            dl_url,
            headers={"X-Room-Code": str(code or DEFAULT_R2_CODE).strip(), "User-Agent": "Mockin-Laptop-Bridge/2.0"},
            method="GET",
        )
    part_path = local_dest + ".part"
    with urlopen_ipv4(req, timeout=300) as resp, open(part_path, "wb") as out_f:
        shutil.copyfileobj(resp, out_f, length=1024 * 1024)
    os.replace(part_path, local_dest)
    return {
        "ok": True,
        "remote_path": remote_path,
        "local_path": os.path.abspath(local_dest),
        "size": os.path.getsize(local_dest),
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
    }


def sync_pull_to_laptop(
    room_id: str,
    remote_path: str,
    laptop_target_dir: str,
    mode: str = "auto",
    skip_existing: bool = True,
    preserve_subpath: bool = False,
    api_base: str = DEFAULT_R2_API_BASE,
    code: str = DEFAULT_R2_CODE,
) -> dict:
    room_id = (room_id or "R-6ZZP").strip()
    remote_clean = (remote_path or "").strip().strip("/")
    os.makedirs(laptop_target_dir, exist_ok=True)

    is_folder = mode == "folder"
    manifest_files = []
    if mode in ("auto", "folder"):
        mf = r2_request("GET", f"/{room_id}/fs/manifest", api_base=api_base, code=code, query={"path": remote_clean})
        if mf["ok"] and isinstance(mf["data"], dict):
            manifest_files = mf["data"].get("files", [])
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
        dest_file = os.path.join(laptop_target_dir, rel_dest.replace("/", os.sep))
        expected_size = int(manifest_files[0].get("size", 0) or 0) if manifest_files else 0
        direct_url = manifest_files[0].get("direct_download_url", "") if manifest_files else ""
        if skip_existing and os.path.exists(dest_file):
            local_sz = os.path.getsize(dest_file)
            if (expected_size > 0 and local_sz == expected_size) or (expected_size == 0 and local_sz > 0):
                skipped.append({"remote_path": remote_clean, "local_path": dest_file, "size": local_sz})
                return {
                    "ok": True,
                    "mode": "file",
                    "room_id": room_id,
                    "remote_path": remote_clean,
                    "local_path": dest_file,
                    "downloaded_count": 0,
                    "skipped_count": 1,
                    "downloaded": [],
                    "skipped": skipped,
                    "errors": [],
                }
        try:
            info = download_one_file(room_id, remote_clean, dest_file, api_base, code, direct_url)
            downloaded.append(info)
        except Exception as e:
            errors.append({"remote_path": remote_clean, "error": str(e)})
        return {
            "ok": len(errors) == 0,
            "mode": "file",
            "room_id": room_id,
            "remote_path": remote_clean,
            "local_path": dest_file,
            "downloaded_count": len(downloaded),
            "skipped_count": len(skipped),
            "downloaded": downloaded,
            "skipped": skipped,
            "errors": errors,
        }

    # Folder sync 1:1
    to_download = []
    for f_item in manifest_files:
        f_remote = f_item.get("path", "")
        f_rel = f_item.get("relative_path") or f_item.get("name") or os.path.basename(f_remote)
        if preserve_subpath and f_remote:
            f_rel = f_remote
        elif remote_clean and not preserve_subpath:
            # Keep folder name if downloading a specific subfolder so files stay organized
            folder_name = os.path.basename(remote_clean.rstrip("/"))
            if folder_name and os.path.basename(laptop_target_dir.rstrip("\\/")).lower() != folder_name.lower():
                f_rel = f"{folder_name}/{f_rel.lstrip('/')}"
        dest_file = os.path.join(laptop_target_dir, f_rel.replace("/", os.sep))
        expected_size = int(f_item.get("size", 0) or 0)
        if skip_existing and os.path.exists(dest_file):
            local_sz = os.path.getsize(dest_file)
            if (expected_size > 0 and local_sz == expected_size) or (expected_size == 0 and local_sz > 0):
                skipped.append({"remote_path": f_remote, "local_path": dest_file, "size": local_sz})
                continue
        to_download.append((f_remote, dest_file, f_item.get("direct_download_url", "")))

    if to_download:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            future_map = {
                pool.submit(download_one_file, room_id, f_rem, dst_f, api_base, code, d_url): f_rem
                for (f_rem, dst_f, d_url) in to_download
            }
            for fut in concurrent.futures.as_completed(future_map):
                f_rem = future_map[fut]
                try:
                    downloaded.append(fut.result())
                except Exception as e:
                    errors.append({"remote_path": f_rem, "error": str(e)})

    return {
        "ok": len(errors) == 0,
        "mode": "folder",
        "room_id": room_id,
        "remote_path": remote_clean,
        "local_path": laptop_target_dir,
        "manifest_total": len(manifest_files),
        "downloaded_count": len(downloaded),
        "skipped_count": len(skipped),
        "downloaded": downloaded,
        "skipped": skipped,
        "errors": errors,
    }


def sync_push_from_laptop(
    room_id: str,
    laptop_source_path: str,
    remote_base_dir: str = "",
    api_base: str = DEFAULT_R2_API_BASE,
    code: str = DEFAULT_R2_CODE,
) -> dict:
    src = os.path.abspath((laptop_source_path or "").strip())
    base_dir = (remote_base_dir or "").strip().strip("/")
    if not os.path.exists(src):
        return {"ok": False, "error": f"Laptop path does not exist: {src}", "uploaded_count": 0}

    local_files = []
    if os.path.isfile(src):
        ctype = mimetypes.guess_type(src)[0] or "application/octet-stream"
        local_files.append({"abs_path": src, "rel_path": os.path.basename(src), "size": os.path.getsize(src), "content_type": ctype})
    else:
        for root, _, files in os.walk(src):
            for fn in files:
                if fn.endswith(".part"):
                    continue
                full_p = os.path.join(root, fn)
                rel_p = os.path.relpath(full_p, src).replace("\\", "/")
                ctype = mimetypes.guess_type(full_p)[0] or "application/octet-stream"
                local_files.append({"abs_path": full_p, "rel_path": rel_p, "size": os.path.getsize(full_p), "content_type": ctype})

    if not local_files:
        return {"ok": False, "error": f"No files in Laptop folder: {src}", "uploaded_count": 0}

    meta_list = [{"path": it["rel_path"], "size": it["size"], "content_type": it["content_type"]} for it in local_files]
    presign_res = r2_request("POST", f"/{room_id}/files/presign", api_base=api_base, code=code, json_body={"base_dir": base_dir, "files": meta_list})
    if not presign_res["ok"] or not isinstance(presign_res["data"], dict):
        return {"ok": False, "error": f"Presign failed: {presign_res.get('error')}", "uploaded_count": 0}

    uploads = presign_res["data"].get("uploads", [])
    uploaded_items = []
    errors = []

    for idx, f_meta in enumerate(local_files):
        if idx >= len(uploads):
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
                headers={"Content-Type": ctype, "Content-Length": str(len(file_bytes)), "User-Agent": "Mockin-Laptop-Bridge/2.0"},
                method="PUT",
            )
            with urlopen_ipv4(put_req, timeout=600) as _:
                pass
            uploaded_items.append(u_info)
        except Exception as e:
            errors.append({"file": f_meta["rel_path"], "error": str(e)})

    if uploaded_items:
        r2_request("POST", f"/{room_id}/files/confirm", api_base=api_base, code=code, json_body={"base_dir": "", "files": uploaded_items})

    return {
        "ok": len(errors) == 0 and len(uploaded_items) > 0,
        "room_id": room_id,
        "remote_base_dir": base_dir,
        "uploaded_count": len(uploaded_items),
        "uploaded_paths": [u.get("path", "") for u in uploaded_items],
        "errors": errors,
    }


def execute_bridge_command(payload: dict) -> dict:
    action = str(payload.get("action") or "get_status").strip()
    state = load_laptop_state()

    if action == "get_status":
        return {"ok": True, **state, "hostname": socket.gethostname()}

    if action == "pick_folder":
        chosen = pick_folder_windows(state["laptop_path"])
        if not chosen:
            return {"ok": False, "cancelled": True, **state}
        new_state = save_laptop_state(chosen)
        return {"ok": True, **new_state}

    if action == "set_folder":
        target = str(payload.get("laptop_path") or "").strip()
        if not target:
            return {"ok": False, "error": "Empty path", **state}
        new_state = save_laptop_state(target)
        return {"ok": True, **new_state}

    if action == "open_explorer":
        target = state["laptop_path"]
        if os.path.isdir(target):
            subprocess.Popen(["explorer.exe", target])
        return {"ok": True, **state}

    if action == "sync_pull":
        api_base = payload.get("api_base") or DEFAULT_R2_API_BASE
        code = payload.get("code") or DEFAULT_R2_CODE
        room_id = payload.get("room_id") or "R-6ZZP"
        remote_path = payload.get("remote_path") or ""
        mode = payload.get("mode") or "auto"
        skip_existing = bool(payload.get("skip_existing", True))
        preserve_subpath = bool(payload.get("preserve_subpath", False))
        custom_dir = str(payload.get("laptop_path") or "").strip()
        target_dir = save_laptop_state(custom_dir)["laptop_path"] if custom_dir else state["laptop_path"]
        res = sync_pull_to_laptop(
            room_id=room_id,
            remote_path=remote_path,
            laptop_target_dir=target_dir,
            mode=mode,
            skip_existing=skip_existing,
            preserve_subpath=preserve_subpath,
            api_base=api_base,
            code=code,
        )
        return {**res, "laptop_path": target_dir, "label": os.path.basename(target_dir.rstrip("\\/")) or target_dir}

    if action == "sync_push":
        api_base = payload.get("api_base") or DEFAULT_R2_API_BASE
        code = payload.get("code") or DEFAULT_R2_CODE
        room_id = payload.get("room_id") or "R-6ZZP"
        remote_base_dir = payload.get("remote_base_dir") or ""
        custom_dir = str(payload.get("laptop_path") or "").strip()
        source_dir = custom_dir if custom_dir else state["laptop_path"]
        res = sync_push_from_laptop(
            room_id=room_id,
            laptop_source_path=source_dir,
            remote_base_dir=remote_base_dir,
            api_base=api_base,
            code=code,
        )
        return {**res, "laptop_path": source_dir}

    return {"ok": False, "error": f"Unknown action: {action}", **state}


class LaptopBridgeHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Cache-Control", "no-store")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = dict(urllib.parse.parse_qsl(parsed.query))
        action = parsed.path.strip("/") or qs.get("action") or "get_status"
        payload = {**qs, "action": action}
        res = execute_bridge_command(payload)
        body = json.dumps(res, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length).decode("utf-8", errors="replace") if length > 0 else "{}"
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {}
        path_action = self.path.strip("/")
        if path_action and "action" not in payload:
            payload["action"] = path_action
        res = execute_bridge_command(payload)
        body = json.dumps(res, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def r2_relay_poller_loop():
    """Polls R-LAPTOP-BRIDGE on R2 so even when ComfyUI is opened via an insecure HTTP tunnel
    where Chrome blocks 127.0.0.1 Private Network Access, commands from H200 UI still execute on Laptop."""
    last_cmd_id = ""
    last_hb = 0.0
    init_state = load_laptop_state()
    r2_request("POST", "/create", json_body={"room_id": BRIDGE_ROOM_ID, "title": "Laptop Bridge", "text": json.dumps({"ok": True, "ts": time.time(), **init_state})})
    while True:
        try:
            now = time.time()
            if now - last_hb > 20.0:
                last_hb = now
                cur_state = load_laptop_state()
                r2_request("POST", f"/{BRIDGE_ROOM_ID}/text", json_body={"text": json.dumps({"ok": True, "ts": now, **cur_state}, ensure_ascii=False)}, timeout=8)
            res = r2_request("GET", f"/{BRIDGE_ROOM_ID}", timeout=10)
            if res.get("ok") and isinstance(res.get("data"), dict):
                notes = res["data"].get("notes") or []
                cmd_note = next((n for n in notes if isinstance(n, dict) and n.get("id") == BRIDGE_NOTE_CMD), None)
                if cmd_note:
                    raw_txt = str(cmd_note.get("text") or "").strip()
                    if raw_txt.startswith("{"):
                        cmd_obj = json.loads(raw_txt)
                        cmd_id = str(cmd_obj.get("cmd_id") or "")
                        if cmd_id and cmd_id != last_cmd_id:
                            last_cmd_id = cmd_id
                            result = execute_bridge_command(cmd_obj)
                            result["cmd_id"] = cmd_id
                            result["ts"] = time.time()
                            result["completed_at"] = time.time()
                            r2_request(
                                "POST",
                                f"/{BRIDGE_ROOM_ID}/notes",
                                json_body={
                                    "id": BRIDGE_NOTE_RES,
                                    "title": "laptop_res",
                                    "text": json.dumps(result, ensure_ascii=False),
                                },
                            )
        except Exception:
            pass
        time.sleep(0.6)


def main():
    # Single-instance check on port 18189
    try:
        server = ThreadingHTTPServer(("127.0.0.1", BRIDGE_PORT), LaptopBridgeHTTPHandler)
    except OSError:
        # Already running
        return
    t = threading.Thread(target=r2_relay_poller_loop, daemon=True)
    t.start()
    print(f"[Mockin Laptop Bridge] Running on http://127.0.0.1:{BRIDGE_PORT} -> {load_laptop_state()['laptop_path']}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
