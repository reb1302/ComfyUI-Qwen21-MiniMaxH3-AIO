"""Turn the page source into the self-contained H200 page (no outside fonts). With a helper path, also embed it there."""
import re
import sys

src_path, page_path = sys.argv[1], sys.argv[2]
helper_paths = sys.argv[3:]

source = open(src_path, encoding="utf-8").read()
body = re.sub(r'^<link [^>]*fonts\.(googleapis|gstatic)\.com[^>]*>\n', "", source, flags=re.M)
body = re.sub(r'^<meta name="claude:viewport-fit-sides"[^>]*>\n', "", body, flags=re.M)
assert "googleapis" not in body and "gstatic" not in body, "outside font left in the page"
page = (
    '<!doctype html>\n<html lang="vi">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    "<style>img{max-width:100%}</style>\n</head>\n<body>\n" + body + "</body>\n</html>\n"
)
assert '"""' not in page, "triple quote would end the Python string"
assert re.search(r"https?://", re.sub(r"http://www\.w3\.org/\S+", "", page)) is None, "page must not name an outside address"
with open(page_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(page)
print("page bytes", len(page.encode("utf-8")))

BEGIN = "# >>> CONCEPT_IMAGE_EDIT_HTML: sinh từ vstone/concept_image_edit_src.html bằng vstone/embed_concept_html.py, đừng sửa tay\n"
END = "# <<< CONCEPT_IMAGE_EDIT_HTML\n"
INSTALLER = '''

def _install_concept_image_edit_html() -> None:
    """Trang sửa ảnh cho nhóm concept, mở tại /extensions/vstone/concept_image_edit.html, cùng cơ chế với JS của node Huy."""
    try:
        import server
        web_root = server.PromptServer.instance.web_root
    except Exception:
        return  # không chạy trong ComfyUI
    path = os.path.join(web_root, "extensions", "vstone", "concept_image_edit.html")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\\n") as html_file:
            html_file.write(CONCEPT_IMAGE_EDIT_HTML)
    except OSError as e:
        print(f"[VStone] Không ghi được trang concept vào {path}: {e}", flush=True)


_install_concept_image_edit_html()
'''
block = BEGIN + 'CONCEPT_IMAGE_EDIT_HTML = r"""' + page + '"""\n' + END
for helper_path in helper_paths:
    helper = open(helper_path, encoding="utf-8", newline="").read()
    if BEGIN in helper:
        start = helper.index(BEGIN)
        stop = helper.index(END) + len(END)
        helper = helper[:start] + block + helper[stop:]
    else:
        anchor = "_install_screenx_editor_js()\n"
        at = helper.rindex(anchor) + len(anchor)
        helper = helper[:at] + "\n\n" + block + INSTALLER + helper[at:]
    with open(helper_path, "w", encoding="utf-8", newline="") as f:
        f.write(helper)
    print("embedded into", helper_path, len(helper.encode("utf-8")))
