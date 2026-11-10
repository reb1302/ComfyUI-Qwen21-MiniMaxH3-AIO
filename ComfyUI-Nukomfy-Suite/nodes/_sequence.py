"""fileseq wrappers for Nuke-style frame patterns.

Three pattern flavors are recognized:

    ####.exr     - Nuke literal hash padding
    %04d.exr     - printf padding
    @@@@.exr     - fileseq PAD_STYLE_HASH1 padding (one @ per digit)

fileseq parses the @ style natively; the other two are normalized to @
on the fly. Falls back to a pure-Python sequence scanner if fileseq is
not installed.
"""
from __future__ import annotations

import os
import re

try:
    import fileseq
except ImportError:
    fileseq = None  # type: ignore[assignment]


_HASH_RE = re.compile(r"#+")
_PRINTF_RE = re.compile(r"%0(\d+)d")
_AT_RE = re.compile(r"@+")
_TRAIL_DIGITS_RE = re.compile(r"^(.*?)(\d+)(\.[^.]+)$")


class _FallbackFileSequence:
    """Minimal pure-Python replacement for fileseq.FileSequence."""

    def __init__(self, directory: str, stem: str, pad_width: int, ext: str, frames: list[int]):
        self._dir = directory
        self._stem = stem
        self._pad = max(1, int(pad_width))
        self._ext = ext
        self._frames = sorted(set(int(f) for f in frames))

    def frameSet(self) -> list[int]:
        return list(self._frames)

    def frame(self, f: int) -> str:
        num = f"{abs(int(f)):0{self._pad}d}"
        if int(f) < 0:
            num = f"-{num}"
        return os.path.join(self._dir, f"{self._stem}{num}{self._ext}")

    def basename(self) -> str:
        return self._stem

    def padding(self) -> str:
        return "#" * self._pad

    def extension(self) -> str:
        return self._ext


def has_padding_token(path: str) -> bool:
    """Return True if the path basename contains ####, %0Nd or @@@@."""
    base = os.path.basename(path)
    return bool(_HASH_RE.search(base) or _PRINTF_RE.search(base) or _AT_RE.search(base))


def classify_path(file_path: str) -> tuple[str, str]:
    """Classify a user-provided file_path.

    Returns a (kind, normalized_path) tuple where kind is one of:
        'pattern'        - basename contains a padding token
        'literal_file'   - existing file on disk, no padding
        'literal_folder' - existing directory on disk
        'invalid'        - empty path or non-existing target without padding
    """
    if not file_path:
        return ("invalid", "")
    norm = os.path.normpath(file_path)
    if has_padding_token(norm):
        return ("pattern", norm)
    if os.path.isdir(norm):
        return ("literal_folder", norm)
    if os.path.isfile(norm):
        return ("literal_file", norm)
    return ("invalid", norm)


def nuke_pattern_to_fileseq(pattern: str) -> str:
    """Normalize ####.exr or %04d.exr into @@@@.exr (PAD_STYLE_HASH1)."""
    m = _HASH_RE.search(pattern)
    if m:
        pad = len(m.group(0))
        return _HASH_RE.sub("@" * pad, pattern, count=1)
    m = _PRINTF_RE.search(pattern)
    if m:
        pad = int(m.group(1))
        return _PRINTF_RE.sub("@" * pad, pattern, count=1)
    return pattern


def detect_padding(path: str) -> int:
    """Return the padding width (digit count) of the first padding token
    in the basename, or 0 if no token is present."""
    base = os.path.basename(path)
    m = _HASH_RE.search(base)
    if m:
        return len(m.group(0))
    m = _PRINTF_RE.search(base)
    if m:
        return int(m.group(1))
    m = _AT_RE.search(base)
    if m:
        return len(m.group(0))
    return 0


def path_for_frame(template: str, frame: int, padding: "int | None" = None) -> str:
    """Substitute a padding token in template with the numeric frame."""
    if padding is None:
        padding = detect_padding(template)
    if padding <= 0:
        return template
    frame_str = f"{abs(frame):0{padding}d}"
    if frame < 0:
        frame_str = f"-{frame_str}"
    out, count = _HASH_RE.subn(frame_str, template, count=1)
    if count:
        return out
    out, count = _PRINTF_RE.subn(frame_str, template, count=1)
    if count:
        return out
    out, count = _AT_RE.subn(frame_str, template, count=1)
    if count:
        return out
    return template


def expand_custom_range(first: int, last: int) -> list[int]:
    """Return the inclusive list of frames between first and last."""
    if first == last:
        return [first]
    if first < last:
        return list(range(first, last + 1))
    return list(range(first, last - 1, -1))


def missing_frame_range(expected: "list[int]", present: "list[int]") -> str:
    """Return a compact range string of the frames missing from ``present``."""
    missing = sorted(set(expected) - set(present))
    if not missing:
        return ""
    if fileseq is not None:
        return fileseq.framesToFrameRange(missing)
    return ",".join(str(f) for f in missing)


def _fallback_find_sequences_in_folder(directory: str) -> list[_FallbackFileSequence]:
    groups: dict[tuple[str, int, str], list[int]] = {}
    try:
        entries = os.listdir(directory)
    except OSError:
        return []
    for name in entries:
        full = os.path.join(directory, name)
        if not os.path.isfile(full):
            continue
        m = _TRAIL_DIGITS_RE.match(name)
        if not m:
            continue
        stem, digits, ext = m.group(1), m.group(2), m.group(3)
        groups.setdefault((stem, len(digits), ext), []).append(int(digits))
    return [
        _FallbackFileSequence(directory, stem, pad, ext, frames)
        for (stem, pad, ext), frames in sorted(groups.items())
    ]


def find_sequences_in_folder(directory: str):
    """Return all FileSequences detected on disk inside ``directory``."""
    if not os.path.isdir(directory):
        return []
    if fileseq is not None:
        try:
            return list(fileseq.findSequencesOnDisk(directory))
        except Exception:
            pass
    return _fallback_find_sequences_in_folder(directory)


def find_sequence_on_disk(template_path: str):
    """Locate a sequence on disk matching the given padded template path."""
    directory = os.path.dirname(template_path) or "."
    basename = os.path.basename(template_path)
    if not os.path.isdir(directory):
        return None
    fs_basename = nuke_pattern_to_fileseq(basename)
    target_stem = _strip_padding_token(fs_basename).rstrip("._-")
    target_ext = os.path.splitext(fs_basename)[1]
    seqs = find_sequences_in_folder(directory)
    for seq in seqs:
        seq_stem = os.path.splitext(seq.basename())[0].rstrip("._-")
        if seq_stem == target_stem and seq.extension() == target_ext:
            return seq
    return None


def _strip_padding_token(name: str) -> str:
    name = _AT_RE.sub("", name)
    name = _PRINTF_RE.sub("", name)
    name = _HASH_RE.sub("", name)
    return os.path.splitext(name)[0]
