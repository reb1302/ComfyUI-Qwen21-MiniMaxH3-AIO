"""fileseq wrappers for Nuke-style frame patterns.

Three pattern flavors are recognized:

    ####.exr     - Nuke literal hash padding
    %04d.exr     - printf padding
    @@@@.exr     - fileseq PAD_STYLE_HASH1 padding (one @ per digit)

fileseq parses the @ style natively; the other two are normalized to @
on the fly.
"""
from __future__ import annotations

import os
import re

import fileseq


_HASH_RE = re.compile(r"#+")
_PRINTF_RE = re.compile(r"%0(\d+)d")
_AT_RE = re.compile(r"@+")


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
    """Substitute a padding token in template with the numeric frame.

    The padding width is taken from the token itself; pass ``padding``
    explicitly to override (useful when the template has no token and
    the caller wants to insert one).
    """
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
    """Return the inclusive list of frames between first and last.

    When ``first > last`` the returned list is reversed for reverse
    playback. Single-frame ranges (first == last) return [first].
    """
    if first == last:
        return [first]
    if first < last:
        return list(range(first, last + 1))
    return list(range(first, last - 1, -1))


def missing_frame_range(expected: "list[int]", present: "list[int]") -> str:
    """Return a compact range string of the frames missing from ``present``.

    Uses fileseq.framesToFrameRange to produce e.g. "3,7-9,12". Returns
    an empty string when nothing is missing.
    """
    missing = sorted(set(expected) - set(present))
    if not missing:
        return ""
    return fileseq.framesToFrameRange(missing)


def find_sequences_in_folder(directory: str) -> "list[fileseq.FileSequence]":
    """Return all FileSequences detected on disk inside ``directory``."""
    if not os.path.isdir(directory):
        return []
    try:
        return list(fileseq.findSequencesOnDisk(directory))
    except fileseq.FileSeqException:
        return []


def find_sequence_on_disk(template_path: str) -> "fileseq.FileSequence | None":
    """Locate a sequence on disk matching the given padded template path.

    The basename must contain a padding token (####, @@@@ or %0Nd). Returns
    the first FileSequence whose stem and extension match the template, or
    None when no candidate is found.
    """
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
