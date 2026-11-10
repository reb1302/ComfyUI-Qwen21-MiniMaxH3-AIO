"""Restart the running ComfyUI process."""
from __future__ import annotations

import os
import sys


def restart_comfyui() -> None:
    """Replace the current ComfyUI process with a fresh instance.

    On comfy-cli sessions (e.g. Comfy Desktop, comfy-cli launch), the
    daemon parent watches a sentinel file at ``${__COMFY_CLI_SESSION__}.reboot``
    and re-spawns the process when it appears, so the function writes the
    sentinel and exits cleanly.

    Everywhere else, ``os.execv`` is used to replace the running process
    in-place (POSIX) or to spawn the replacement under the same shell
    parent (Windows portable launchers, native installs), so the new
    process inherits the same console as the original startup. The HTTP
    response must be flushed before this is called; the route schedules
    it via asyncio.create_task with a short delay.

    The Windows portable build flips ``--auto-launch`` on by default,
    which would pop a new browser tab on every admin reboot. Inject
    ``--disable-auto-launch`` so the existing tab keeps the session.
    Duplicates are idempotent: ComfyUI's parser short-circuits on the
    flag when it is already present.
    """
    cli_session = os.environ.get("__COMFY_CLI_SESSION__")
    if cli_session:
        open(cli_session + ".reboot", "w").close()
        sys.exit(0)
    args = list(sys.argv)
    if "--disable-auto-launch" not in args:
        args.append("--disable-auto-launch")
    os.execv(sys.executable, [sys.executable] + args)
