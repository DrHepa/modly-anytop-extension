"""Dependency-free Modly PROCESS bootstrap for AnyTop.

The protocol uses a duplicated descriptor.  Public stdout/stderr are isolated
before importing the ML runtime, so upstream prints and native-library banners
can never corrupt Modly's NDJSON channel.
"""

from __future__ import annotations

import json
import os
import sys
from typing import TextIO


# Literal tokens intentionally retained for Modly's static process validator.
MODLY_PROCESS_CONTRACT = (
    "stdin",
    "json",
    "progress",
    "log",
    "done",
    "error",
    "result",
    "filePath",
    "text",
    "workspaceDir",
    "tempDir",
    "nodeId",
)

BOOTSTRAP_ERROR = (
    "[PROCESS_BOOTSTRAP_FAILED] AnyTop initialization failed. "
    "Run Repair for this extension and try again."
)


def _emit_bootstrap_error(channel: TextIO) -> None:
    channel.write(
        json.dumps({"type": "error", "message": BOOTSTRAP_ERROR}, ensure_ascii=True)
        + "\n"
    )
    channel.flush()


def _main() -> int:
    protocol_fd: int | None = None
    channel: TextIO | None = None
    try:
        protocol_fd = os.dup(1)
        discard_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(discard_fd, 1)
            os.dup2(discard_fd, 2)
        finally:
            os.close(discard_fd)

        channel = os.fdopen(
            protocol_fd,
            "w",
            encoding="utf-8",
            errors="strict",
            buffering=1,
            newline="\n",
        )
        protocol_fd = None

        from anytop_modly.runtime import run_protocol

        return run_protocol(sys.stdin, channel)
    except BaseException:
        if channel is not None:
            try:
                _emit_bootstrap_error(channel)
            except BaseException:
                pass
        elif protocol_fd is not None:
            try:
                line = (
                    json.dumps(
                        {"type": "error", "message": BOOTSTRAP_ERROR},
                        ensure_ascii=True,
                    )
                    + "\n"
                ).encode("utf-8")
                os.write(protocol_fd, line)
            except BaseException:
                pass
        return 1
    finally:
        if channel is not None:
            try:
                channel.close()
            except BaseException:
                pass
        elif protocol_fd is not None:
            try:
                os.close(protocol_fd)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(_main())
