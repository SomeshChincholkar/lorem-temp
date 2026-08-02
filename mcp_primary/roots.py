"""
roots.py

Resolves the authorized filesystem root for the Primary MCP Clinical
Tools Server via the MCP *roots* protocol, and provides a safe path
join helper to prevent directory-traversal outside that root.

The server never accepts a raw filesystem path from a tool argument.
Instead, the connecting client (Discharge Monitor Agent) registers a
root (e.g. file:///.../Data/incoming) during MCP session initialization,
and every filesystem-touching tool must resolve that root first.
"""

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from mcp.server.fastmcp import Context


def _uri_to_path(uri: str) -> Path:
    """
    Convert a file:// URI to a Path, correctly on both Windows and
    POSIX. A naive str.replace("file://", "") breaks on Windows because
    file:///C:/Users/... leaves a leading slash before the drive letter
    (\\C:\\Users\\...). url2pathname handles that correctly.
    """
    parsed = urlparse(uri)
    return Path(url2pathname(parsed.path))


async def resolve_authorized_root(ctx: Context) -> Path:
    """
    Ask the connected client which folder(s) it authorizes this server
    to operate on, via ctx.session.list_roots(). We only ever trust
    roots[0].

    Raises:
        PermissionError: no roots were registered by the client.
        FileNotFoundError: the registered root doesn't exist on disk.
    """
    result = await ctx.session.list_roots()  # -> ListRootsResult(roots=[Root(uri=..., name=...), ...])
    roots = result.roots if result else []
    if not roots:
        raise PermissionError(
            "No authorized root registered by the client. "
            "The Discharge Monitor Agent must register a root "
            "(e.g. file:///path/to/Data/incoming) before calling "
            "any filesystem tool on this server."
        )

    root_path = _uri_to_path(str(roots[0].uri))

    if not root_path.exists():
        raise FileNotFoundError(f"Authorized root does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Authorized root is not a directory: {root_path}")

    return root_path.resolve()


def safe_join(root: Path, relative: str) -> Path:
    """
    Join `relative` onto `root` and guarantee the result stays inside
    `root`. Blocks path traversal attempts like "../../etc/passwd" or
    absolute-path overrides.

    Raises:
        ValueError: the resolved candidate path escapes `root`.
    """
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError(
            f"Path traversal blocked: '{relative}' resolves outside "
            f"authorized root '{root}'"
        )
    return candidate