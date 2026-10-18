"""Bounded, regular-file reads for control metadata, never an execution API."""
import hashlib
import json
import os
from pathlib import Path
import stat


def contained_path(root, path):
    root, path = Path(root).absolute(), Path(path).absolute()
    relative = path.relative_to(root)
    if '..' in relative.parts or root.is_symlink():
        raise ValueError('unsafe root or path traversal')
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError('symlink component refused')
    return path


def read_bytes(path, limit=2_000_000, root=None):
    path = Path(path)
    contained_path(root or path.parent, path)
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
    with os.fdopen(os.open(path, flags), 'rb') as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('not a regular file')
        if before.st_size > limit:
            raise ValueError('file exceeds read budget')
        raw = handle.read(limit + 1)
        after = os.fstat(handle.fileno())
        final = path.stat(follow_symlinks=False)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if len(raw) > limit or identity(before) != identity(after) or identity(after) != identity(final):
            raise ValueError('file changed or exceeded read budget')
        return raw


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def read_json(path, limit=2_000_000, root=None):
    return json.loads(read_bytes(path, limit, root), object_pairs_hook=unique_object)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()
