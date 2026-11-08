"""Fixed local search weights shared by all tasks."""

import json
from pathlib import Path

from .asset_download import check_file, copy_checked, download, hub_url


def search_pin():
    return json.loads(Path(__file__).with_name("assets.lock.json").read_text())[
        "search"
    ]


def search_path(root, pin=None):
    pin = pin or search_pin()
    return Path(root) / ".benchmark" / "search" / pin["revision"]


def check_search(root, pin=None, *, cache=None, full=False):
    pin = pin or search_pin()
    target = search_path(root, pin)
    for row in pin["files"]:
        path = target / row["path"]
        if cache is None or full:
            check_file(path, row["bytes"], row["sha256"])
        else:
            cache.check(path, row["sha256"], row["bytes"])
    return target


def prepare_search(root, client, pin=None, *, source=None, progress=print):
    pin = pin or search_pin()
    target = search_path(root, pin)
    progress("Checking search model")
    for row in pin["files"]:
        destination = target / row["path"]
        if source is not None:
            copy_checked(
                Path(source) / row["path"], destination, row["bytes"], row["sha256"]
            )
        else:
            download(
                client,
                hub_url(pin["repository"], pin["revision"], row["path"]),
                destination,
                row["bytes"],
                row["sha256"],
                progress=progress,
            )
    return target
