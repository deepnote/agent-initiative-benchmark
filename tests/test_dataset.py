import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import httpx

from benchmark.asset_download import download, hub_url
from benchmark.dataset import Release, check_data, prepare_data, preparation_lock
from benchmark.runtime.failure_causes import RuntimeInputError
from benchmark.runtime.baseline_access import BaselineAccessError


def sha(data):
    return hashlib.sha256(data).hexdigest()


def small_release(root, *, members=None):
    root = Path(root)
    (root / "data").mkdir(parents=True)
    contents = {
        "slack/channels.parquet": b"example channels",
        "slack/users.parquet": b"example users",
    }
    index = b"".join(
        json.dumps(dict(path=name, size_bytes=len(data), sha256=sha(data))).encode()
        + b"\n"
        for name, data in contents.items()
    )
    index = gzip.compress(index, mtime=0)
    (root / "file-index.jsonl.gz").write_bytes(index)
    archive = root / "data/slack.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name, data, kind in members or [
            (name, data, tarfile.REGTYPE) for name, data in contents.items()
        ]:
            member = tarfile.TarInfo(name)
            member.size, member.type = len(data), kind
            member.linkname = "../../escape"
            stream.addfile(member, io.BytesIO(data))
    package = dict(
        service="slack",
        path="data/slack.tar.gz",
        extraction_root=".",
        size_bytes=archive.stat().st_size,
        sha256=sha(archive.read_bytes()),
        file_count=len(contents),
        unpacked_bytes=sum(map(len, contents.values())),
    )
    manifest = dict(
        schema_version=1,
        version="1.0.0",
        source_snapshot_sha256="0" * 64,
        packages=[package],
        file_count=len(contents),
        unpacked_bytes=package["unpacked_bytes"],
        archive_bytes=package["size_bytes"],
        file_index=dict(path="file-index.jsonl.gz", sha256=sha(index)),
    )
    raw = json.dumps(manifest).encode()
    (root / "manifest.json").write_bytes(raw)
    search = root / "search-source"
    search.mkdir()
    (search / "model.onnx").write_bytes(b"pinned model")
    lock = dict(
        dataset=dict(
            manifest_sha256=sha(raw),
            file_index_sha256=sha(index),
            file_index_bytes=len(index),
            build_ref=dict(manifest_sha256="0" * 64),
            huggingface=dict(repository=None, revision=None),
        ),
        search=dict(
            model="test-model",
            repository="example/model",
            revision="1" * 40,
            files=[dict(path="model.onnx", bytes=12, sha256=sha(b"pinned model"))],
        ),
    )
    return Release(lock, raw), search, contents


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "release"
        self.release, self.search, self.contents = small_release(self.source)
        self.target, self.cache = self.root / "prepared", self.root / "cache"
        self.options = dict(
            release=self.release, cache_dir=self.cache, progress=lambda value: None
        )

    def prepare(self, **changes):
        return prepare_data(
            self.target,
            source=self.source,
            search_source=self.search,
            **{**self.options, **changes},
        )

    def test_local_prepare_check_and_repeat_without_network_or_copies(self):
        with patch(
            "httpx.Client.send", side_effect=AssertionError("Network forbidden")
        ):
            result = self.prepare()
            self.assertEqual(result["files"], 2)
            identities = {
                name: (self.target / name).stat().st_ino for name in self.contents
            }
            check_data(self.target, **self.options)
            check_data(self.target, full=True, **self.options)
            self.prepare()
        for name, data in self.contents.items():
            self.assertEqual((self.target / name).read_bytes(), data)
            self.assertEqual((self.target / name).stat().st_ino, identities[name])
        self.assertFalse(list(self.target.rglob("*.tar.gz")))
        self.assertTrue((self.source / "data/slack.tar.gz").is_file())

    def test_existing_extracted_folder_needs_no_archives(self):
        self.prepare()
        with patch(
            "httpx.Client.send", side_effect=AssertionError("Network forbidden")
        ):
            prepare_data(
                self.target,
                source=self.target,
                search_source=self.search,
                **self.options,
            )
        self.assertFalse((self.target / "data/slack.tar.gz").exists())

    def test_completed_file_checks_survive_a_later_interruption(self):
        with patch("benchmark.dataset.prepare_search", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.prepare()
        self.assertTrue((self.target / "slack").is_dir())
        from benchmark.runtime import baseline_access

        original = baseline_access._file_hash

        def checked(path):
            if Path(path).parent.name == "slack":
                raise AssertionError("Completed service must reuse committed checks")
            return original(path)

        with patch("benchmark.runtime.baseline_access._file_hash", side_effect=checked):
            self.prepare()

    def test_changed_file_is_detected_after_cached_verification(self):
        self.prepare()
        file = self.target / "slack/users.parquet"
        file.write_bytes(b"changed users")
        with self.assertRaises((ValueError, BaselineAccessError)):
            check_data(self.target, **self.options)
        with self.assertRaises((ValueError, BaselineAccessError)):
            self.prepare()
        self.assertEqual(file.read_bytes(), b"changed users")

    def test_linked_file_and_wrong_metadata_are_rejected(self):
        self.prepare()
        path = self.target / "slack/users.parquet"
        path.unlink()
        path.symlink_to(self.target / "slack/channels.parquet")
        with self.assertRaises((ValueError, BaselineAccessError)):
            check_data(self.target, **self.options)
        (self.target / "manifest.json").write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeInputError, "size differs"):
            check_data(self.target, **self.options)

    def test_unknown_links_traversal_duplicates_and_wrong_bytes_do_not_publish(self):
        bad_members = [
            [("slack/users.parquet", b"example users", tarfile.SYMTYPE)],
            [("slack/../escape", b"bad", tarfile.REGTYPE)],
            [("slack/unlisted", b"bad", tarfile.REGTYPE)],
            [("slack/users.parquet", b"example users", tarfile.REGTYPE)] * 2,
            [("slack/users.parquet", b"changed users", tarfile.REGTYPE)],
        ]
        for n, members in enumerate(bad_members):
            with self.subTest(n=n):
                source = self.root / f"bad-{n}"
                release, search, _ = small_release(source, members=members)
                target = self.root / f"out-{n}"
                with self.assertRaises((ValueError, BaselineAccessError)):
                    prepare_data(
                        target,
                        source=source,
                        search_source=search,
                        **{**self.options, "release": release},
                    )
                self.assertFalse((target / "slack").exists())
                self.assertFalse((self.root / "escape").exists())

    def test_interrupted_extract_resumes_without_replacing_completed_service(self):
        from benchmark import dataset

        original = dataset.extract_service
        with patch("benchmark.dataset.extract_service", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.prepare()
        self.assertFalse((self.target / "slack").exists())
        with patch("benchmark.dataset.extract_service", wraps=original) as extract:
            self.prepare()
            self.assertEqual(extract.call_count, 1)
        with patch(
            "benchmark.dataset.extract_service",
            side_effect=AssertionError("Do not extract again"),
        ):
            self.prepare()

    def test_dry_run_has_no_side_effects_and_unset_source_fails_clearly(self):
        self.assertGreater(
            prepare_data(self.target, dry_run=True, **self.options)["peak_free_bytes"],
            0,
        )
        self.assertFalse(self.target.exists())
        with self.assertRaisesRegex(RuntimeInputError, "not configured"):
            prepare_data(self.target, **self.options)
        self.assertFalse(self.target.exists())

    def test_insufficient_space_and_concurrent_prepare(self):
        with patch(
            "benchmark.dataset.shutil.disk_usage",
            return_value=type("Disk", (), {"free": 0})(),
        ):
            with self.assertRaisesRegex(RuntimeInputError, "Not enough space"):
                self.prepare()
        with preparation_lock(self.target):
            with self.assertRaisesRegex(ValueError, "already running"):
                self.prepare()

    def test_mock_hub_download_cleans_archives_and_never_fetches_latest(self):
        requests = []

        def handler(request):
            requests.append(request.url.path)
            tail = request.url.path.split("/resolve/" + "2" * 40 + "/")[-1]
            if request.url.path.startswith("/datasets/"):
                return httpx.Response(200, content=(self.source / tail).read_bytes())
            return httpx.Response(200, content=b"pinned model")

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            prepare_data(
                self.target,
                repository="example/dataset",
                revision="2" * 40,
                client=client,
                **self.options,
            )
            count = len(requests)
            prepare_data(
                self.target,
                repository="example/dataset",
                revision="2" * 40,
                client=client,
                **self.options,
            )
        self.assertEqual(len(requests), count)
        self.assertTrue(all("/main/" not in name for name in requests))
        self.assertFalse(list(self.target.rglob("*.tar.gz")))
        self.assertTrue((self.target / "slack/users.parquet").is_file())

    def test_search_weights_are_checked_and_changed_revision_is_rejected(self):
        self.prepare()
        path = self.target / ".benchmark/search" / ("1" * 40) / "model.onnx"
        path.write_bytes(b"wrong model!")
        with self.assertRaises((ValueError, BaselineAccessError)):
            check_data(self.target, **self.options)
        changed = copy.deepcopy(self.release.lock)
        changed["dataset"]["manifest_sha256"] = "0" * 64
        with self.assertRaises(RuntimeInputError):
            Release(changed, self.release.raw)


class DownloadTests(unittest.TestCase):
    def test_resumes_exact_range_and_restarts_when_range_is_ignored(self):
        for ignore in [False, True]:
            with self.subTest(
                ignore=ignore
            ), tempfile.TemporaryDirectory() as directory:
                target = Path(directory).resolve() / "archive"
                target.with_suffix(".part").write_bytes(b"first")
                data = b"first-and-last"
                seen = []

                def handler(request):
                    seen.append(request.headers["range"])
                    return httpx.Response(
                        200 if ignore else 206,
                        content=data if ignore else data[5:],
                        headers=(
                            {}
                            if ignore
                            else {"content-range": f"bytes 5-{len(data)-1}/{len(data)}"}
                        ),
                    )

                with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                    download(
                        client,
                        "https://example.test/archive",
                        target,
                        len(data),
                        sha(data),
                    )
                self.assertEqual(target.read_bytes(), data)
                self.assertEqual(seen, ["bytes=5-"])

    def test_wrong_range_hash_and_oversize_fail(self):
        for headers, data, size, checksum in [
            ({"content-range": "bytes 1-4/5"}, b"abcde", 5, sha(b"abcde")),
            ({}, b"abcde", 5, "0" * 64),
            ({}, b"too large", 2, "0" * 64),
        ]:
            with self.subTest(data=data), tempfile.TemporaryDirectory() as directory:
                target = Path(directory).resolve() / "archive"
                status = 206 if headers else 200
                with httpx.Client(
                    transport=httpx.MockTransport(
                        lambda req: httpx.Response(
                            status, headers=headers, content=data
                        )
                    )
                ) as client:
                    with self.assertRaises(RuntimeInputError):
                        download(
                            client,
                            "https://example.test/archive",
                            target,
                            size,
                            checksum,
                        )
                self.assertFalse(target.exists())

    def test_interrupted_stream_keeps_verified_prefix_for_next_request(self):
        class Broken(httpx.SyncByteStream):
            def __iter__(self):
                yield b"12345678"
                raise httpx.ReadError("Disconnected")

        data = b"123456789012"
        with tempfile.TemporaryDirectory() as directory, patch(
            "benchmark.asset_download.CHUNK", 4
        ):
            target = Path(directory).resolve() / "archive"
            with httpx.Client(
                transport=httpx.MockTransport(
                    lambda req: httpx.Response(200, stream=Broken())
                )
            ) as client:
                with self.assertRaisesRegex(RuntimeInputError, "resume"):
                    download(
                        client,
                        "https://example.test/archive",
                        target,
                        len(data),
                        sha(data),
                    )
            self.assertEqual(target.with_suffix(".part").read_bytes(), data[:8])

            def resumed(req):
                self.assertEqual(req.headers["range"], "bytes=8-")
                return httpx.Response(
                    206, content=data[8:], headers={"content-range": "bytes 8-11/12"}
                )

            with httpx.Client(transport=httpx.MockTransport(resumed)) as client:
                download(
                    client, "https://example.test/archive", target, len(data), sha(data)
                )
            self.assertEqual(target.read_bytes(), data)

    def test_revision_must_be_a_complete_commit(self):
        for revision in [None, "main", "v1.0", "123abcd"]:
            with self.assertRaises(RuntimeInputError):
                hub_url("example/dataset", revision, "manifest.json", dataset=True)
        self.assertIn(
            "/datasets/example/dataset/resolve/" + "a" * 40 + "/",
            hub_url("example/dataset", "a" * 40, "manifest.json", dataset=True),
        )


if __name__ == "__main__":
    unittest.main()
