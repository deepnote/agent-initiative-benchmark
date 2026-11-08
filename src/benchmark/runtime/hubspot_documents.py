"""Read only the contract linked by an accessible pinned CRM record."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from contextlib import contextmanager
import threading

from benchmark.tools.hubspot.hubspot import HubSpotToolError
from .hubspot_environment import document_path, check, HubSpotEnvironmentError
from .packages import hash_file
from benchmark.prompts import load_text
from .local_objects import StorageError

_PDF_LOCK = threading.RLock()


@contextmanager
def bounded_pdf_streams(limit):
    from pypdf import filters

    names = (
        "MAX_ARRAY_BASED_STREAM_OUTPUT_LENGTH",
        "LZW_MAX_OUTPUT_LENGTH",
        "RUN_LENGTH_MAX_OUTPUT_LENGTH",
        "ZLIB_MAX_OUTPUT_LENGTH",
    )
    with _PDF_LOCK:
        original = {name: getattr(filters, name) for name in names}
        try:
            for name in names:
                setattr(filters, name, min(original[name], limit))
            yield
        finally:
            for name, value in original.items():
                setattr(filters, name, value)


def safe_file(root, relative, limit):
    root = Path(root)
    target = root
    check(not root.is_symlink(), "Contract root cannot be a symlink")
    for part in relative.split("/"):
        target = target / part
        check(not target.is_symlink(), "Linked contract cannot traverse symlinks")
    check(
        target.is_file() and target.stat().st_nlink == 1,
        "Linked contract is missing or unsafe",
    )
    check(
        target.stat().st_size <= limit,
        "Linked contract exceeds file byte limit",
        "source_file_limit",
    )
    return target


def linked_contract_paths(env, selector, view):
    """Readable versions share the selected parent's current access checks."""
    selector = dict(
        object_type=env.tools._object_type(selector["object_type"]),
        record_id=env.tools._record_id(selector["record_id"]),
        property_name=selector["property_name"],
    )
    # A missing link definition on this object can still offer real native navigation.
    if (
        selector["object_type"],
        selector["property_name"],
    ) not in env.tools._definitions:
        env.tools.readable(selector["object_type"], selector["record_id"])
        raise HubSpotToolError(
            "invalid_property", load_text("tools/hubspot/wrong_parent.txt").strip()
        )
    env._visible("documents", selector, view)
    check(
        selector["property_name"]
        in {"contract_document_path", "contract_markdown_path"},
        "Property is not a supported contract link",
    )
    linked = env._linked_paths(selector, view)
    if view == "effective":
        linked |= env._linked_paths(selector, "baseline")
    return linked


def contract_artifact(
    env,
    object_type,
    record_id,
    property_name="contract_document_path",
    document_id=None,
    view="effective",
):
    """Exact linked bytes for native text reads and the owner PDF panel."""
    kind = env.tools._object_type(object_type)
    rid = env.tools._record_id(record_id)
    selector = dict(object_type=kind, record_id=rid, property_name=property_name)
    # Only stored document-link properties qualify. A random CRM string cannot
    # turn into a file selector, even if that string looks like a safe path.
    linked = linked_contract_paths(env, selector, view)
    current = env._row("property_values", selector, view)
    document_id = (
        document_id
        if document_id is not None
        else (current["value"] if current else None)
    )
    if document_id is None:
        raise HubSpotToolError("not_found", "This record has no linked contract")
    document_path(document_id)
    # Original linked artifacts remain addressable after a replacement, using
    # the same still-readable parent. No task can overwrite their bytes.
    if document_id not in linked:
        raise HubSpotToolError(
            "not_linked", load_text("tools/hubspot/not_linked.txt").strip()
        )
    limit = env.settings["max_document_bytes"]
    delta = (
        next(
            (
                r
                for r in env.delta.get("documents", [])
                if all(r[k] == v for k, v in dict(selector, id=document_id).items())
            ),
            None,
        )
        if view == "effective"
        else None
    )
    if delta:
        check(
            delta["bytes"] <= limit,
            "Contract artifact exceeds file byte limit",
            "source_file_limit",
        )
        path = safe_file(env.package, "delta/hubspot/" + document_id, limit)
        check(
            hash_file(path) == (delta["bytes"], delta["sha256"]),
            "Pinned contract artifact changed",
        )
    else:
        root = env.acquire(("hubspot/" + document_id,), max_file_bytes=limit)
        path = safe_file(root, document_id, limit)
    byte_count, sha = hash_file(path)
    raw = path.read_bytes()
    check(
        len(raw) == byte_count and hashlib.sha256(raw).hexdigest() == sha,
        "Contract changed during read",
    )
    return raw, dict(
        document_id=document_id,
        sha256=sha,
        bytes=byte_count,
        object_type=kind,
        record_id=rid,
        property_name=property_name,
    )


EXTRACTION_VERSION = "hubspot-text-v3"


def cache_scope(env):
    # Callers may select a shared disk root. Content never crosses access scopes.
    return dict(
        source=env.document_source,
        owner=env.settings["agent_owner_id"],
        teams=sorted(env.settings["visible_team_ids"]),
        restricted=env.settings["allow_restricted_properties"],
        limits={
            k: env.settings[k]
            for k in ("max_document_bytes", "max_document_pages", "max_page_text_bytes")
        },
        extraction=EXTRACTION_VERSION,
        parser=__import__("pypdf").__version__,
    )


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _cache_file(env, identity, page):
    scope = _digest(cache_scope(env))
    version = _digest([identity["document_id"], identity["sha256"]])
    return env.document_cache / scope / version / (str(page) + ".json")


def _cached(path, limit):
    if not path.is_file() or path.is_symlink() or path.stat().st_size > limit:
        return None
    try:
        payload = json.loads(path.read_text())
        value = payload["value"]
        if payload["sha256"] != _digest(value):
            return None
        return value
    except (ValueError, KeyError, OSError):
        return None


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".text-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(
                dict(value=value, sha256=_digest(value)), stream, ensure_ascii=False
            )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ContractText:
    """Verify original bytes on each call, then reuse only exact scoped extraction."""

    def __init__(self, env, selector, view="effective"):
        self.env = env
        self.raw, self.identity = contract_artifact(env, **selector, view=view)
        self.pdf = None
        self.md = self.identity["document_id"].endswith(".md")
        self.status = "readable"
        header = _cached(_cache_file(env, self.identity, 0), 4096)
        if isinstance(header, dict) and (
            (
                type(header.get("page_count")) is int
                and 1 <= header["page_count"] <= env.settings["max_document_pages"]
                and header.get("status") == "readable"
            )
            or (
                header.get("page_count") is None
                and header.get("status") == "unreadable_pdf"
            )
        ):
            self.page_count = header["page_count"]
            self.status = header["status"]
        else:
            header = None
            if self.md:
                self.page_count = 1
            else:
                self._parse_pdf()
        if header is None:
            _save(
                _cache_file(env, self.identity, 0),
                dict(page_count=self.page_count, status=self.status),
            )

    def _parse_pdf(self):
        from io import BytesIO
        from pypdf import PdfReader
        from pypdf.errors import LimitReachedError

        try:
            with bounded_pdf_streams(self.env.settings["max_page_text_bytes"]):
                self.pdf = PdfReader(BytesIO(self.raw), strict=True)
                self.page_count = len(self.pdf.pages)
                check(
                    0 < self.page_count <= self.env.settings["max_document_pages"],
                    "Contract exceeds PDF page limit",
                    "source_file_limit",
                )
        except (HubSpotToolError, HubSpotEnvironmentError):
            raise
        except LimitReachedError:
            check(False, "PDF decoded stream exceeds byte limit", "source_file_limit")
        except Exception:
            self.status, self.page_count = "unreadable_pdf", None

    def page(self, number):
        if self.page_count is None:
            return dict(text="", status=self.status)
        if not 1 <= number <= self.page_count:
            raise HubSpotToolError("invalid_input", "Contract page is out of range")
        path = _cache_file(self.env, self.identity, number)
        cached = _cached(path, 8 * self.env.settings["max_page_text_bytes"] + 4096)
        if (
            isinstance(cached, dict)
            and isinstance(cached.get("text"), str)
            and len(cached["text"].encode()) <= self.env.settings["max_page_text_bytes"]
        ):
            return cached
        text, status = "", self.status
        if self.md:
            try:
                text = self.raw.decode("utf-8")
                status = "readable" if text.strip() else "unreadable_or_scanned_page"
            except UnicodeError:
                status = "unreadable_encoding"
        elif status == "readable":
            if self.pdf is None:
                self._parse_pdf()
            if self.status != "readable":
                status = self.status
            else:
                from pypdf.errors import LimitReachedError

                try:
                    with bounded_pdf_streams(self.env.settings["max_page_text_bytes"]):
                        text = self.pdf.pages[number - 1].extract_text() or ""
                    status = (
                        "readable" if text.strip() else "unreadable_or_scanned_page"
                    )
                except LimitReachedError:
                    check(
                        False,
                        "PDF decoded stream exceeds byte limit",
                        "source_file_limit",
                    )
                except Exception:
                    status = "unreadable_pdf"
        check(
            len(text.encode()) <= self.env.settings["max_page_text_bytes"],
            "Contract page text exceeds byte limit",
            "result_limit",
        )
        result = dict(text=text, status=status)
        _save(path, result)
        return result


def read_contract(
    env,
    object_type,
    record_id,
    property_name,
    document_id,
    page,
    offset,
    max_chars,
    view,
    max_pages=1,
):
    document = ContractText(
        env,
        dict(
            object_type=object_type,
            record_id=record_id,
            property_name=property_name,
            document_id=document_id,
        ),
        view,
    )
    pages, remaining = [], max_chars
    next_position = None
    stop = (
        min(document.page_count + 1, page + max_pages)
        if document.page_count is not None
        else page + 1
    )
    for number in range(page, stop):
        value = document.page(number)
        text = value["text"]
        start = offset if number == page else 0
        if document.page_count is None:
            start = 0
        if start > len(text):
            raise HubSpotToolError(
                "invalid_input", "Contract text offset is out of range"
            )
        end = min(len(text), start + remaining)
        pages.append(
            dict(
                page=number,
                status=value["status"],
                text=text[start:end],
                offset=start,
                end=end,
            )
        )
        remaining -= end - start
        next_position = (
            dict(page=number, offset=end)
            if end < len(text)
            else (
                dict(page=number + 1, offset=0)
                if document.page_count is not None and number < document.page_count
                else None
            )
        )
        if remaining <= 0:
            break
    if not pages:
        raise HubSpotToolError("invalid_input", "Contract page is out of range")
    base = dict(**document.identity, page_count=document.page_count, next=next_position)
    if max_pages == 1:
        return dict(**base, **pages[0])
    return dict(**base, pages=pages)


def document_links(env, archived=False, object_type=None, record_id=None):
    """Stream only links with a readable current parent and readable link field.

    Baseline links remain reachable when the effective property was replaced.
    A task cannot introduce a document merely by placing a private artifact on disk.
    """
    if record_id is not None and object_type is None:
        raise HubSpotToolError(
            "invalid_input",
            json.loads(load_text("tools/hubspot/messages.json"))[
                "parent_type_required"
            ],
        )
    if object_type is not None:
        object_type = env.tools._object_type(object_type)
    if record_id is not None:
        record_id = env.tools._record_id(record_id)
    teams = ",".join("?" for _ in env.settings["visible_team_ids"])
    # Start with current readable parents. The native history index begins with
    # (object_type, record_id, property_name). A property-only UNION scans the
    # full CRM history again for every preparation chunk and every query.
    for kind in (
        (object_type,)
        if object_type is not None
        else ("companies", "contacts", "deals")
    ):
        sql = (
            "SELECT r.id record_id,d.name property_name FROM effective_" + kind + " r "
            "CROSS JOIN baseline.property_definitions d "
            "WHERE r.access_team_id IN ("
            + teams
            + ") AND r.archived=? AND d.object_type=? "
            "AND d.name IN ('contract_document_path','contract_markdown_path')"
        )
        args = [*env.settings["visible_team_ids"], int(archived), kind]
        if not env.settings["allow_restricted_properties"]:
            sql += " AND d.access_level='standard'"
        if record_id is not None:
            sql += " AND r.id=?"
            args.append(record_id)
        sql += " ORDER BY r.id,d.name"
        for row in env.tools._connection.execute(sql, args):
            selector = dict(
                object_type=kind,
                record_id=row["record_id"],
                property_name=row["property_name"],
            )
            paths = env._linked_paths(selector, "effective") | env._linked_paths(
                selector, "baseline"
            )
            for path in sorted(p for p in paths if p):
                yield dict(document_id=path, **selector)


def _inventory(env, archived, object_type, record_id):
    # Only the small native link catalog is materialized, never file payloads.
    check(
        bool(env.scope & {"hubspot_read_contract", "hubspot_search_contracts"}),
        "HubSpot evidence is inaccessible through configured reads",
    )
    documents = {}
    for row in document_links(env, archived, object_type, record_id):
        row = dict(row)
        document_id = row.pop("document_id")
        documents.setdefault(document_id, []).append(
            dict(row, url="hubspot://" + row["object_type"] + "/" + row["record_id"])
        )
    return sorted(documents.items())


def _extract_contracts(
    env,
    query,
    object_type=None,
    record_id=None,
    archived=False,
    after=None,
    page_size=5,
    max_documents=5,
    max_pages=10,
    context_chars=1500,
):
    inventory = _inventory(env, archived, object_type, record_id)
    # Visible link identities bind continuation. Hidden links do not change offsets.
    scope = _digest(
        [
            "contract_search",
            query,
            archived,
            object_type,
            record_id,
            context_chars,
            cache_scope(env),
            inventory,
        ]
    )
    from benchmark.tools.hubspot.hubspot import HubSpotTools

    coordinate, prior_failure = divmod(
        HubSpotTools._decode_cursor(env.tools, after, scope), 2
    )
    stride = env.settings["max_page_text_bytes"] + 1
    pages_per_doc = env.settings["max_document_pages"] + 1
    index, rest = divmod(coordinate, stride * pages_per_doc)
    page, offset = divmod(rest, stride)
    page = page or 1
    _prefetch_contracts(env, inventory[index : index + max_documents])
    results, failures, scanned = [], [], 0
    opened = 0
    while (
        index < len(inventory)
        and scanned < max_pages
        and opened < max_documents
        and len(results) < page_size
    ):
        document_id, parents = inventory[index]
        selector = {k: v for k, v in parents[0].items() if k != "url"}
        try:
            document = ContractText(env, dict(selector, document_id=document_id))
        except (
            HubSpotToolError,
            HubSpotEnvironmentError,
            StorageError,
            OSError,
            ValueError,
        ) as error:
            failures.append(
                dict(
                    document_id=document_id,
                    parents=parents,
                    status="unavailable",
                    code=getattr(error, "code", "unavailable"),
                )
            )
            index, page, offset = index + 1, 1, 0
            opened += 1
            continue
        opened += 1
        if document.page_count is None:
            failures.append(
                dict(
                    document_id=document_id,
                    parents=parents,
                    status=document.status,
                    page_count=None,
                )
            )
            index, page, offset = index + 1, 1, 0
            continue
        while (
            page <= document.page_count
            and scanned < max_pages
            and len(results) < page_size
        ):
            try:
                value = document.page(page)
            except (
                HubSpotToolError,
                HubSpotEnvironmentError,
                StorageError,
                OSError,
                ValueError,
            ) as error:
                failures.append(
                    dict(
                        document_id=document_id,
                        page=page,
                        parents=parents,
                        status="unavailable",
                        code=getattr(error, "code", "unavailable"),
                    )
                )
                page, offset, scanned = page + 1, 0, scanned + 1
                continue
            scanned += 1
            if value["status"] != "readable":
                failures.append(
                    dict(
                        document_id=document_id,
                        page=page,
                        status=value["status"],
                        parents=parents,
                    )
                )
            text = value["text"]
            match = re.search(re.escape(query), text[offset:], re.IGNORECASE)
            if match:
                start_match, end_match = offset + match.start(), offset + match.end()
                start, end = max(0, start_match - context_chars), min(
                    len(text), end_match + context_chars
                )
                results.append(
                    dict(
                        **document.identity,
                        parents=parents,
                        page=page,
                        page_count=document.page_count,
                        offset=start,
                        end=end,
                        text=text[start:end],
                        match_offset=start_match,
                        match_end=end_match,
                    )
                )
                offset = (
                    end if end == len(text) else max(end_match, end - len(query) + 1)
                )
                if offset < len(text):
                    continue
            page, offset = page + 1, 0
        if page > document.page_count:
            index, page, offset = index + 1, 1, 0
        else:
            break
    continuation = None
    if index < len(inventory):
        position = index * stride * pages_per_doc + page * stride + offset
        continuation = HubSpotTools._encode_cursor(
            env.tools, position * 2 + int(bool(prior_failure or failures)), scope
        )
    return dict(
        results=results,
        failures=failures,
        coverage=dict(
            total_documents=len(inventory),
            next_document=index,
            pages_examined=scanned,
            complete=continuation is None and not failures and not prior_failure,
            prior_pages=bool(after),
            unsearched=continuation is not None,
            failed_documents_this_call=len({r["document_id"] for r in failures}),
            prior_failures=bool(prior_failure),
        ),
        **({"paging": dict(next=dict(after=continuation))} if continuation else {}),
    )


def prepare_contract_search(
    env,
    *,
    after=None,
    max_documents=5,
    max_pages=10,
    archived=False,
    object_type=None,
    record_id=None,
):
    """Callable bounded preparation hook. C owns scheduling and persistent cache root.

    Uses exactly the same scoped scanner and cache as the read tool. A NUL marker
    is reserved for preparation and never exposed as a public evidence query.
    """
    return _extract_contracts(
        env,
        query="\0",
        after=after,
        max_documents=max_documents,
        max_pages=max_pages,
        archived=archived,
        object_type=object_type,
        record_id=record_id,
    )


# Independent of extraction chunks. These bound one prepared-corpus query.
QUERY_MAX_DOCUMENTS = 10000
QUERY_MAX_BYTES = 134217728
QUERY_MAX_PAGES = 100000


def _prefetch_contracts(env, inventory):
    """Acquire a bounded visible corpus once through the existing lease owner.

    Local readers need no prefetch. Oversized selections keep the incremental
    path. A failed batch falls back to exact reads so one missing document does
    not conceal every other document or turn a coverage gap into absence.
    """
    describe = getattr(env.acquire, "file_inventory", None)
    if describe is None:
        return
    paths = []
    for document_id, parents in inventory[:QUERY_MAX_DOCUMENTS]:
        selector = {k: v for k, v in parents[0].items() if k != "url"}
        if not any(
            all(row[k] == value for k, value in dict(selector, id=document_id).items())
            for row in env.delta.get("documents", [])
        ):
            paths.append("hubspot/" + document_id)
    if not paths:
        return
    try:
        records = describe(paths)
        if (
            records is None
            or sum(row["size_bytes"] for row in records) > QUERY_MAX_BYTES
            or any(
                row["size_bytes"] > env.settings["max_document_bytes"]
                for row in records
            )
        ):
            return
        env.acquire(tuple(paths), max_file_bytes=env.settings["max_document_bytes"])
    except (StorageError, OSError, ValueError):
        pass  # Individual reads retain exact failures and all native checks.


def search_contracts(
    env,
    query,
    object_type=None,
    record_id=None,
    archived=False,
    after=None,
    page_size=5,
    context_chars=1500,
    match_mode="all",
):
    """Query verified prepared page text, without parsing or extraction continuations.

    Cache entries are disposable. Missing, failed or over-limit pages are gaps.
    Only matching passages are paged. Every query rechecks the accessible source.
    """
    terms = list(dict.fromkeys(re.findall(r"\w+", query, re.UNICODE)))
    if match_mode != "phrase" and not 1 <= len(terms) <= 12:
        raise HubSpotToolError(
            "invalid_input", load_text("tools/hubspot/contract_query_error.txt").strip()
        )
    inventory = _inventory(env, archived, object_type, record_id)
    _prefetch_contracts(env, inventory)
    rows, failures, identities = [], [], []
    byte_count = pages = prepared = 0
    bounded = False
    for index, (document_id, parents) in enumerate(inventory):
        if index >= QUERY_MAX_DOCUMENTS:
            bounded = True
            break
        selector = {k: v for k, v in parents[0].items() if k != "url"}
        try:
            raw, identity = contract_artifact(env, **selector, document_id=document_id)
            byte_count += len(raw)
            if byte_count > QUERY_MAX_BYTES:
                bounded = True
                break
            identities.append(identity)
            header = _cached(_cache_file(env, identity, 0), 4096)
            count = header.get("page_count") if isinstance(header, dict) else None
            if (
                type(count) is not int
                or not 1 <= count <= env.settings["max_document_pages"]
                or header.get("status") != "readable"
            ):
                failures.append(
                    dict(
                        document_id=document_id,
                        parents=parents,
                        status="unprepared_or_unreadable",
                    )
                )
                continue
            document_complete = True
            for page in range(1, count + 1):
                pages += 1
                if pages > QUERY_MAX_PAGES:
                    bounded = True
                    break
                value = _cached(
                    _cache_file(env, identity, page),
                    8 * env.settings["max_page_text_bytes"] + 4096,
                )
                if not isinstance(value, dict) or not isinstance(
                    value.get("text"), str
                ):
                    failures.append(
                        dict(
                            document_id=document_id,
                            page=page,
                            parents=parents,
                            status="unprepared",
                        )
                    )
                    document_complete = False
                    continue
                text = value["text"]
                byte_count += len(text.encode())
                if byte_count > QUERY_MAX_BYTES:
                    bounded = True
                    break
                if (
                    len(text.encode()) > env.settings["max_page_text_bytes"]
                    or value.get("status") != "readable"
                ):
                    failures.append(
                        dict(
                            document_id=document_id,
                            page=page,
                            parents=parents,
                            status="unreadable",
                        )
                    )
                    document_complete = False
                    continue
                rows.append((identity, parents, page, count, text))
            if bounded:
                break
            prepared += int(document_complete)
        except (
            HubSpotToolError,
            HubSpotEnvironmentError,
            StorageError,
            OSError,
            ValueError,
        ) as error:
            failures.append(
                dict(
                    document_id=document_id,
                    parents=parents,
                    status="unavailable",
                    code=getattr(error, "code", "unavailable"),
                )
            )
    # Bind result paging to the exact accessible versions, prepared text and gaps.
    scope = _digest(
        [
            "prepared-contract-query-v2",
            query,
            match_mode,
            archived,
            object_type,
            record_id,
            context_chars,
            cache_scope(env),
            inventory,
            identities,
            rows,
            failures,
            bounded,
        ]
    )
    from benchmark.tools.hubspot.hubspot import HubSpotTools

    skip = HubSpotTools._decode_cursor(env.tools, after, scope)
    results, found, more = [], 0, False
    patterns = (
        [re.compile(re.escape(query), re.IGNORECASE)]
        if match_mode == "phrase"
        else [
            re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)
            for term in terms
        ]
    )
    for identity, parents, page, count, text in rows:
        present = [
            (term, pattern.search(text))
            for term, pattern in zip(
                terms if match_mode != "phrase" else [query], patterns
            )
        ]
        if match_mode == "all" and any(match is None for _, match in present):
            continue
        present = [(term, match) for term, match in present if match is not None]
        if not present:
            continue
        pattern = (
            patterns[0]
            if match_mode != "any"
            else re.compile("|".join(p.pattern for p in patterns), re.IGNORECASE)
        )
        offset = 0
        while (match := pattern.search(text, offset)) is not None:
            start, end = max(0, match.start() - context_chars), min(
                len(text), match.end() + context_chars
            )
            if found >= skip:
                if len(results) == page_size:
                    more = True
                    break
                results.append(
                    dict(
                        **identity,
                        parents=parents,
                        page=page,
                        page_count=count,
                        offset=start,
                        end=end,
                        text=text[start:end],
                        match_offset=match.start(),
                        match_end=match.end(),
                        **(
                            dict(
                                matched_terms=[
                                    dict(term=term, offset=hit.start(), end=hit.end())
                                    for term, hit in present
                                ]
                            )
                            if match_mode != "phrase"
                            else {}
                        ),
                    )
                )
            found += 1
            offset = end if end == len(text) else max(match.end(), end - len(query) + 1)
        if more:
            break
    complete = not bounded and not failures
    coverage = dict(
        total_documents=len(inventory),
        prepared_documents=prepared,
        pages_examined=min(pages, QUERY_MAX_PAGES),
        complete=complete,
        unsearched=not complete,
        work_limit=bounded,
        failed_documents_this_call=len({r["document_id"] for r in failures}),
        prior_pages=bool(after),
        prior_failures=False,
    )
    result = dict(results=results, failures=failures[:20], coverage=coverage)
    coverage["failures_truncated"] = len(failures) > 20
    if more:
        result["paging"] = dict(
            next=dict(
                after=HubSpotTools._encode_cursor(env.tools, skip + len(results), scope)
            )
        )
    return result
