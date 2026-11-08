"""Atomic snapshot copies with writes limited to changed native rows."""

from pathlib import Path
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from itertools import islice


def _clone_readonly_database(source, target):
    # Immutable native snapshots have no active journal. APFS clones keep an
    # independent inode and copy only changed pages when the new day is saved.
    if (
        sys.platform != "darwin"
        or source.stat().st_mode & 0o222
        or any(Path(str(source) + suffix).exists() for suffix in ("-wal", "-journal"))
    ):
        return False
    try:
        subprocess.run(
            ["/bin/cp", "-c", str(source), str(target)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        target.chmod(0o600)
        return True
    except (OSError, subprocess.CalledProcessError):
        target.unlink(missing_ok=True)
        return False


def start_database(path, schema, *, incremental=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".building")
    temporary.unlink(missing_ok=True)
    cloned = (
        incremental and path.is_file() and _clone_readonly_database(path, temporary)
    )
    connection = sqlite3.connect(temporary)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA cache_size=-16384")
    connection.execute("PRAGMA temp_store=FILE")
    if incremental and path.is_file() and not cloned:
        with closing(
            sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        ) as previous:
            previous.backup(connection)
    elif not cloned:
        connection.executescript(schema)
    return connection


def sync_rows(connection, table, columns, values, *, delete_missing=True):
    """Compare rows in SQLite with bounded Python batches, preserving omissions on request."""
    columns = tuple(columns)
    info = list(connection.execute(f'PRAGMA table_info("{table}")'))
    key_names = [row[1] for row in sorted(info, key=lambda row: row[5]) if row[5]]
    if not key_names:
        raise ValueError("incremental table requires a primary key: " + table)
    key_indexes = [columns.index(name) for name in key_names]
    names = ",".join('"' + name + '"' for name in columns)
    placeholders = ",".join("?" for _ in columns)
    insert = f'INSERT INTO "{table}" ({names}) VALUES ({placeholders})'
    if connection.execute(f'SELECT 1 FROM "{table}" LIMIT 1').fetchone() is None:
        before = connection.total_changes
        try:
            connection.executemany(insert, (tuple(value) for value in values))
        except sqlite3.IntegrityError as error:
            if error.sqlite_errorcode in {
                sqlite3.SQLITE_CONSTRAINT_PRIMARYKEY,
                sqlite3.SQLITE_CONSTRAINT_UNIQUE,
            }:
                raise ValueError("duplicate native row in " + table) from error
            raise
        return connection.total_changes - before
    keys = ",".join('"' + name + '"' for name in key_names)
    # Store only incoming identities on disk. The previous implementation kept
    # every saved row and all decoded field strings in a Python dictionary.
    connection.execute(
        f'CREATE TEMP TABLE _native_keys AS SELECT {keys} FROM "{table}" WHERE 0'
    )
    connection.execute(
        f"CREATE UNIQUE INDEX _native_keys_unique ON _native_keys ({keys})"
    )
    key_insert = (
        "INSERT INTO _native_keys VALUES (" + ",".join("?" for _ in key_names) + ")"
    )
    changed = [name for name in columns if name not in key_names]
    assignments = ",".join(f'"{name}"=excluded."{name}"' for name in changed)
    different = " OR ".join(
        f'"{table}"."{name}" IS NOT excluded."{name}"' for name in changed
    )
    conflict = (
        f"DO UPDATE SET {assignments} WHERE {different}" if changed else "DO NOTHING"
    )
    changes = 0
    iterator = iter(values)
    try:
        while batch := list(islice(iterator, 1000)):
            batch = [tuple(value) for value in batch]
            try:
                connection.executemany(
                    key_insert,
                    (tuple(value[index] for index in key_indexes) for value in batch),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError("duplicate native row in " + table) from error
            before = connection.total_changes
            connection.executemany(insert + " ON CONFLICT " + conflict, batch)
            changes += connection.total_changes - before
        if delete_missing:
            same_key = " AND ".join(
                f'_native_keys."{name}" IS "{table}"."{name}"' for name in key_names
            )
            changes += connection.execute(
                f'DELETE FROM "{table}" WHERE NOT EXISTS (SELECT 1 FROM _native_keys WHERE {same_key})'
            ).rowcount
    finally:
        connection.execute("DROP TABLE _native_keys")
    return changes


def finish_database(connection, path):
    temporary = Path(connection.execute("PRAGMA database_list").fetchone()[2])
    try:
        connection.commit()
    finally:
        connection.close()
    os.replace(temporary, path)
