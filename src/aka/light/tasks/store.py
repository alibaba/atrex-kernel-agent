"""Transactional task/events ledger; raw experiment artifacts remain separate files."""
from __future__ import annotations
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time
from .content import document


class Connection(sqlite3.Connection):
    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.close()


class Store:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, request_digest TEXT NOT NULL, manifest TEXT NOT NULL,
                state TEXT NOT NULL, control TEXT NOT NULL DEFAULT 'run',
                session TEXT NOT NULL DEFAULT '', episodes INTEGER NOT NULL DEFAULT 0,
                tokens INTEGER NOT NULL DEFAULT 0, stop_reason TEXT NOT NULL DEFAULT '',
                created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
                kind TEXT NOT NULL, data TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS evaluations (
                id TEXT PRIMARY KEY, task_id TEXT NOT NULL, candidate_id TEXT NOT NULL,
                revision TEXT NOT NULL, scope TEXT NOT NULL, role TEXT NOT NULL,
                state TEXT NOT NULL, plan TEXT NOT NULL, receipt TEXT, assessment TEXT,
                job_id TEXT NOT NULL DEFAULT '', created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS submissions (
                id TEXT PRIMARY KEY, task_id TEXT NOT NULL, candidate_id TEXT NOT NULL,
                evaluation_id TEXT NOT NULL, revision TEXT NOT NULL, created REAL NOT NULL,
                UNIQUE(task_id, evaluation_id));
            ''')

    def connect(self):
        db = sqlite3.connect(self.root / "tasks.sqlite3", timeout=30, factory=Connection)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        return db

    @contextmanager
    def transaction(self):
        db = self.connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def emit(db, task_id, kind, data):
        db.execute("INSERT INTO events(task_id,kind,data,created) VALUES(?,?,?,?)",
                   (task_id, kind, document(data), time.time()))

    def task(self, task_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown task")
        result = dict(row)
        result["manifest"] = json.loads(result["manifest"])
        return result

    def events(self, task_id, after=0):
        self.task(task_id)
        with self.connect() as db:
            return [dict(row, data=json.loads(row["data"])) for row in db.execute(
                "SELECT * FROM events WHERE task_id=? AND seq>? ORDER BY seq LIMIT 1000", (task_id, after))]

    def evaluation(self, evaluation_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM evaluations WHERE id=?", (evaluation_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown evaluation")
        value = dict(row)
        for key in ("plan", "receipt", "assessment"):
            value[key] = json.loads(value[key]) if value[key] else None
        return value

    def save_job(self, evaluation_id, job_id):
        with self.transaction() as db:
            db.execute("UPDATE evaluations SET job_id=?,state='running' WHERE id=?", (job_id, evaluation_id))

    def state(self, task_id, state, reason=""):
        with self.transaction() as db:
            db.execute("UPDATE tasks SET state=?,stop_reason=? WHERE id=?", (state, reason, task_id))
            self.emit(db, task_id, "state_changed", {"state": state, "reason": reason})
