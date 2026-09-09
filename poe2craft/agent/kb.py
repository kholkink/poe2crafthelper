"""Knowledge base: SQLite FTS5 (BM25) over chunked documents.

No embeddings, no external services: the corpus (wiki pages, patch notes, local
notes) is small and English, and lexical search with field weighting is good
enough for a first version.  The model turns the user's question into a short
English keyword query.
"""
from __future__ import annotations

import html as htmllib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Iterable

from . import KB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,          -- 'wiki' | 'patchnotes' | 'local'
    url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    version TEXT,                  -- wiki revid / thread id / mtime: skip re-ingest when unchanged
    fetched_at TEXT NOT NULL,
    n_chars INTEGER NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
    title, section, body,
    doc_id UNINDEXED, url UNINDEXED, source UNINDEXED, ord UNINDEXED,
    tokenize = 'porter unicode61 remove_diacritics 2'
);
"""

STOP = set("""a an the of to in on for and or is are be with by as at from this that
these those it its into can will what which how do does did when where why who
your you i me my we our they their than then there here also""".split())


# ---------------------------------------------------------------------------
# HTML -> text (keeps table rows as ' | ' separated lines)
# ---------------------------------------------------------------------------
_SKIP_CLASSES = {"mw-editsection", "toc", "navbox", "mw-references-wrap", "printfooter",
                 "catlinks", "mw-jump-link", "noprint", "reference", "mw-indicators",
                 "infobox-icon", "hidden",
                 "c-item-hoverbox__display", "hoverbox__display", "c-skill-hoverbox__display"}
_BLOCK = {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "br", "table",
          "ul", "ol", "section", "blockquote", "pre", "dd", "dt"}


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0          # >0 while inside a skipped subtree
        self.stack: list[bool] = []  # per open tag: did it start a skip?
        self.headings: list[tuple[int, int]] = []   # (position in parts, level)
        self.in_heading = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = set((a.get("class") or "").split())
        skip = tag in ("script", "style") or bool(cls & _SKIP_CLASSES)
        self.stack.append(skip)
        if skip:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in ("td", "th"):
            self.parts.append(" | ")
        elif tag in ("h2", "h3", "h4"):
            self.parts.append("\n\n")
            self.headings.append((len(self.parts), int(tag[1])))
            self.in_heading = int(tag[1])
            self.parts.append(f"{'#' * int(tag[1])} ")
        elif tag in _BLOCK:
            self.parts.append("\n")
            if tag == "li":
                self.parts.append("- ")

    def handle_endtag(self, tag):
        if self.stack:
            skip = self.stack.pop()
            if skip:
                self.skip_depth -= 1
                return
        if self.skip_depth:
            return
        if tag in ("h2", "h3", "h4"):
            self.parts.append("\n")
            self.in_heading = 0
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip_depth:
            return
        self.parts.append(data)

    def text(self) -> str:
        t = "".join(self.parts)
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r" ?\| ?", " | ", t)
        t = re.sub(r"\n(?: \| )+", "\n", t)          # leading cell separator
        t = re.sub(r"(?: \| )+\n", "\n", t)          # trailing cell separator
        t = re.sub(r"\[ ?edit ?\]", "", t)
        t = re.sub(r"\n\s*\n\s*\n+", "\n\n", t)
        return t.strip()


def html_to_text(raw_html: str) -> str:
    p = _Text()
    p.feed(raw_html)
    p.close()
    return p.text()


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    section: str
    body: str


def chunk_text(text: str, max_chars: int = 1400, min_chars: int = 200) -> list[Chunk]:
    """Split on markdown-style headings, then pack paragraphs up to max_chars."""
    sections: list[tuple[str, str]] = []
    cur_title, buf = "", []
    for line in text.split("\n"):
        m = re.match(r"^(#{2,4}) (.+)$", line)
        if m:
            if buf:
                sections.append((cur_title, "\n".join(buf)))
            cur_title, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    if buf:
        sections.append((cur_title, "\n".join(buf)))

    out: list[Chunk] = []
    for title, body in sections:
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        acc: list[str] = []
        size = 0
        for p in paras:
            # a single huge paragraph (e.g. a table) is split by lines
            pieces = [p] if len(p) <= max_chars else _split_lines(p, max_chars)
            for piece in pieces:
                if size + len(piece) > max_chars and acc:
                    out.append(Chunk(title, "\n\n".join(acc)))
                    acc, size = [], 0
                acc.append(piece)
                size += len(piece) + 2
        if acc:
            out.append(Chunk(title, "\n\n".join(acc)))
    # merge tiny trailing chunks into their predecessor
    merged: list[Chunk] = []
    for c in out:
        if merged and len(c.body) < min_chars and merged[-1].section == c.section:
            merged[-1] = Chunk(c.section, merged[-1].body + "\n\n" + c.body)
        else:
            merged.append(c)
    return [c for c in merged if len(c.body.strip()) >= 40]


def _split_lines(p: str, max_chars: int) -> list[str]:
    out, acc, size = [], [], 0
    for line in p.split("\n"):
        if size + len(line) > max_chars and acc:
            out.append("\n".join(acc))
            acc, size = [], 0
        acc.append(line)
        size += len(line) + 1
    if acc:
        out.append("\n".join(acc))
    return out


# ---------------------------------------------------------------------------
# storage + search
# ---------------------------------------------------------------------------
class KB:
    def __init__(self, path: str = KB_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.con = sqlite3.connect(path)
        self.con.executescript(SCHEMA)

    # ---- writes ----------------------------------------------------------
    def has_version(self, url: str, version: str | None) -> bool:
        row = self.con.execute("SELECT version FROM docs WHERE url=?", (url,)).fetchone()
        return bool(row) and version is not None and row[0] == version

    def upsert(self, source: str, url: str, title: str, text: str, version: str | None,
               fetched_at: str) -> int:
        chunks = chunk_text(text)
        cur = self.con.cursor()
        row = cur.execute("SELECT id FROM docs WHERE url=?", (url,)).fetchone()
        if row:
            doc_id = row[0]
            cur.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            cur.execute("UPDATE docs SET title=?, version=?, fetched_at=?, n_chars=? WHERE id=?",
                        (title, version, fetched_at, len(text), doc_id))
        else:
            cur.execute("INSERT INTO docs(source,url,title,version,fetched_at,n_chars) VALUES (?,?,?,?,?,?)",
                        (source, url, title, version, fetched_at, len(text)))
            doc_id = cur.lastrowid
        cur.executemany(
            "INSERT INTO chunks(title,section,body,doc_id,url,source,ord) VALUES (?,?,?,?,?,?,?)",
            [(title, c.section, c.body, doc_id, url, source, i) for i, c in enumerate(chunks)])
        self.con.commit()
        return len(chunks)

    def delete_source(self, source: str) -> None:
        self.con.execute("DELETE FROM chunks WHERE source=?", (source,))
        self.con.execute("DELETE FROM docs WHERE source=?", (source,))
        self.con.commit()

    # ---- reads -----------------------------------------------------------
    def stats(self) -> dict:
        rows = self.con.execute(
            "SELECT source, COUNT(*), SUM(n_chars) FROM docs GROUP BY source").fetchall()
        n_chunks = self.con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"docs": {r[0]: {"count": r[1], "chars": r[2]} for r in rows},
                "chunks": n_chunks, "path": self.path}

    @staticmethod
    def _terms(query: str) -> list[str]:
        toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9'%+-]*", query)
        toks = [t.strip("'-+") for t in toks]
        return [t for t in toks if t and t.lower() not in STOP and len(t) > 1]

    def search(self, query: str, k: int = 6, source: str | None = None) -> list[dict]:
        """BM25 search: all terms first, then any-term fallback. Phrases in quotes kept."""
        phrases = re.findall(r'"([^"]+)"', query)
        rest = re.sub(r'"[^"]+"', " ", query)
        terms = self._terms(rest)
        if not terms and not phrases:
            return []
        q_parts = [f'"{p}"' for p in phrases] + [f'"{t}"' for t in terms]
        results = self._run(" AND ".join(q_parts), k, source)
        if len(results) < k and len(q_parts) > 1:
            seen = {r["url"] + str(r["ord"]) for r in results}
            for r in self._run(" OR ".join(q_parts), k * 2, source):
                if r["url"] + str(r["ord"]) not in seen:
                    results.append(r)
                    seen.add(r["url"] + str(r["ord"]))
                if len(results) >= k:
                    break
        return results[:k]

    def _run(self, match: str, k: int, source: str | None) -> list[dict]:
        sql = ("SELECT title, section, body, url, source, ord, bm25(chunks, 6.0, 3.0, 1.0) AS score "
               "FROM chunks WHERE chunks MATCH ?")
        args: list = [match]
        if source:
            sql += " AND source=?"
            args.append(source)
        sql += " ORDER BY score LIMIT ?"
        args.append(k)
        try:
            rows = self.con.execute(sql, args).fetchall()
        except sqlite3.OperationalError:
            return []
        return [dict(title=r[0], section=r[1], body=r[2], url=r[3], source=r[4], ord=r[5],
                     score=round(-r[6], 3)) for r in rows]

    def get_doc(self, title_or_url: str, max_chars: int = 12000) -> dict | None:
        row = self.con.execute(
            "SELECT id, title, url, source FROM docs WHERE url=? OR lower(title)=lower(?) LIMIT 1",
            (title_or_url, title_or_url)).fetchone()
        if not row:
            row = self.con.execute(
                "SELECT id, title, url, source FROM docs WHERE lower(title) LIKE lower(?) "
                "ORDER BY length(title) LIMIT 1", (f"%{title_or_url}%",)).fetchone()
        if not row:
            return None
        parts = self.con.execute(
            "SELECT section, body FROM chunks WHERE doc_id=? ORDER BY ord", (row[0],)).fetchall()
        text = "\n\n".join((f"## {s}\n{b}" if s else b) for s, b in parts)
        truncated = len(text) > max_chars
        return {"title": row[1], "url": row[2], "source": row[3],
                "text": text[:max_chars], "truncated": truncated}


def format_hits(hits: Iterable[dict], max_body: int = 1200) -> str:
    out = []
    for i, h in enumerate(hits, 1):
        sec = f" > {h['section']}" if h["section"] else ""
        body = h["body"] if len(h["body"]) <= max_body else h["body"][:max_body] + " …"
        out.append(f"[{i}] {h['title']}{sec}  ({h['source']}: {h['url']})\n{body}")
    return "\n\n".join(out) if out else "No results."
