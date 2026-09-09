"""Ingest sources into the knowledge base.

    python -m poe2craft.agent.ingest default              # curated wiki set + patch notes
    python -m poe2craft.agent.ingest wiki --category Omens --category Essences
    python -m poe2craft.agent.ingest wiki --page Omen --page "Exalted Orb"
    python -m poe2craft.agent.ingest patchnotes --pages 3  # first 3 forum listing pages
    python -m poe2craft.agent.ingest local notes/*.md
    python -m poe2craft.agent.ingest stats

Wiki content is CC BY-NC-SA (poe2wiki.net); patch notes (c) Grinding Gear Games.
Both are stored locally for personal use only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import html as htmllib
import os
import re
import sys
import time

import requests

from . import USER_AGENT
from .kb import KB, html_to_text

WIKI_API = "https://www.poe2wiki.net/api.php"
WIKI_PAGE = "https://www.poe2wiki.net/wiki/"
FORUM = "https://www.pathofexile.com/forum"
PATCH_FORUM_ID = 2212          # "Early Access Patch Notes"
DELAY = 0.35                   # seconds between requests (be polite)

# Curated categories: crafting + economy + mechanics + uniques + endgame.
DEFAULT_CATEGORIES = [
    "Currency items", "Omens", "Essences", "Runes", "Soul cores", "Crafting",
    "Game mechanics", "Combat mechanics", "Abyss", "Breach", "Expedition", "Delirium",
    "Ritual", "Item classes", "Lists of unique items", "Keystone passive skills",
    "Map areas", "Map boss unique monsters", "Versions",
]
DEFAULT_UNIQUE_SUBCATS = True   # every "Unique <slot>" subcategory of Category:Unique items
DEFAULT_PAGES = [
    "Crafting", "Modifier", "Item level", "Rarity", "Corruption", "Fracturing Orb",
    "Desecration", "Well of Souls", "Recombinator", "Reforging Bench", "Salvage Bench",
    "Atlas", "Waystone", "Endgame", "Pinnacle boss", "Tower", "Precursor Tablet",
    "Item filter", "Trade", "Currency Exchange", "Vendor recipe", "Quality",
    "Catalyst", "Alloy", "Flux", "Liquid Emotion", "Distilled Emotion", "Hinekora's Lock",
    "Socketable", "Rune", "Talisman", "Charm", "Jewel", "Time-Lost Jewel", "Relic",
    "Sanctum", "Trial of the Sekhemas", "Trial of Chaos", "Ascendancy class",
]


class Session:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = USER_AGENT
        self.last = 0.0

    def get(self, url: str, **kw) -> requests.Response:
        wait = DELAY - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(4):
            r = self.s.get(url, timeout=40, **kw)
            self.last = time.time()
            if r.status_code in (429, 502, 503, 504):
                time.sleep(2.0 * (attempt + 1))
                continue
            return r
        return r


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# wiki
# ---------------------------------------------------------------------------
def wiki_category_members(sess: Session, category: str, subcats: bool = False) -> list[str]:
    titles, cont = [], {}
    cmtype = "subcat" if subcats else "page"
    while True:
        params = dict(action="query", list="categorymembers", cmtitle=f"Category:{category}",
                      cmlimit=500, cmtype=cmtype, format="json", **cont)
        d = sess.get(WIKI_API, params=params).json()
        titles += [m["title"] for m in d.get("query", {}).get("categorymembers", [])]
        cont = d.get("continue")
        if not cont:
            break
    return titles


def wiki_fetch(sess: Session, title: str) -> tuple[str, str, str, str] | None:
    """-> (canonical title, url, text, revid) or None when the page is missing."""
    params = dict(action="parse", page=title, prop="text|revid|displaytitle",
                  redirects=1, disablelimitreport=1, format="json")
    for attempt in range(3):
        try:
            d = sess.get(WIKI_API, params=params).json()
            break
        except ValueError:                       # HTML error page instead of JSON: back off
            if attempt == 2:
                raise
            time.sleep(3.0 * (attempt + 1))
    if "error" in d:
        return None
    p = d["parse"]
    text = html_to_text(p["text"]["*"])
    canon = p["title"]
    url = WIKI_PAGE + canon.replace(" ", "_")
    return canon, url, text, str(p.get("revid", ""))


def wiki_revids(sess: Session, titles: list[str]) -> dict[str, str]:
    """Current revision id per title (redirects resolved), 50 titles per request."""
    out: dict[str, str] = {}
    for k in range(0, len(titles), 50):
        chunk = titles[k:k + 50]
        try:
            d = sess.get(WIKI_API, params=dict(action="query", prop="revisions", rvprop="ids",
                                               titles="|".join(chunk), redirects=1, format="json")).json()
        except ValueError:
            continue
        q = d.get("query", {})
        redirect = {r["from"]: r["to"] for r in q.get("redirects", [])}
        canon_rev = {}
        for page in q.get("pages", {}).values():
            revs = page.get("revisions")
            if revs:
                canon_rev[page["title"]] = str(revs[0]["revid"])
        for t in chunk:
            canon = redirect.get(t, t)
            if canon in canon_rev:
                out[t] = canon_rev[canon]
    return out


def ingest_wiki(kb: KB, titles: list[str], force: bool = False, verbose: bool = True) -> dict:
    sess = Session()
    titles = [t for t in dict.fromkeys(titles) if not t.startswith(("Category:", "File:"))]
    revs = {} if force else wiki_revids(sess, titles)
    n_new = n_skip = n_miss = 0
    todo = []
    for t in titles:
        rev = revs.get(t)
        url_guess = WIKI_PAGE + t.replace(" ", "_")
        if rev is not None and kb.has_version(url_guess, rev):
            n_skip += 1
            continue
        todo.append(t)
    if verbose:
        print(f"  {n_skip} pages unchanged, {len(todo)} to fetch", flush=True)
    for i, t in enumerate(todo, 1):
        try:
            res = wiki_fetch(sess, t)
        except Exception as e:                      # network hiccup: keep going
            print(f"  ! {t}: {e}", file=sys.stderr)
            n_miss += 1
            continue
        if res is None:
            n_miss += 1
            continue
        canon, url, text, revid = res
        if not force and kb.has_version(url, revid):
            n_skip += 1
            continue
        if len(text) < 80:
            n_miss += 1
            continue
        n = kb.upsert("wiki", url, canon, text, revid, _now())
        n_new += 1
        if verbose:
            print(f"  [{i}/{len(todo)}] {canon}: {len(text)} chars, {n} chunks", flush=True)
    return dict(new=n_new, unchanged=n_skip, missing=n_miss)


def default_wiki_titles(sess: Session) -> list[str]:
    titles = list(DEFAULT_PAGES)
    for c in DEFAULT_CATEGORIES:
        m = wiki_category_members(sess, c)
        print(f"  category {c}: {len(m)} pages", flush=True)
        titles += m
    if DEFAULT_UNIQUE_SUBCATS:
        for sub in wiki_category_members(sess, "Unique items", subcats=True):
            name = sub.removeprefix("Category:")
            if "legacy" in name.lower() or name in ("Lists of unique items", "Vaal unique items"):
                continue
            m = wiki_category_members(sess, name)
            print(f"  category {name}: {len(m)} pages", flush=True)
            titles += m
    # de-dup, keep order
    out, seen = [], set()
    for t in titles:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# patch notes (official forum, first post of each thread)
# ---------------------------------------------------------------------------
_THREAD_RE = re.compile(r'<div class="title">\s*<a href="/forum/view-thread/(\d+)"[^>]*>([^<]*)</a>')
_SKIP_TITLES = re.compile(r"server maintenance|restart\b|maintenance$|code of conduct", re.I)


def patch_threads(sess: Session, pages: int) -> list[tuple[str, str]]:
    out = []
    for p in range(1, pages + 1):
        r = sess.get(f"{FORUM}/view-forum/{PATCH_FORUM_ID}/page/{p}")
        rows = _THREAD_RE.findall(r.text)
        for tid, title in rows:
            title = htmllib.unescape(title).strip()
            if _SKIP_TITLES.search(title):
                continue
            out.append((tid, title))
        if not rows:
            break
    return out


def patch_thread_body(sess: Session, tid: str) -> str | None:
    r = sess.get(f"{FORUM}/view-thread/{tid}")
    m = re.search(r'<div class="content">(.*?)</div>\s*</td>', r.text, re.S)
    if not m:
        return None
    body = re.sub(r"<br\s*/?>", "\n", m.group(1))
    body = re.sub(r"</(p|li|div|h\d)>", "\n", body)
    body = re.sub(r"<li>", "- ", body)
    body = re.sub(r"<[^>]+>", "", body)
    body = htmllib.unescape(body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    date = re.search(r'class="post_date">([^<]*)', r.text)
    if date:
        body = f"Posted: {date.group(1).strip()}\n\n" + body
    return body


def ingest_patchnotes(kb: KB, pages: int = 3, force: bool = False, verbose: bool = True) -> dict:
    sess = Session()
    threads = patch_threads(sess, pages)
    n_new = n_skip = 0
    for i, (tid, title) in enumerate(threads, 1):
        url = f"{FORUM}/view-thread/{tid}"
        if not force and kb.has_version(url, tid):
            n_skip += 1
            continue
        body = patch_thread_body(sess, tid)
        if not body or len(body) < 40:
            continue
        # headings inside patch notes are plain lines in CAPS or bold; add the title as context
        text = f"{title}\n\n{body}"
        n = kb.upsert("patchnotes", url, f"Patch notes: {title}", text, tid, _now())
        n_new += 1
        if verbose:
            print(f"  [{i}/{len(threads)}] {title}: {len(text)} chars, {n} chunks", flush=True)
    return dict(new=n_new, unchanged=n_skip, threads=len(threads))


# ---------------------------------------------------------------------------
# local files (markdown / text)
# ---------------------------------------------------------------------------
def ingest_local(kb: KB, patterns: list[str], force: bool = False) -> dict:
    n_new = n_skip = 0
    for pat in patterns:
        for path in sorted(glob.glob(os.path.expanduser(pat))):
            if not os.path.isfile(path):
                continue
            version = str(int(os.path.getmtime(path)))
            url = "file://" + os.path.abspath(path)
            if not force and kb.has_version(url, version):
                n_skip += 1
                continue
            text = open(path, encoding="utf-8", errors="replace").read()
            if path.lower().endswith((".html", ".htm")):
                text = html_to_text(text)
            title = os.path.basename(path)
            m = re.search(r"^#\s+(.+)$", text, re.M)
            if m:
                title = m.group(1).strip()
            n = kb.upsert("local", url, title, text, version, _now())
            print(f"  {path}: {len(text)} chars, {n} chunks")
            n_new += 1
    return dict(new=n_new, unchanged=n_skip)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("wiki")
    w.add_argument("--category", action="append", default=[])
    w.add_argument("--page", action="append", default=[])
    w.add_argument("--force", action="store_true")
    p = sub.add_parser("patchnotes")
    p.add_argument("--pages", type=int, default=3)
    p.add_argument("--force", action="store_true")
    l = sub.add_parser("local")
    l.add_argument("paths", nargs="+")
    l.add_argument("--force", action="store_true")
    d = sub.add_parser("default")
    d.add_argument("--force", action="store_true")
    d.add_argument("--patch-pages", type=int, default=6)
    sub.add_parser("stats")
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    a = ap.parse_args(argv)

    kb = KB()
    t0 = time.time()
    if a.cmd == "wiki":
        sess = Session()
        titles = list(a.page)
        for c in a.category:
            titles += wiki_category_members(sess, c)
        print(ingest_wiki(kb, titles, force=a.force))
    elif a.cmd == "patchnotes":
        print(ingest_patchnotes(kb, pages=a.pages, force=a.force))
    elif a.cmd == "local":
        print(ingest_local(kb, a.paths, force=a.force))
    elif a.cmd == "default":
        sess = Session()
        titles = default_wiki_titles(sess)
        print(f"{len(titles)} wiki titles to fetch", flush=True)
        print("wiki:", ingest_wiki(kb, titles, force=a.force))
        print("patchnotes:", ingest_patchnotes(kb, pages=a.patch_pages, force=a.force))
    elif a.cmd == "stats":
        print(kb.stats())
    elif a.cmd == "search":
        from .kb import format_hits
        print(format_hits(kb.search(a.query, k=a.k)))
    print(f"done in {time.time() - t0:.0f}s; kb: {kb.stats()}")


if __name__ == "__main__":
    main()
