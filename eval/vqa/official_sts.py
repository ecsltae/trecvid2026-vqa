"""
Client for the OFFICIAL TRECVID VQA semantic-similarity metric.

The official AG score is NOT sentence-transformers cosine. It is the UMBC
Semantic Textual Similarity web service (Han et al., *SEM 2013), still live at
swoogle.umbc.edu. Its behaviour is nothing like an embedding model — it is
brutally sensitive to surface form:

    ref "he fell down"  vs  "he fell"                              -> 1.000
    ref "he fell down"  vs  "In the video, the man appears to fall down" -> 0.059

Four words of preamble cost ~0.94. Any model selection done with MiniLM cosine
is therefore selecting against a number NIST will never compute.

This module is deliberately conservative about the remote service: it is a
small 2013 academic endpoint, not a commercial API. Every score is cached
permanently in SQLite, requests are rate-limited and serialized through a
token bucket, and failures back off exponentially. Re-scoring an already-seen
pair costs zero network.

Usage:
    from official_sts import OfficialSTS
    sts = OfficialSTS()
    sts.score("he fell", "he fell down")          # -> 1.0
    sts.score_many([(p1, p2), ...])               # -> [float, ...]
    print(sts.stats())                            # cache hits / calls / errors
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_ENDPOINT = "http://swoogle.umbc.edu/StsService/GetStsSim"
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "vqa" / "sts_cache.sqlite"

# The service is a small academic host. Default to a polite ~1.5 req/s.
DEFAULT_RATE = float(os.environ.get("UMBC_STS_RATE", "1.5"))


class RateLimiter:
    """Token bucket shared across threads."""

    def __init__(self, rate_per_sec):
        self.min_interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.0
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_allowed = max(now, self._next_allowed) + self.min_interval


class OfficialSTS:
    def __init__(self, cache_path=DEFAULT_CACHE, rate=DEFAULT_RATE, endpoint=DEFAULT_ENDPOINT,
                 timeout=30, retries=4, offline=False):
        """
        Args:
            offline: if True, never hit the network — return cached values only
                     and raise KeyError on a miss. Use for reproducible reruns.
        """
        self.endpoint = endpoint
        self.timeout = timeout
        self.retries = retries
        self.offline = offline
        self.limiter = RateLimiter(rate)
        self._db_lock = threading.Lock()
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(cache_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS sts ("
            "  p1 TEXT NOT NULL, p2 TEXT NOT NULL, score REAL NOT NULL,"
            "  PRIMARY KEY (p1, p2))"
        )
        self.conn.commit()
        self.n_hits = self.n_calls = self.n_errors = 0

    # ---------------- cache ----------------

    @staticmethod
    def _norm(text):
        """Whitespace-normalize only. Case and punctuation are NOT touched —
        the service is sensitive to them, so they are part of the key."""
        return " ".join(str(text or "").split())

    def _cache_get(self, p1, p2):
        with self._db_lock:
            row = self.conn.execute(
                "SELECT score FROM sts WHERE p1=? AND p2=?", (p1, p2)).fetchone()
        return row[0] if row else None

    def _cache_put(self, p1, p2, score):
        with self._db_lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO sts (p1, p2, score) VALUES (?,?,?)", (p1, p2, score))
            self.conn.commit()

    # ---------------- remote ----------------

    def _fetch(self, p1, p2):
        qs = urllib.parse.urlencode({"operation": "api", "phrase1": p1, "phrase2": p2})
        url = f"{self.endpoint}?{qs}"
        last_err = None
        for attempt in range(self.retries):
            self.limiter.acquire()
            try:
                with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                    body = resp.read().decode("utf-8", errors="replace").strip()
                self.n_calls += 1
                try:
                    return float(body)
                except ValueError:
                    # Service returns an HTML error page under load.
                    last_err = RuntimeError(f"non-numeric response: {body[:120]!r}")
            except Exception as e:  # network, timeout, HTTP error
                last_err = e
            time.sleep(min(2.0 * (2 ** attempt), 30.0))
        self.n_errors += 1
        raise RuntimeError(f"UMBC STS failed after {self.retries} attempts: {last_err}")

    # ---------------- public ----------------

    def score(self, phrase1, phrase2):
        """Similarity in [0,1]. Empty input short-circuits to 0.0 (no call)."""
        p1, p2 = self._norm(phrase1), self._norm(phrase2)
        if not p1 or not p2:
            return 0.0
        cached = self._cache_get(p1, p2)
        if cached is not None:
            self.n_hits += 1
            return cached
        if self.offline:
            raise KeyError(f"offline mode, pair not cached: {p1[:50]!r} / {p2[:50]!r}")
        score = self._fetch(p1, p2)
        self._cache_put(p1, p2, score)
        return score

    def score_many(self, pairs, workers=3, progress_every=0):
        """Score a list of (phrase1, phrase2). Cached pairs cost nothing;
        uncached ones are fetched concurrently but still globally rate-limited."""
        from concurrent.futures import ThreadPoolExecutor

        results = [None] * len(pairs)
        todo = []
        for i, (a, b) in enumerate(pairs):
            p1, p2 = self._norm(a), self._norm(b)
            if not p1 or not p2:
                results[i] = 0.0
                continue
            cached = self._cache_get(p1, p2)
            if cached is not None:
                self.n_hits += 1
                results[i] = cached
            else:
                todo.append((i, p1, p2))

        if todo and self.offline:
            raise KeyError(f"offline mode: {len(todo)} pairs not cached")

        done = 0
        if todo:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(self._fetch, p1, p2): (i, p1, p2) for i, p1, p2 in todo}
                for fut in futs:
                    pass  # submitted
                from concurrent.futures import as_completed
                for fut in as_completed(futs):
                    i, p1, p2 = futs[fut]
                    try:
                        score = fut.result()
                        self._cache_put(p1, p2, score)
                    except Exception:
                        score = float("nan")
                    results[i] = score
                    done += 1
                    if progress_every and done % progress_every == 0:
                        print(f"    STS {done}/{len(todo)} fetched "
                              f"(cache hits {self.n_hits}, errors {self.n_errors})", flush=True)
        return results

    def cache_size(self):
        with self._db_lock:
            return self.conn.execute("SELECT COUNT(*) FROM sts").fetchone()[0]

    def stats(self):
        return {"cache_hits": self.n_hits, "remote_calls": self.n_calls,
                "errors": self.n_errors, "cache_rows": self.cache_size()}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Probe the official UMBC STS service")
    ap.add_argument("phrase1")
    ap.add_argument("phrase2")
    args = ap.parse_args()
    s = OfficialSTS()
    print(s.score(args.phrase1, args.phrase2))
    print(s.stats())
