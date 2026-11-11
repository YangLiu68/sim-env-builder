"""Poly Pizza access: official API when a key is configured, the public website otherwise.

Keyless mode reads the same public pages a browser does (``/search/<q>`` and ``/m/<id>``), which
embed their data as ``window.__SERVER_APP_STATE__`` JSON. It honours robots.txt (``Crawl-delay:
10`` for poly.pizza) with a lock file shared by every server process on the machine, and caches
pages on disk so repeated searches cost nothing. GLB files come from the static CDN
(``static.poly.pizza/<uuid>.glb``) and are cached by UUID.

API mode uses ``https://api.poly.pizza/v1.1`` with the ``x-auth-token`` header. The key comes from
``POLY_PIZZA_API_KEY`` or ``~/.poly-pizza/config.json`` (``{"apiKey": "..."}``). Any API error
falls back to keyless mode.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

SITE = "https://poly.pizza"
API = "https://api.poly.pizza/v1.1"
STATIC = "https://static.poly.pizza"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0 Safari/537.36 polypizza-mcp/0.1"
)
CACHE_DIR = Path(os.environ.get("POLYPIZZA_CACHE") or "~/.cache/polypizza-mcp").expanduser()
SITE_DELAY_S = float(os.environ.get("POLYPIZZA_SITE_DELAY", "10"))  # robots.txt Crawl-delay
STATIC_DELAY_S = 0.3
SEARCH_TTL_S = 7 * 24 * 3600
MODEL_TTL_S = 30 * 24 * 3600
LICENCE_URLS = {
    "CC0 1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC-BY 3.0": "https://creativecommons.org/licenses/by/3.0/",
    "CC-BY 4.0": "https://creativecommons.org/licenses/by/4.0/",
}
_STATE_RE = re.compile(r"window\.__SERVER_APP_STATE__\s*=\s*(\{.*?\})\s*</script>", re.S)
_UUID_RE = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,20}$")


class PolyPizzaError(RuntimeError):
    pass


@dataclass
class ModelInfo:
    id: str
    title: str
    creator: str
    licence: str
    glb_url: str
    preview_url: str
    page_url: str
    resource_id: str = ""
    tris: int | None = None
    category: str | None = None
    tags: list[str] = field(default_factory=list)
    description: str | None = None
    animated: bool | None = None

    @property
    def attribution(self) -> str:
        lic = self.licence or "unknown licence"
        lic_url = LICENCE_URLS.get(lic, "")
        lic_txt = f"{lic} {lic_url}".strip()
        return f'"{self.title}" by {self.creator} ({lic_txt}) via Poly Pizza {self.page_url}'

    def to_dict(self) -> dict:
        d = asdict(self)
        d["attribution"] = self.attribution
        d["requires_attribution"] = not (self.licence or "").upper().startswith("CC0")
        return d


# ----------------------------------------------------------------------------- key + cache
def api_key() -> str | None:
    key = os.environ.get("POLY_PIZZA_API_KEY") or os.environ.get("POLYPIZZA_API_KEY")
    if key:
        return key.strip()
    cfg = Path("~/.poly-pizza/config.json").expanduser()
    if cfg.exists():
        try:
            return (json.loads(cfg.read_text()).get("apiKey") or "").strip() or None
        except (OSError, ValueError):
            return None
    return None


def mode() -> str:
    return "api" if api_key() else "keyless"


def _cache_path(kind: str, key: str, ext: str = "json") -> Path:
    h = hashlib.sha1(key.encode()).hexdigest()[:20]
    return CACHE_DIR / kind / f"{h}.{ext}"


def _cache_get(kind: str, key: str, ttl: float):
    p = _cache_path(kind, key)
    if p.exists() and time.time() - p.stat().st_mtime < ttl:
        try:
            return json.loads(p.read_text())
        except ValueError:
            return None
    return None


def _cache_put(kind: str, key: str, value) -> None:
    p = _cache_path(kind, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(value))
    os.replace(tmp, p)


# ----------------------------------------------------------------------------- polite HTTP
@contextlib.contextmanager
def _host_slot(host: str, delay: float):
    """Wait until ``delay`` seconds have passed since the last request to ``host`` by ANY process."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lock = CACHE_DIR / f"rate_{host.replace('.', '_')}.lock"
    with open(lock, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0)
            raw = fh.read().strip()
            last = float(raw) if raw else 0.0
            wait = last + delay - time.time()
            if wait > 0:
                time.sleep(wait)
            yield
        finally:
            fh.seek(0)
            fh.truncate()
            fh.write(str(time.time()))
            fh.flush()
            fcntl.flock(fh, fcntl.LOCK_UN)


def _get(url: str, headers: dict | None = None, timeout: float = 45.0, retries: int = 2) -> bytes:
    host = urllib.parse.urlparse(url).netloc
    delay = SITE_DELAY_S if host == "poly.pizza" else STATIC_DELAY_S if host.startswith("static.") else 0.5
    hdrs = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    hdrs.update(headers or {})
    err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with _host_slot(host, delay):
                req = urllib.request.Request(url, headers=hdrs)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.read()
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise PolyPizzaError(f"HTTP {e.code} for {url}") from e
            err = e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            err = e
        time.sleep(2.0 * (attempt + 1))
    raise PolyPizzaError(f"request failed for {url}: {err}")


def _page_state(html: str) -> dict:
    m = _STATE_RE.search(html)
    if not m:
        raise PolyPizzaError("page layout changed: no __SERVER_APP_STATE__ found")
    return json.loads(m.group(1))


def _uuid_from(url: str) -> str:
    m = _UUID_RE.search(url or "")
    return m.group(1) if m else ""


# ----------------------------------------------------------------------------- search
def _from_site_card(c: dict) -> ModelInfo:
    pid = c.get("publicID") or (c.get("url") or "").rsplit("/", 1)[-1]
    uuid = _uuid_from(c.get("previewUrl", ""))
    creator = c.get("creator") or {}
    return ModelInfo(
        id=pid,
        title=c.get("title") or c.get("alt") or "",
        creator=creator.get("username", "") if isinstance(creator, dict) else str(creator),
        licence=c.get("licence") or "",
        glb_url=f"{STATIC}/{uuid}.glb" if uuid else "",
        preview_url=c.get("previewUrl") or "",
        page_url=f"{SITE}/m/{pid}",
        resource_id=uuid,
    )


def _from_api(r: dict) -> ModelInfo:
    pid = str(r.get("ID") or r.get("PublicID") or r.get("id") or "")
    creator = r.get("Creator") or {}
    download = r.get("Download") or r.get("download") or ""
    thumb = r.get("Thumbnail") or r.get("thumbnail") or ""
    uuid = _uuid_from(download) or _uuid_from(thumb)
    tris = r.get("Tri Count", r.get("Tris", r.get("TriCount")))
    return ModelInfo(
        id=pid,
        title=r.get("Title", ""),
        creator=creator.get("Username", "") if isinstance(creator, dict) else str(creator),
        licence=r.get("Licence") or r.get("License") or "",
        glb_url=download or (f"{STATIC}/{uuid}.glb" if uuid else ""),
        preview_url=thumb,
        page_url=f"{SITE}/m/{pid}",
        resource_id=uuid,
        tris=int(tris) if isinstance(tris, (int, float)) else None,
        category=r.get("Category"),
        tags=list(r.get("Tags") or []),
        description=r.get("Description"),
        animated=r.get("Animated"),
    )


def search(query: str, limit: int = 20) -> list[ModelInfo]:
    """Models matching ``query`` (site relevance order). Cached for a week."""
    q = query.strip()
    if not q:
        return []
    key = f"search:{q.lower()}"
    cached = _cache_get("search", key, SEARCH_TTL_S)
    if cached is not None:
        return [ModelInfo(**c) for c in cached][:limit]
    results: list[ModelInfo] = []
    k = api_key()
    if k:
        try:
            raw = _get(f"{API}/search/{urllib.parse.quote(q)}?Limit=32", {"x-auth-token": k})
            results = [_from_api(r) for r in json.loads(raw).get("results", [])]
        except (PolyPizzaError, ValueError):
            results = []
    if not results:
        html = _get(f"{SITE}/search/{urllib.parse.quote(q)}").decode("utf-8", "replace")
        state = _page_state(html).get("initialData", {})
        cards = state.get("result") or []
        if not cards:
            for chunk in state.get("firstChunk") or []:
                cards.extend(chunk if isinstance(chunk, list) else [chunk])
        results = [_from_site_card(c) for c in cards if isinstance(c, dict)]
        results = [r for r in results if r.id and r.glb_url]
    _cache_put("search", key, [asdict(r) for r in results])
    for r in results:  # index by id so previews/downloads need no extra page request
        if _cache_get("index", r.id, MODEL_TTL_S) is None:
            _cache_put("index", r.id, asdict(r))
    return results[:limit]


def lookup(model_id: str) -> ModelInfo:
    """Metadata from the search index if this id appeared in a search, else ``get_model``."""
    model_id = model_id.strip().rsplit("/", 1)[-1]
    full = _cache_get("model", f"model:{model_id}", MODEL_TTL_S)
    if full is not None:
        return ModelInfo(**full)
    hit = _cache_get("index", model_id, MODEL_TTL_S)
    if hit is not None and hit.get("glb_url") and hit.get("licence"):
        return ModelInfo(**hit)
    return get_model(model_id)


def get_model(model_id: str) -> ModelInfo:
    """Full metadata (tris, tags, description...) for one model. Cached for a month."""
    model_id = model_id.strip().rsplit("/", 1)[-1]
    if not _ID_RE.match(model_id):
        raise PolyPizzaError(f"not a Poly Pizza model id: {model_id!r}")
    key = f"model:{model_id}"
    cached = _cache_get("model", key, MODEL_TTL_S)
    if cached is not None:
        return ModelInfo(**cached)
    info: ModelInfo | None = None
    k = api_key()
    if k:
        try:
            raw = json.loads(_get(f"{API}/model/{model_id}", {"x-auth-token": k}))
            info = _from_api(raw)
        except (PolyPizzaError, ValueError):
            info = None
    if info is None:
        html = _get(f"{SITE}/m/{model_id}").decode("utf-8", "replace")
        m = (_page_state(html).get("initialData") or {}).get("model")
        if not m:
            raise PolyPizzaError(f"model {model_id} not found")
        creator = m.get("Creator") or {}
        uuid = m.get("ResourceID") or ""
        info = ModelInfo(
            id=m.get("PublicID") or model_id,
            title=m.get("Title", ""),
            creator=creator.get("Username", "") if isinstance(creator, dict) else str(creator),
            licence=m.get("Licence", ""),
            glb_url=f"{STATIC}/{uuid}.glb" if uuid else "",
            preview_url=f"{STATIC}/{uuid}.webp" if uuid else "",
            page_url=f"{SITE}/m/{model_id}",
            resource_id=uuid,
            tris=m.get("Tris"),
            category=m.get("Category"),
            tags=list(m.get("Tags") or []),
            description=m.get("Description"),
            animated=m.get("Animated"),
        )
    _cache_put("model", key, asdict(info))
    return info


def fetch_bytes_cached(url: str, kind: str) -> bytes:
    """Static file (GLB / preview image) cached by URL."""
    ext = url.rsplit(".", 1)[-1][:5] if "." in url.rsplit("/", 1)[-1] else "bin"
    p = _cache_path(kind, url, ext)
    if p.exists() and p.stat().st_size > 0:
        return p.read_bytes()
    data = _get(url, timeout=120.0)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, p)
    return data


def download_glb(info: ModelInfo) -> bytes:
    if not info.glb_url:
        raise PolyPizzaError(f"model {info.id} has no GLB url")
    data = fetch_bytes_cached(info.glb_url, "glb")
    if data[:4] != b"glTF":
        raise PolyPizzaError(f"{info.glb_url} did not return a GLB file")
    return data


# ----------------------------------------------------------------------------- relevance
_STOP = {"a", "an", "the", "of", "with", "and", "for", "low", "poly", "lowpoly", "model", "3d"}


def _tokens(s: str) -> set[str]:
    toks = {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _STOP}
    out = set(toks)
    for t in toks:  # crude singularisation so "chairs" matches "chair"
        if len(t) > 3 and t.endswith("es"):
            out.add(t[:-2])
        if len(t) > 3 and t.endswith("s"):
            out.add(t[:-1])
    return out


def match_quality(query: str, info: ModelInfo) -> str:
    """``exact`` (all query words in title/tags), ``partial`` (some), or ``none``."""
    q = _tokens(query)
    if not q:
        return "none"
    have = _tokens(info.title) | _tokens(" ".join(info.tags)) | _tokens(info.category or "")
    hit = len(q & have)
    if hit == len(q):
        return "exact"
    return "partial" if hit else "none"
