"""
Universal ingestion agent.

Turns ANY input (uploaded file or API response) into a pandas DataFrame:
CSV, TSV, Excel, JSON, Parquet, XML, PDF, TXT, ZIP (loads the table inside), or unknown text formats.
"""

import io
import zipfile
import json
import os
from urllib.parse import urlparse

import pandas as pd
import requests


def _from_json(payload):
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            return pd.json_normalize(payload)
        return pd.DataFrame({"value": payload})

    if isinstance(payload, dict):
        # records wrapped under a common key, or any key holding a list of dicts
        for key in ("data", "results", "items", "records", "rows"):
            if isinstance(payload.get(key), list):
                return _from_json(payload[key])
        for value in payload.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return pd.json_normalize(value)
        return pd.json_normalize(payload)

    raise ValueError("JSON is not a list or object")


def _from_text(text):
    # 1) JSON  2) CSV/delimited  3) XML  4) fallback: one row per line
    try:
        return _from_json(json.loads(text))
    except Exception:
        pass

    first_line = text.lstrip().splitlines()[0] if text.strip() else ""
    if any(d in first_line for d in (",", ";", "\t", "|")):
        try:
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
            if df.shape[1] > 1:
                return df
        except Exception:
            pass

    if text.lstrip().startswith("<"):
        try:
            return pd.read_xml(io.StringIO(text))
        except Exception:
            pass

    return pd.DataFrame({"line": [l for l in text.splitlines() if l.strip()]})


def _read_csv(data: bytes, **kwargs) -> pd.DataFrame:
    """Read a CSV/TSV, trying common encodings (many real-world files are not UTF-8)."""
    last_error = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_error = e
    raise last_error


TABLE_EXT = (".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet", ".xml")


def _load_zip(data):
    """Open a zip and load the largest table-like file inside it."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        files = [i for i in z.infolist() if not i.is_dir() and "__MACOSX" not in i.filename]
        tables = [i for i in files if i.filename.lower().endswith(TABLE_EXT)]
        if not tables:
            raise ValueError("No CSV/Excel/JSON/Parquet file found inside the zip. Files: "
                             + ", ".join(i.filename for i in files[:10]))
        best = max(tables, key=lambda i: i.file_size)
        return load_bytes(z.read(best.filename), best.filename.split("/")[-1])


def load_bytes(data: bytes, filename: str) -> pd.DataFrame:
    """Load raw bytes into a DataFrame, choosing the parser by file extension."""

    ext = os.path.splitext(filename)[1].lower()

    if ext == ".zip":
        return _load_zip(data)

    buf = io.BytesIO(data)

    if ext == ".csv":
        return _read_csv(data, sep=None, engine="python")
    if ext == ".tsv":
        return _read_csv(data, sep="\t")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(buf)
    if ext == ".parquet":
        return pd.read_parquet(buf)
    if ext == ".json":
        return _from_json(json.loads(data.decode("utf-8", errors="replace")))
    if ext == ".xml":
        return pd.read_xml(buf)
    if ext == ".pdf":
        from pypdf import PdfReader  # pip install pypdf
        text = "\n".join(p.extract_text() or "" for p in PdfReader(buf).pages)
        return pd.DataFrame({"line": [l for l in text.splitlines() if l.strip()]})

    # unknown / text-like formats
    return _from_text(data.decode("utf-8", errors="replace"))


def fetch_api(url, method="GET", auth="None", key="", key_name="", headers=None, timeout=20):
    """
    Call any API and return (DataFrame, size_in_bytes).

    auth: "None" | "Bearer token" | "Custom header" | "Query parameter"
    Different websites expect the key in different places, so all common
    styles are supported.
    """

    headers = dict(headers or {})
    params = {}

    if auth == "Bearer token" and key:
        headers["Authorization"] = f"Bearer {key}"
    elif auth == "Custom header" and key:
        headers[key_name or "X-API-Key"] = key
    elif auth == "Query parameter" and key:
        params[key_name or "api_key"] = key

    resp = requests.request(method, url, headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()

    # guess the format from the response type, else from the URL
    ctype = resp.headers.get("Content-Type", "").lower()
    name = os.path.basename(urlparse(url).path) or "response"

    if "json" in ctype:
        name += ".json"
    elif "csv" in ctype:
        name += ".csv"
    elif "xml" in ctype:
        name += ".xml"

    return load_bytes(resp.content, name), len(resp.content)