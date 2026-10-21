"""Fetch the 20 screening postmortems and store their text ENCRYPTED (copyright: the
plain text is never committed). Encryption: random AES-256 key per run, wrapped with the
screening public key; only the holder of the private key can read the bundle.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import requests
from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "research-screening (AgileOps-Agentic-Framework revision)"}


def text(url: str) -> tuple[str, str]:
    try:
        r = requests.get(url, headers=UA, timeout=40, allow_redirects=True)
        if r.status_code != 200:
            return f"http_{r.status_code}", ""
        soup = BeautifulSoup(r.text, "html.parser")
        for s in soup(["script", "style", "nav", "header", "footer"]):
            s.decompose()
        lines = [ln.strip() for ln in soup.get_text("\n").splitlines()]
        return "ok", "\n".join(ln for ln in lines if ln)[:120_000]
    except Exception as exc:
        return type(exc).__name__, ""


def main() -> None:
    sample = json.loads((HERE / "sample_20.json").read_text())["sample"]
    bundle = []
    for item in sample:
        status, body = text(item["url"])
        bundle.append({**item, "status": status, "chars": len(body), "text": body})
    with tempfile.TemporaryDirectory() as tmp:
        plain, key = Path(tmp) / "bundle.json", Path(tmp) / "key.bin"
        plain.write_text(json.dumps(bundle))
        key.write_bytes(os.urandom(32))
        subprocess.run(["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-salt", "-in", str(plain),
                        "-out", str(HERE / "bundle.json.enc"), "-pass", f"file:{key}"], check=True)
        subprocess.run(["openssl", "pkeyutl", "-encrypt", "-pubin", "-inkey", str(HERE / "screen_public.pem"),
                        "-in", str(key), "-out", str(HERE / "bundle.key.enc")], check=True)
    (HERE / "fetch_status.json").write_text(json.dumps([{k: v for k, v in b.items() if k != "text"} for b in bundle], indent=1))
    print(json.dumps([{k: b[k] for k in ("id", "org", "status", "chars")} for b in bundle], indent=1))


if __name__ == "__main__":
    main()
