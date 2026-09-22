"""Corroborate authors' writing languages with live Wikidata P6886 claims."""

import concurrent.futures
import gzip
import hashlib
import json
import re
import time
import urllib.request

from build_anchor_swap_gemini import OUT, write
from source_audit_anchor_swap import SOURCE


def fetch_entity(qid):
    path = SOURCE / f"wikidata_{qid}.json"
    if path.exists():
        return json.loads(path.read_text())
    req = urllib.request.Request(
        f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json",
        headers={"User-Agent": "JLENS-academic-research/1.0"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as f:
                d = json.load(f)
            write(path, d)
            return d
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def supplement(row):
    key = hashlib.sha256(row["source_url"].encode()).hexdigest()[:24]
    path = SOURCE / f"{key}.json"
    while not path.exists():
        time.sleep(1)
    s = json.loads(path.read_text())
    if "wikidata_writing_languages" in s:
        return
    try:
        with gzip.open(SOURCE / f"{key}.html.gz", "rt") as f:
            h = f.read()
        match = re.search(r'"wgWikibaseItemId":"(Q\d+)"', h)
        if match is None:
            raise ValueError("No article Wikidata ID")
        qid = match[1]
        raw = fetch_entity(qid)
        entity = raw["entities"][qid]
        claims = []
        for c in entity.get("claims", {}).get("P6886", []):
            if c.get("rank") == "deprecated":
                continue
            value = c.get("mainsnak", {}).get("datavalue", {}).get("value", {})
            if not isinstance(value, dict) or "id" not in value:
                continue
            language = value["id"]
            lang_entity = fetch_entity(language)["entities"][language]
            claims.append(
                {
                    "qid": language,
                    "label": lang_entity.get("labels", {}).get("en", {}).get("value"),
                    "rank": c.get("rank"),
                    "references_present": bool(c.get("references")),
                }
            )
        s["wikidata_writing_languages"] = {
            "subject_qid": qid,
            "subject_label": entity.get("labels", {}).get("en", {}).get("value"),
            "property": "P6886 writing language",
            "claims": claims,
            "source_url": f"https://www.wikidata.org/wiki/{qid}",
            "snapshot_sha256": hashlib.sha256(
                (SOURCE / f"wikidata_{qid}.json").read_bytes()
            ).hexdigest(),
        }
    except Exception as e:
        s["wikidata_writing_languages"] = {"error": str(e)}
    write(path, s)


if __name__ == "__main__":
    rows = json.loads((OUT / "authors_candidates.json").read_text())["accepted"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(supplement, rows))
    print("Supplemented", len(rows), "authors", flush=True)
    # Wait until the first audit has finished so its result cannot overwrite v2.
    while not (OUT / "authors_sourced.json").exists():
        time.sleep(3)
    import source_audit_anchor_swap as audit

    audit.domain("authors", audit_tag="source_wikidata_audit")
    (OUT / "authors_wikidata_complete.flag").write_text("complete\n")
