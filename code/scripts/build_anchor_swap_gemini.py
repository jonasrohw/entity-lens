"""Generate and audit a frozen paired-fact dataset through Vertex Gemini.

Credentials stay in gcloud; responses, prompts, and token usage are retained.
Generation/audit never sees OLMo outcomes. No external Python dependencies.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/anchor_swap_gemini"
RAW = ROOT / "generated/anchor_swap_gemini/api"
SPECS = {
    "countries": dict(
        relations=["capital", "currency"],
        templates=["The capital city of {} is", "The currency unit used in {} is"],
        instruction="Current sovereign countries, across all continents. Exactly one unambiguous capital. Currency answers must be generic unit names, e.g. Dollar, Euro, Pound, Dinar, Yen; omit national adjectives. Exclude multiple-capital or multiple-currency countries and recent changes after 2023.",
    ),
    "athletes": dict(
        relations=["sport", "position"],
        templates=["The sport played by {} is", "The main playing position of {} is"],
        instruction="Famous professional athletes with a single well-established main sport and playing position. Mix association football (answer soccer), American football (answer football), basketball, baseball, ice hockey, rugby, cricket, volleyball. No individual sports without playing positions. Include women and non-US athletes. Prefer retired players. Plain conventional position names; no handedness modifiers.",
    ),
    "authors": dict(
        relations=["birth_city", "writing_language"],
        templates=[
            "The city where {} was born is",
            "The primary language in which {} wrote is",
        ],
        instruction="Famous deceased novelists, poets, playwrights. Birthplace must be a city/town with a well-established English name. Writing language must be a single clearly primary language; exclude famous multilingual authors. Mix continents, eras, genders; at least half not English-language. Use primary literary language, not nationality or guessed native language.",
    ),
    "films": dict(
        relations=["director", "composer"],
        templates=[
            "The director of the film {} is",
            "The composer of the original score for the film {} is",
        ],
        instruction="Famous films released before 2020, with one director and one clearly credited original-score composer. Exclude co-directors, co-composers, compilation soundtracks, ambiguous remake titles. Mix decades, genres, countries; at least one quarter non-Hollywood. Use full person names. Titles must identify the film unambiguously without added metadata.",
    ),
    "landmarks": dict(
        relations=["country", "city"],
        templates=["The country containing {} is", "The city containing {} is"],
        instruction="Famous individual human-made landmarks located unambiguously within a city/town. Mix continents, landmark types, eras; no more than one third Europe. Exclude subject names containing the answer country/city or demonym, geographic regions, natural landmarks, boundary-crossing sites, disputed territories. City should be the actual municipality, not nearby tourism hub.",
    ),
    "cities": dict(
        relations=["country", "continent"],
        templates=[
            "The country containing the city of {} is",
            "The continent containing the city of {} is",
        ],
        instruction="Famous current cities across inhabited continents. Unambiguous names, stable current sovereign country. Six continent convention: Africa, Asia, Europe, North America, South America, Oceania. Exclude transcontinental cities, disputed territories, country or continent names copied in city names. No more than one third Europe; diverse countries.",
    ),
}
LOCK = threading.Lock()
TOKEN = {"value": "", "time": 0}


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def call(prompt, name, schema, model):
    path = RAW / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())["parsed"]
    with LOCK:
        if time.time() - TOKEN["time"] > 1800:
            p = subprocess.run(
                ["gcloud", "auth", "application-default", "print-access-token"],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            TOKEN.update(value=p.stdout.strip(), time=time.time())
        token = TOKEN["value"]
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    url = f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/publishers/google/models/{model}:generateContent"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema,
            "maxOutputTokens": 32768,
            "thinkingConfig": {"thinkingLevel": "HIGH"},
        },
    }
    write(RAW / f"{name}_request.json", body)
    for attempt in range(5):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode(),
                headers={
                    "Authorization": "Bearer " + token,
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=600) as f:
                response = json.load(f)
            parts = response["candidates"][0]["content"]["parts"]
            parsed = json.loads(
                "".join(x.get("text", "") for x in parts if not x.get("thought"))
            )
            write(
                path,
                dict(
                    parsed=parsed,
                    response=response,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    requested_model=model,
                ),
            )
            print(
                name,
                response.get("modelVersion"),
                response.get("usageMetadata"),
                flush=True,
            )
            return parsed
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:1000]
            if e.code not in [429, 500, 502, 503, 504]:
                raise RuntimeError(f"HTTP {e.code}: {msg}") from None
        except (
            TimeoutError,
            urllib.error.URLError,
            json.JSONDecodeError,
            KeyError,
        ) as e:
            msg = str(e)
        time.sleep(2**attempt)
    raise RuntimeError(name + ": " + msg)


ROW_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        k: {"type": "STRING"}
        for k in ["subject", "object_a", "object_b", "region", "source_url"]
    },
    "required": ["subject", "object_a", "object_b", "region", "source_url"],
}
GEN_SCHEMA = {
    "type": "OBJECT",
    "properties": {"rows": {"type": "ARRAY", "items": ROW_SCHEMA}},
    "required": ["rows"],
}
AUDIT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "rows": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "index": {"type": "INTEGER"},
                    "verdict": {
                        "type": "STRING",
                        "enum": ["accept", "reject", "uncertain"],
                    },
                    "reason": {"type": "STRING"},
                },
                "required": ["index", "verdict", "reason"],
            },
        }
    },
    "required": ["rows"],
}


def domain(name, model):
    spec = SPECS[name]
    rows = []
    for batch in range(2):
        prompt = f"""Build 80 factual subject-relation-object PAIRS for a scientific generalization test. Each subject has both relations: {spec["relations"]}. The hypothesis may fail; do NOT choose based on expected model performance. {spec["instruction"]}
Use widely known subjects likely known before 2024. Short conventional English answers, standard capitalisation, no lists, explanations, slash alternatives or parentheses. Facts must be correct and single-valued under the supplied relation. Include a relevant English Wikipedia source URL per subject (a source pointer, not a claim you fetched it). Region describes geographical coverage. Output exactly 80 distinct rows. Exclude these subjects already chosen: {json.dumps([r["subject"] for r in rows])}. Fields subject, object_a, object_b, region, source_url."""
        rows.extend(call(prompt, f"{name}_generate_{batch}", GEN_SCHEMA, model)["rows"])
    # Duplicates and exact same answers cannot distinguish relation selection.
    seen = set()
    candidates = []
    exclusions = []
    for r in rows:
        key = r["subject"].casefold()
        if key in seen or r["object_a"].casefold() == r["object_b"].casefold():
            exclusions.append(
                dict(row=r, reason="duplicate subject or identical answers")
            )
            continue
        seen.add(key)
        candidates.append(r)
    accepted = []
    for batch, start in enumerate(range(0, len(candidates), 32)):
        chunk = candidates[start : start + 32]
        bad = {
            "subject": "France",
            "object_a": "__FALSE_INVALID_ANSWER__",
            "object_b": "__FALSE_INVALID_ANSWER__",
            "region": "control",
            "source_url": "",
        }
        prompt = f"""Independently fact-check every row below for a paired-fact experiment. Relations {spec["relations"]}; specification: {spec["instruction"]}. Reject false facts, ambiguous entity labels, multiple equally defensible answers, misleading geography, title ambiguity, answer leakage in subject. Use uncertain whenever you cannot confidently verify from your knowledge. Source URLs are unverified pointers, NOT evidence. Do not favor any hypothesis. Do not rewrite answers. Return an explicit verdict for EVERY row index 0..{len(chunk)}. The final row is an invalid negative control and must be rejected. Data:\n{json.dumps(chunk + [bad], ensure_ascii=False)}"""
        audit = call(prompt, f"{name}_audit_{batch}", AUDIT_SCHEMA, model)["rows"]
        by = {r["index"]: r for r in audit}
        assert len(audit) == len(chunk) + 1 and set(by) == set(range(len(chunk) + 1))
        assert by[len(chunk)]["verdict"] == "reject"
        for i, r in enumerate(chunk):
            verdict = by[i]
            (accepted if verdict["verdict"] == "accept" else exclusions).append(
                dict(r, audit=verdict)
                if verdict["verdict"] == "accept"
                else dict(row=r, reason=verdict)
            )
    write(
        OUT / f"{name}_candidates.json",
        dict(spec=spec, accepted=accepted, exclusions=exclusions),
    )
    print(name, "accepted", len(accepted), "excluded", len(exclusions), flush=True)
    return name, len(accepted)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.7-flash")
    ap.add_argument("--domains", nargs="*", default=list(SPECS))
    a = ap.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for result in pool.map(lambda x: domain(x, a.model), a.domains):
            print(result, flush=True)


if __name__ == "__main__":
    main()
