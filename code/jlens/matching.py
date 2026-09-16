"""Lexical object matching and name-leakage controls."""

import re

STOP = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "de",
    "del",
    "van",
    "von",
    "der",
    "den",
    "di",
    "da",
    "la",
    "le",
    "el",
    "ibn",
    "bin",
    "jr",
    "sr",
    "ii",
    "iii",
    "iv",
    "king",
    "queen",
    "doctor",
    "dr",
    "professor",
    "prof",
    "mr",
    "mrs",
    "ms",
    "miss",
    "sir",
    "lord",
    "lady",
    "captain",
    "major",
    "general",
    "colonel",
    "saint",
    "st",
    "baron",
    "count",
    "duke",
    "master",
    "mister",
    "madame",
    "president",
    "father",
    "mother",
    "uncle",
    "aunt",
    "lady",
    "dame",
}


def leaky(obj, subj):
    """True if a gold word is a morphological variant of a subject word.
    Prefix agreement on the first 4 characters, either direction --- enough
    for Persei/Perseus and Librae/Libra without merging unrelated names."""
    sw = [w for w in re.findall(r"[A-Za-z]+", subj) if len(w) > 2]
    ow = [w for w in re.findall(r"[A-Za-z]+", obj) if len(w) > 2]
    return any(
        len(o) >= 4 and len(w) >= 4 and o[:4].lower() == w[:4].lower()
        for o in ow
        for w in sw
    )


def distinctive(obj, subj):
    """Words of the object that could identify it: not a title/particle, and
    not shared with the subject's own name (surnames are inherited, so
    `Johann Georg Kant` would otherwise be 'found' in any text about Kant)."""
    sw = {w.lower() for w in re.findall(r"[A-Za-z]+", subj)}
    return {
        w.lower()
        for w in re.findall(r"[A-Za-z]+", obj)
        if len(w) > 2 and w.lower() not in STOP and w.lower() not in sw
    }
