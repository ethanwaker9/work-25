import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def ensure_dirs():
    for d in (RESULTS, FIGURES):
        if not os.path.isdir(d):
            os.makedirs(d)


def save(name, obj):
    ensure_dirs()
    with open(os.path.join(RESULTS, name + ".json"), "w") as fh:
        json.dump(obj, fh, indent=1, default=float)


def load(name):
    with open(os.path.join(RESULTS, name + ".json")) as fh:
        return json.load(fh)


def fmt_table(rows, header, sep=" & ", end=" \\\\"):
    out = [sep.join(header) + end]
    for r in rows:
        out.append(sep.join(str(x) for x in r) + end)
    return "\n".join(out)
