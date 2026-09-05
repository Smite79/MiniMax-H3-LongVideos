"""The README carries the work-in-progress notice, at the top, in bold.

Asked for on 2026-09-05: it goes on every commit that changes the node. A rule I
have to remember is a rule that lapses, so it is checked here instead -- the suites
run before every push, and this fails if the notice is missing, demoted below the
title, or has drifted in wording.

Run directly, or as part of `python test_node.py && python test_smoke.py &&
python test_readme.py`.
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

NOTICE = ("This node is a constant work in progress! If you are noticing bugs or "
          "features that do not work, please ensure that you are pulling the most "
          "recent version and updating your workflows.")

_FAILED = []


def check(label, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + ("" if ok else f"  [{detail}]"))
    if not ok:
        _FAILED.append(label)


def _flat(text):
    """Wrapping is a formatting choice; the sentence is what has to be there."""
    return re.sub(r"\s+", " ", text).strip()


def main():
    print("\n=== the README carries the work-in-progress notice ===")
    path = os.path.join(HERE, "README.md")
    raw = io.open(path, encoding="utf-8").read()
    flat = _flat(raw)

    check("the notice is present", NOTICE in flat,
          "not found -- put it back at the top of README.md")

    # Bold, so it reads as a notice rather than as a paragraph.
    check("...in bold", ("**" + NOTICE + "**") in flat,
          "found, but not wrapped in **")

    # Above the title. A warning below the fold is a warning nobody reads.
    if NOTICE in flat:
        title = flat.find("# H3-LongVideos")
        check("...above the title", 0 <= flat.find(NOTICE) < title,
              "it sits below the heading")

    # The Ko-fi button is allowed to precede it; nothing else should.
    head = [l.strip() for l in raw.split("\n") if l.strip()][:3]
    check("...within the first few lines",
          any(NOTICE.split("!")[0] in _flat(l) for l in head),
          "; ".join(h[:40] for h in head))

    print("\nRESULT: " + ("ALL PASSED" if not _FAILED
                          else f"{len(_FAILED)} FAILURE(S): " + "; ".join(_FAILED)))
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
