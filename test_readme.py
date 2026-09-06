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
import json
import os
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

    # YAML front matter is metadata for the Hugging Face Hub, not visible content,
    # so it does not count against the notice's position. The Ko-fi button is
    # allowed to precede it; nothing else should.
    body = raw
    if body.lstrip().startswith("---"):
        _rest = body.lstrip()[3:]
        _end = _rest.find("\n---")
        if _end != -1:
            body = _rest[_end + 4:]
    head = [l.strip() for l in body.split("\n") if l.strip()][:3]
    check("...within the first few lines",
          any(NOTICE.split("!")[0] in _flat(l) for l in head),
          "; ".join(h[:40] for h in head))


    # ------------------------------------------------------------------ Hub
    # The Hugging Face Hub counts a download as an HTTP request for a QUERY FILE,
    # picked so a multi-file repo is not counted many times over. With no library
    # declared it looks for config.json, config.yaml, hyperparams.yaml,
    # params.json or meta.yaml -- and a repo with none of them is never counted at
    # all. This repo read 0 downloads against 93 likes for exactly that reason.
    #
    # Enforced here so it survives: a file whose only job is to be requested is
    # the first thing somebody deletes as clutter.
    # https://huggingface.co/docs/hub/models-download-stats
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = os.path.join(here, "config.json")
    check("config.json exists, so the Hub can count downloads", os.path.isfile(cfg),
          "without it the download counter stays at zero for ever")
    if os.path.isfile(cfg):
        try:
            with io.open(cfg, encoding="utf-8") as fh:
                data = json.load(fh)
            ok = isinstance(data, dict) and bool(data)
        except Exception as exc:
            ok, data = False, {}
            check("...and is valid JSON", False, str(exc)[:60])
        else:
            check("...and is valid JSON", ok)
        check("...and says why it is there, so it is not deleted as clutter",
              "download" in json.dumps(data).lower())

    # Repo-card metadata: without it the Hub cannot categorise the repo, and every
    # push warns about it.
    front = raw.lstrip()
    check("the README carries YAML front matter", front.startswith("---"),
          front[:40])
    if front.startswith("---"):
        end = front[3:].find("\n---")
        meta = front[3:3 + end] if end != -1 else ""
        check("...with tags", "tags:" in meta)
        check("...naming comfyui", "comfyui" in meta)
        check("...and a license", "license:" in meta)

    print("\nRESULT: " + ("ALL PASSED" if not _FAILED
                          else f"{len(_FAILED)} FAILURE(S): " + "; ".join(_FAILED)))
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
