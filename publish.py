# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.
"""Publish this node to BOTH remotes, and verify what landed.

    python test_node.py && python test_smoke.py && python test_readme.py
    python publish.py "commit message"

GitHub and the Hugging Face mirror share no merge base, so HF is not a git push
-- it is a file upload per file. That made it easy to push a subset by accident
and leave the mirror half-updated, so the file list lives HERE rather than in
whatever script was to hand, and every upload is verified by downloading the file
back and comparing SHA-256.

config.json is in the list for a reason that is not obvious: the Hub counts a
download as an HTTP request for a QUERY FILE, and with no library declared it
looks for config.json. A repo without one is never counted -- this one read 0
downloads against 93 likes until it was added. test_readme.py fails if it goes
missing; this makes sure it is actually uploaded.
https://huggingface.co/docs/hub/models-download-stats
"""
import hashlib
import io
import os
import subprocess
import sys

HF_REPO = "Smite79/MiniMax-H3-LongVideos"
HERE = os.path.dirname(os.path.abspath(__file__))

# Everything the mirror needs. Add to this when a file is added to the node.
FILES = [
    "config.json",          # the Hub's download counter reads this. Do not drop it.
    "README.md",            # carries the repo-card metadata in its front matter
    "LICENSE",              # the terms. Shipping the node without them is the bug.
    "__init__.py",
    "sampler.py",
    # sampler.py IMPORTS this. Leaving it off the list published a mirror whose
    # sampler could not load at all -- the verify step passed, because it only
    # checks that the files it knows about match.
    "engine.py",
    "shot_length.py",
    "inspector.py",
    "overlay.py",
]

# NOT PUBLISHED. The suites are how this node is developed, not how it is used:
# nobody installing it runs them, and they are the largest thing in the repo. They
# stay in git, where the history and the bug reports they encode belong, and off
# the mirror people download. Anything added here is deleted from the Hub on the
# next publish rather than quietly left behind.
NOT_SHIPPED = [
    "test_node.py",
    "test_smoke.py",
    "test_engine.py",
    "test_readme.py",
]


def _sha(path):
    with io.open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    msg = sys.argv[1] if len(sys.argv) > 1 else "Update"

    missing = [f for f in FILES if not os.path.isfile(os.path.join(HERE, f))]
    if missing:
        print("MISSING, refusing to publish a partial mirror:", ", ".join(missing))
        return 1

    print("== GitHub ==")
    r = subprocess.run(["git", "push", "origin", "main"], cwd=HERE,
                       capture_output=True, text=True)
    print((r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr)
          else "(nothing to push)")

    print()
    print("== Hugging Face ==")
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    bad = 0
    for name in FILES:
        p = os.path.join(HERE, name)
        api.upload_file(path_or_fileobj=p, path_in_repo=name, repo_id=HF_REPO,
                        repo_type="model", commit_message=msg)
        back = hf_hub_download(HF_REPO, name, repo_type="model",
                               force_download=True)
        ok = _sha(p) == _sha(back)
        bad += not ok
        print("  %-16s %s  %s" % (name, _sha(p)[:16], "MATCH" if ok else "MISMATCH"))

    listing = set(api.list_repo_files(HF_REPO, repo_type="model"))

    # Take down anything that should not be there. Dropping a file from FILES
    # only stops it being UPDATED -- the copy already on the Hub stays, and stale
    # is worse than absent because it still looks current to whoever downloads it.
    for name in NOT_SHIPPED:
        if name in listing:
            api.delete_file(path_in_repo=name, repo_id=HF_REPO,
                            repo_type="model",
                            commit_message="not shipped: " + name)
            print("  %-16s %s" % (name, "DELETED from the mirror"))
    listing = set(api.list_repo_files(HF_REPO, repo_type="model"))
    left = [n for n in NOT_SHIPPED if n in listing]
    if left:
        print("  STILL PRESENT:", ", ".join(left))
        bad += len(left)

    print()
    print("  config.json on the Hub:", "config.json" in listing)
    if "config.json" not in listing:
        bad += 1
    print()
    print("RESULT:", "all verified" if not bad else f"{bad} PROBLEM(S)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
