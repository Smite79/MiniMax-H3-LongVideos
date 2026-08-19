"""
REF2VA -- a second sampler in the H3-LongVideos pack
====================================================
A variant of the FL2VA sampler, developed alongside it rather than replacing it.
This folder is NESTED inside the pack, so ComfyUI never imports it on its own
(only top-level entries of custom_nodes/ are scanned). The pack's own
`__init__.py` one directory up imports what is exported here.

Nodes registered by REF2VA:
  * H3 Long Videos REF2VA  (sampler.py)  - the variant sampler

Deliberately NOT registered here:
  * H3 Shot Length     (shot_length.py)
  * H3 Model Inspector (inspector.py)
These are copies of the pack's utility nodes and carry the SAME registration
keys. ComfyUI keeps one flat registry, so registering them again would overwrite
the originals with an identical-looking duplicate and leave no way to tell which
copy a workflow is running. Both nodes are model- and sampler-agnostic, so REF2VA
wires to the pack's single instance of each instead. The files stay here only so
this folder remains a self-contained, runnable copy.
"""

from .sampler import NODE_CLASS_MAPPINGS as _s_c, NODE_DISPLAY_NAME_MAPPINGS as _s_d

NODE_CLASS_MAPPINGS = dict(_s_c)
NODE_DISPLAY_NAME_MAPPINGS = dict(_s_d)
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
