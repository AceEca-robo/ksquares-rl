"""Build and load the C tree walk of agents/az_mcts.py (cffi, API mode).

``load()`` returns the compiled module (``ffi``, ``lib``) or None when no C
compiler is available; the extension is built on first use next to this
file (``agents/_mcts_tree*.so``, not in git) and rebuilt when the C source
is newer. ``python -m agents.mcts_c`` builds it explicitly.
"""

from __future__ import annotations

import importlib
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "csrc" / "mcts_tree.c"
CDEF = """
typedef struct {
    double *P, *N, *W;
    int32_t *kid;
    int32_t nm;
    int32_t n;
    int32_t player;
    int32_t terminal;
    double value;
} node_t;
int mcts_descend(node_t *nodes, int root, double c_puct, double fpu_red, double vl,
                 int32_t *pn, int32_t *pa, int *len);
void mcts_undo(node_t *nodes, const int32_t *pn, const int32_t *pa, int len, double vl);
void mcts_backup(node_t *nodes, const int32_t *pn, const int32_t *pa, int len, int leaf, double vl);
"""
_mod = False  # not tried yet


def build() -> None:
    from cffi import FFI
    ffi = FFI()
    ffi.cdef(CDEF)
    ffi.set_source("agents._mcts_tree", SRC.read_text(),
                   extra_compile_args=["-O2", "-ffp-contract=off", "-fno-fast-math"])
    so = ffi.compile(tmpdir=str(HERE.parent / "build" / "cffi"), verbose=False)
    shutil.copy2(so, HERE / Path(so).name)


def _built():
    for p in HERE.glob("_mcts_tree*.so"):
        if p.stat().st_mtime >= SRC.stat().st_mtime:
            return True
    return False


def load():
    global _mod
    if _mod is False:
        _mod = None
        if os.environ.get("KSQ_MCTS_PY"):
            return None
        try:
            if not _built():
                build()
            _mod = importlib.import_module("agents._mcts_tree")
        except Exception:  # no compiler / no cffi: the Python tree is used
            _mod = None
    return _mod


if __name__ == "__main__":
    build()
    print("built", sorted(p.name for p in HERE.glob("_mcts_tree*.so")))
