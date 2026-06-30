#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static anti-cheat validator for AKA / SOL-ExecBench kernel solutions.

This is the *single source of truth* for AKA's anti-cheat rules. It is a
standalone script (NOT a Claude Code hook) so it can be wired into the
evaluation / packaging path where the agent cannot bypass it. It only uses the
``ast`` module and NEVER imports or executes the kernel under test.

Cheat classes detected (see plan / SOL-ExecBench incident analysis):

  C1  Library delegation  -- the operator is computed by a library / torch op
                             instead of a self-written GPU kernel (067:
                             ``F.linear`` + ``F.scaled_dot_product_attention``;
                             015: ``flashinfer.BatchPrefillWithPagedKVCacheWrapper``).
  C2  Language-tag camouflage -- ``solution.json`` declares a real-kernel
                             framework (triton / cute_dsl / cutile / cuda_cpp)
                             that is NOT actually launched from the entry point,
                             and/or decorated kernels that are dead code
                             (015: dead ``@cute.kernel`` "for language
                             classification").
  C3  Shape/input-keyed memoization -- a process-global mutable cache (or
                             ``lru_cache``) keyed on input shape metadata that
                             gates genuine per-call work (host sync, python
                             loop, H2D copy) so it is excluded from the timed
                             region (015: ``_tile_cache_key=(total_q, ...)``).
  CX  Dynamic dispatch in the entry path (``exec``/``eval``/``importlib``/
                             ``getattr`` with a non-literal name) that defeats
                             static analysis -- hard fail.

C4 (timing-methodology gaming), C5 (masked-error PASS) and C6 (fabricated
target / Ref=0.000) are behavioural and are enforced by the dynamic gate in the
adapter-generated ``test_kernel.py`` and by honest reference benchmarking; this
file only adds a light static heuristic for C6.

Exit codes:  0 = clean (no findings / only INFO),  1 = WARN (needs
justification),  2 = hard FAIL.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# Policy + rule data
# --------------------------------------------------------------------------- #

POLICY_REQUIRE_SELF_WRITTEN = "require_self_written_kernel"
POLICY_ALLOW_LIBS = "allow_libs"

# Real, self-written GPU-kernel frameworks. Buckets are normalized:
#   triton, cute_dsl, cutile, cpp
SELF_WRITTEN_BUCKETS = {"triton", "cute_dsl", "cutile", "cpp"}

# Map declared SOL language -> self-written bucket (library langs map to None).
LANG_TO_BUCKET = {
    "triton": "triton",
    "cute_dsl": "cute_dsl",
    "cutile": "cutile",
    "cuda_cpp": "cpp",
    "cutlass": "cpp",
    # library / non-kernel languages:
    "pytorch": None,
    "cublas": None,
    "cudnn": None,
    "cudnn_frontend": None,
}

# Whole-operator libraries: their mere presence on the compute path (or import)
# is delegation -- there is no legitimate "glue" use of these.
BANNED_OPERATOR_LIB_ROOTS = {"flashinfer", "flash_attn", "xformers", "vllm", "aiter"}
# Whole-operator torch ops that ARE the operator being benchmarked.
BANNED_TORCH_OP_SUFFIXES = {"scaled_dot_product_attention"}

# Heavy torch ops that may be legitimate glue (projection) inside a real fused
# kernel -> WARN if a real kernel exists, count toward delegation if no kernel.
SOFT_LIBRARY_OP_SUFFIXES = {
    "linear", "conv1d", "conv2d", "conv3d", "matmul", "bmm", "einsum",
    "multi_head_attention_forward",
}

# Device-transfer / host-sync signatures that, when gated behind a cross-call
# cache, indicate real per-call work was moved out of the timed region (C3).
D2H_SYNC_METHODS = {"cpu", "item", "tolist", "numpy"}
TENSOR_CTORS = {"torch.tensor", "torch.as_tensor"}

DYNAMIC_DISPATCH_NAMES = {"exec", "eval", "compile", "__import__"}
DYNAMIC_DISPATCH_FQNS = {"importlib.import_module"}

SEV_FAIL = "FAIL"
SEV_WARN = "WARN"
SEV_INFO = "INFO"


@dataclass
class Finding:
    code: str
    severity: str
    message: str
    lineno: int = 0

    def as_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity,
                "message": self.message, "lineno": self.lineno}


# --------------------------------------------------------------------------- #
# Name / import resolution
# --------------------------------------------------------------------------- #

def _dotted(node: ast.AST) -> Optional[str]:
    """Return the dotted name for a Name/Attribute chain, else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _build_alias_map(tree: ast.Module) -> dict[str, str]:
    """Map local binding -> canonical module/symbol path from imports."""
    alias: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                # `import a.b.c` binds `a`; `import a.b.c as x` binds `x`->a.b.c
                if a.asname:
                    alias[a.asname] = a.name
                else:
                    alias[a.name.split(".")[0]] = a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for a in node.names:
                local = a.asname or a.name
                alias[local] = f"{mod}.{a.name}" if mod else a.name
    return alias


def _canonical(dotted: Optional[str], alias: dict[str, str]) -> Optional[str]:
    if not dotted:
        return None
    head, _, rest = dotted.partition(".")
    if head in alias:
        base = alias[head]
        return f"{base}.{rest}" if rest else base
    return dotted


# --------------------------------------------------------------------------- #
# Kernel-decorator detection
# --------------------------------------------------------------------------- #

def _decorator_framework(dec: ast.AST, alias: dict[str, str]) -> Optional[str]:
    """Return the self-written framework bucket a decorator marks, else None."""
    node = dec.func if isinstance(dec, ast.Call) else dec
    fqn = _canonical(_dotted(node), alias)
    if not fqn:
        return None
    last = fqn.rsplit(".", 1)[-1]
    if fqn.startswith("triton.") and last in {"jit", "autotune", "heuristics"}:
        return "triton"
    if last in {"jit", "kernel"} and (".cute" in fqn or fqn.startswith("cute.")
                                      or "cutlass.cute" in fqn):
        return "cute_dsl"
    if "cutile" in fqn and last in {"jit", "kernel"}:
        return "cutile"
    return None


def _collect_kernels(tree: ast.Module, alias: dict[str, str]) -> dict[str, str]:
    """name -> framework bucket for every *decorated* kernel/host FunctionDef."""
    kernels: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                fw = _decorator_framework(dec, alias)
                if fw:
                    kernels[node.name] = fw
                    break
    return kernels


def _with_aliases(tree: ast.Module, decorated: dict[str, str]) -> dict[str, str]:
    """Extend decorated kernels with simple local aliases (``k = _kernel``) so a
    kernel launched through a renamed handle is still recognized. Used only for
    launch detection, never for the dead-decorator check."""
    kernels = dict(decorated)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in kernels
                    and node.targets[0].id not in kernels):
                kernels[node.targets[0].id] = kernels[node.value.id]
                changed = True
    return kernels


# --------------------------------------------------------------------------- #
# Call-graph reachability from the entry point
# --------------------------------------------------------------------------- #

class _FuncIndex:
    def __init__(self, tree: ast.Module):
        self.defs: dict[str, ast.AST] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.defs[node.name] = node


def _reachable_funcs(tree: ast.Module, entry: str,
                     kernels: dict[str, str]) -> set[str]:
    """Names of local functions reachable from `entry` (kernels are leaves)."""
    index = _FuncIndex(tree)
    if entry not in index.defs:
        return set()
    seen: set[str] = set()
    stack = [entry]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        fn = index.defs.get(name)
        if fn is None:
            continue
        for call in ast.walk(fn):
            if not isinstance(call, ast.Call):
                continue
            # subscript-launch `kern[grid](...)` -> reach the kernel, don't recurse
            target = call.func
            if isinstance(target, ast.Subscript):
                base = _dotted(target.value)
                if base in index.defs and base not in seen:
                    seen.add(base)
                continue
            callee = _dotted(target)
            if callee and callee in index.defs:
                stack.append(callee)
    return seen


# --------------------------------------------------------------------------- #
# Core analysis
# --------------------------------------------------------------------------- #

def _iter_calls(fn: ast.AST):
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            yield node


def _launch_sites(fn: ast.AST, kernels: dict[str, str],
                  alias: dict[str, str]) -> list[tuple[str, Optional[str]]]:
    """(framework bucket, kernel name|None) for each launch inside `fn`."""
    found: list[tuple[str, Optional[str]]] = []
    for call in _iter_calls(fn):
        target = call.func
        # triton subscript launch: kernel[grid](...)
        if isinstance(target, ast.Subscript):
            base = _dotted(target.value)
            if base in kernels:
                found.append((kernels[base], base))
            continue
        callee = _dotted(target)
        # cute/cutile host (@cute.jit) called directly -> launch
        if callee in kernels:
            found.append((kernels[callee], callee))
            continue
        fqn = _canonical(callee, alias)
        if not fqn:
            continue
        last = fqn.rsplit(".", 1)[-1]
        # inline C++/CUDA JIT compile+load -> cpp bucket
        if last in {"load_inline", "load"} and ("cpp_extension" in fqn
                                                 or "load_inline" in fqn):
            found.append(("cpp", None))
        # `<cute.kernel call>.launch(...)`
        if last == "launch":
            inner = target.value if isinstance(target, ast.Attribute) else None
            if isinstance(inner, ast.Call):
                inner_callee = _dotted(inner.func)
                if inner_callee in kernels:
                    found.append((kernels[inner_callee], inner_callee))
    return found


def _gated_global_cache(fn: ast.AST) -> Optional[ast.AST]:
    """Return the function node if it is a cross-call cache gating real work.

    Heuristic for C3: the function (a) declares `global X` and rebinds it
    (cross-call mutable state) or is decorated with lru_cache/cache, (b) has an
    early-return guarded by a comparison/membership test, and (c) contains
    genuinely recomputable per-call work behind that guard -- a D2H sync, a
    python loop, or a device tensor construction. Mere scratch allocation
    (`torch.empty`) is NOT flagged.
    """
    # (a) cross-call state
    globals_declared: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Global):
            globals_declared.update(node.names)
    has_lru = False
    if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for dec in fn.decorator_list:
            d = _dotted(dec.func if isinstance(dec, ast.Call) else dec) or ""
            if d.rsplit(".", 1)[-1] in {"lru_cache", "cache"}:
                has_lru = True
    if not globals_declared and not has_lru:
        return None

    # (b) early-return guarded by a test
    has_guarded_return = False
    for node in ast.walk(fn):
        if isinstance(node, ast.If):
            for stmt in node.body:
                if isinstance(stmt, ast.Return):
                    has_guarded_return = True
    if has_lru:
        has_guarded_return = True  # lru_cache is itself the guard

    # (c) recomputable per-call work
    has_work = False
    for node in ast.walk(fn):
        if isinstance(node, (ast.For, ast.While)):
            has_work = True
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr in D2H_SYNC_METHODS:
                has_work = True
            dotted = _dotted(f)
            if dotted and dotted.rsplit(".", 1)[-1] in {"tensor", "as_tensor"}:
                if any(kw.arg == "device" for kw in node.keywords):
                    has_work = True
    if has_guarded_return and has_work:
        return fn
    return None


def _key_uses_shape(fn: ast.AST) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and node.attr in {"shape", "numel", "size"}:
            return True
        if isinstance(node, ast.Call):
            d = _dotted(node.func)
            if d == "len" or (d and d.rsplit(".", 1)[-1] == "data_ptr"):
                return True
    return False


def analyze(kernel_source: str, solution: Optional[dict] = None,
            policy: str = POLICY_REQUIRE_SELF_WRITTEN,
            entry_symbol: Optional[str] = None) -> list[Finding]:
    """Run all static checks. Returns a list of Findings (possibly empty)."""
    findings: list[Finding] = []
    try:
        tree = ast.parse(kernel_source)
    except SyntaxError as e:
        return [Finding("CX", SEV_FAIL, f"kernel source does not parse: {e}", e.lineno or 0)]

    alias = _build_alias_map(tree)

    # entry symbol: explicit arg > solution entry_point > 'run'
    if entry_symbol is None and solution:
        ep = (solution.get("spec") or {}).get("entry_point", "")
        if "::" in ep:
            entry_symbol = ep.split("::")[-1]
    entry_symbol = entry_symbol or "run"

    index = _FuncIndex(tree)
    if entry_symbol not in index.defs:
        findings.append(Finding("CX", SEV_FAIL,
                                f"entry function '{entry_symbol}' not found in kernel source", 0))
        return findings

    decorated = _collect_kernels(tree, alias)       # decorated FunctionDefs only
    kernels = _with_aliases(tree, decorated)         # + local aliases (launch detection)
    reachable = _reachable_funcs(tree, entry_symbol, kernels)
    reachable_funcs = [index.defs[n] for n in reachable if n in index.defs]

    # frameworks + kernel names actually launched on the data path from entry
    frameworks_on_path: set[str] = set()
    launched_names: set[str] = set()
    for fn in reachable_funcs:
        for fw, name in _launch_sites(fn, kernels, alias):
            frameworks_on_path.add(fw)
            if name:
                launched_names.add(name)
    reached_kernels = reachable | launched_names

    # banned / soft library usage on the reachable path
    banned_lib_hits: list[tuple[str, int]] = []
    soft_lib_hits: list[tuple[str, int]] = []
    for fn in reachable_funcs:
        for call in _iter_calls(fn):
            # dynamic dispatch -> hard fail
            d = _dotted(call.func)
            cfqn = _canonical(d, alias)
            last = cfqn.rsplit(".", 1)[-1] if cfqn else None
            if last in DYNAMIC_DISPATCH_NAMES or cfqn in DYNAMIC_DISPATCH_FQNS:
                findings.append(Finding("CX", SEV_FAIL,
                    f"dynamic dispatch '{cfqn}' in entry path defeats static analysis", call.lineno))
            if last == "getattr" and len(call.args) >= 2 and not isinstance(call.args[1], ast.Constant):
                findings.append(Finding("CX", SEV_FAIL,
                    "getattr() with a non-literal attribute name in entry path", call.lineno))
            if not cfqn:
                continue
            root = cfqn.split(".")[0]
            if root in BANNED_OPERATOR_LIB_ROOTS or last in BANNED_TORCH_OP_SUFFIXES:
                banned_lib_hits.append((cfqn, call.lineno))
            elif last in SOFT_LIBRARY_OP_SUFFIXES:
                soft_lib_hits.append((cfqn, call.lineno))

    # imports of whole-operator libs anywhere (catches wrapper-object usage)
    imported_banned = sorted({
        v.split(".")[0] for v in alias.values()
        if v.split(".")[0] in BANNED_OPERATOR_LIB_ROOTS
    })

    require = policy == POLICY_REQUIRE_SELF_WRITTEN

    # ---- C1: library delegation ----
    for lib in imported_banned:
        findings.append(Finding("C1", SEV_FAIL if require else SEV_WARN,
            f"imports whole-operator library '{lib}' (delegation)", 0))
    for fqn, ln in banned_lib_hits:
        findings.append(Finding("C1", SEV_FAIL if require else SEV_WARN,
            f"calls whole-operator library op '{fqn}' on the compute path", ln))
    if not (frameworks_on_path & SELF_WRITTEN_BUCKETS):
        # no self-written kernel reachable from entry at all
        sev = SEV_FAIL if require else SEV_WARN
        detail = ""
        if soft_lib_hits:
            detail = f" (output produced by {sorted({h[0] for h in soft_lib_hits})})"
        findings.append(Finding("C1", sev,
            f"no self-written kernel (triton/cute_dsl/cutile/cuda_cpp) reachable "
            f"from '{entry_symbol}'; operator is delegated to torch/libraries{detail}", 0))
    else:
        # kernel exists; a single soft op may be legit glue -> WARN
        for fqn, ln in soft_lib_hits:
            findings.append(Finding("C1", SEV_WARN,
                f"heavy torch op '{fqn}' on compute path alongside a real kernel "
                f"-- ensure it is glue (projection), not the operator", ln))

    # ---- C2: language-tag camouflage ----
    declared_langs = []
    if solution:
        declared_langs = (solution.get("spec") or {}).get("languages", []) or []
    declared_buckets = {LANG_TO_BUCKET.get(l) for l in declared_langs}
    declared_buckets.discard(None)
    for bucket in sorted(declared_buckets):
        if bucket not in frameworks_on_path:
            findings.append(Finding("C2", SEV_FAIL,
                f"solution declares language bucket '{bucket}' but no such kernel "
                f"is launched from '{entry_symbol}' (tag camouflage)", 0))
    # decorated kernels that are dead code (only real decorated defs, not aliases).
    # A kernel whose framework IS on the data path is legitimately used (possibly
    # via an alias / one of several dispatched kernels) -> not camouflage.
    dead = {n: fw for n, fw in decorated.items()
            if n not in reached_kernels and fw not in frameworks_on_path}
    for name, fw in sorted(dead.items()):
        if fw in declared_buckets:
            findings.append(Finding("C2", SEV_FAIL,
                f"decorated {fw} kernel '{name}' is never reached from "
                f"'{entry_symbol}' yet '{fw}' is declared as a language (dead-decorator camouflage)",
                getattr(index.defs.get(name), "lineno", 0)))
        else:
            findings.append(Finding("C2", SEV_WARN,
                f"decorated {fw} kernel '{name}' is dead code (never reached from '{entry_symbol}')",
                getattr(index.defs.get(name), "lineno", 0)))

    # ---- C3: shape/input-keyed memoization ----
    for fn in reachable_funcs:
        gated = _gated_global_cache(fn)
        if gated is not None:
            shape_note = " keyed on input shape metadata" if _key_uses_shape(fn) else ""
            findings.append(Finding("C3", SEV_FAIL,
                f"function '{getattr(fn, 'name', '?')}' uses a cross-call cache{shape_note} "
                f"that gates per-call work (host sync / loop / H2D); this hides work from "
                f"the timed region", getattr(fn, "lineno", 0)))

    # ---- C6 (light static heuristic): hardcoded performance target ----
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            tgt = node.targets[0] if node.targets else None
            name = _dotted(tgt) or ""
            if any(k in name.lower() for k in ("target", "peak", "sol_target", "tflops_goal")):
                if isinstance(node.value, (ast.Constant, ast.BinOp)):
                    findings.append(Finding("C6", SEV_WARN,
                        f"hardcoded performance target '{name}' -- targets must derive from a "
                        f"measured reference (T_b), never fabricated", node.lineno))

    # dedup
    seen = set()
    uniq: list[Finding] = []
    for f in findings:
        key = (f.code, f.severity, f.message)
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    return uniq


def has_self_written_kernel(kernel_source: str, entry_symbol: str = "run") -> bool:
    """Reused by install.sh's generated_kernel_violations(): True iff a real
    kernel (triton/cute_dsl/cutile/cuda_cpp) is launched from the entry point."""
    try:
        tree = ast.parse(kernel_source)
    except SyntaxError:
        return False
    alias = _build_alias_map(tree)
    kernels = _with_aliases(tree, _collect_kernels(tree, alias))
    reachable = _reachable_funcs(tree, entry_symbol, kernels)
    index = _FuncIndex(tree)
    fws: set[str] = set()
    for n in reachable:
        if n in index.defs:
            fws |= {fw for fw, _ in _launch_sites(index.defs[n], kernels, alias)}
    return bool(fws & SELF_WRITTEN_BUCKETS)


def detected_frameworks(kernel_source: str, entry_symbol: str = "run") -> set[str]:
    """Self-written framework buckets actually launched from the entry point.

    Used by sol_adapter.py to label solution.json *truthfully* (declared
    languages must match what is really on the data path)."""
    try:
        tree = ast.parse(kernel_source)
    except SyntaxError:
        return set()
    alias = _build_alias_map(tree)
    kernels = _with_aliases(tree, _collect_kernels(tree, alias))
    reachable = _reachable_funcs(tree, entry_symbol, kernels)
    index = _FuncIndex(tree)
    fws: set[str] = set()
    for n in reachable:
        if n in index.defs:
            fws |= {fw for fw, _ in _launch_sites(index.defs[n], kernels, alias)}
    return fws & SELF_WRITTEN_BUCKETS


def verdict(findings: list[Finding]) -> str:
    if any(f.severity == SEV_FAIL for f in findings):
        return SEV_FAIL
    if any(f.severity == SEV_WARN for f in findings):
        return SEV_WARN
    return "OK"


def exit_code(findings: list[Finding]) -> int:
    return {"FAIL": 2, "WARN": 1, "OK": 0}[verdict(findings)]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="AKA static anti-cheat validator")
    p.add_argument("--kernel", required=True, help="path to the kernel .py file")
    p.add_argument("--solution", help="path to solution.json (optional)")
    p.add_argument("--policy", default=POLICY_REQUIRE_SELF_WRITTEN,
                   choices=[POLICY_REQUIRE_SELF_WRITTEN, POLICY_ALLOW_LIBS])
    p.add_argument("--entry", help="entry symbol (default: run / solution entry_point)")
    p.add_argument("--json", action="store_true", help="emit findings as JSON")
    args = p.parse_args(argv)

    kernel_source = Path(args.kernel).read_text(encoding="utf-8")
    solution = None
    if args.solution:
        solution = json.loads(Path(args.solution).read_text(encoding="utf-8"))
        # if the solution inlines the entry source, prefer it
        if not kernel_source.strip():
            for s in solution.get("sources", []):
                if s.get("path", "").endswith(".py"):
                    kernel_source = s["content"]
                    break

    findings = analyze(kernel_source, solution, args.policy, args.entry)
    v = verdict(findings)
    if args.json:
        print(json.dumps({"verdict": v, "findings": [f.as_dict() for f in findings]}, indent=2))
    else:
        if not findings:
            print("OK: no anti-cheat findings")
        for f in findings:
            print(f"[{f.severity}] {f.code} (line {f.lineno}): {f.message}")
        print(f"\nverdict: {v}")
    return exit_code(findings)


if __name__ == "__main__":
    raise SystemExit(main())
