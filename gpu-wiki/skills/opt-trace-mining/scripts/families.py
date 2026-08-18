#!/usr/bin/env python3
"""Operator naming: raw trace name -> record slug + workload family.

This is the corpus-specific file. Porting the skill to a different trace archive
means editing here and nowhere else, the same convention the sibling
`session-trace-mining` skill uses for its own `families.py`.

Self-contained on purpose. The predecessor imported these tables from a module
in another tree, which coupled this skill's output *paths* to code that could
move without warning -- and a change there would silently refile records,
because the layout gate derives a record's directory from these two functions.
The tables therefore live here, with the two behaviours that matter pinned by
`self_test()`:

  slugify("kernel_opt_002_gqa_paged_decode_h32_kv8_d128") == "gqa-paged-decode"
  family_of("flash_attention") == "attention"

The first keeps a trace-derived record filed under the same operator name the
store already uses for that operator, so a second run joins the first instead of
starting a near-duplicate namespace. The second decides the last directory level
and the L1 sibling-transfer key, so it may only return a value the schema's
`generality.workload_family` enum allows -- which `self_test()` checks against
the schema file itself rather than against a copy of the enum.

Run `python3 families.py` for that self-test.
"""
import re

# The closed vocabulary of schema.json `retrieval.generality.workload_family`.
# Mirrored here so this module stays importable without reading the schema;
# self_test() proves the mirror is still faithful.
WORKLOAD_FAMILIES = ("attention", "conv-vision", "decoder-layer",
                     "gemm-projection", "mask-index", "misc", "mlp-activation",
                     "moe", "norm", "rope", "ssm-linear-attention", "any")

# Operators whose optimisation story is the same technique set applied to a
# different shape or dtype collapse onto one name, so two traces of "the same"
# kernel share an operator_family and their records can be compared.
MERGE_RULES = (
    (re.compile(r"^gemm_n\d+_k\d+$"), "dense-gemm-shape-sweep"),
    (re.compile(r"^(fused_add_)?rmsnorm_h\d+$"), "rmsnorm-and-fused-add-rmsnorm"),
    (re.compile(r"^gqa_paged_decode_"), "gqa-paged-decode"),
    (re.compile(r"^gqa_paged_prefill_"), "gqa-paged-prefill-causal"),
    (re.compile(r"^gqa_ragged_prefill_"), "gqa-ragged-prefill-causal"),
    (re.compile(r"^mla_paged_decode_"), "mla-paged-decode"),
    (re.compile(r"^mla_paged_prefill_"), "mla-paged-prefill-causal"),
    (re.compile(r"^moe_fp8_block_scale_"), "moe-fp8-block-scale-routing"),
)

# A slug should read like a name, not like a directory listing, so the long words
# that recur across every operator are shortened and the grammar words dropped.
SLUG_ABBREV = {
    "attention": "attn", "backward": "bwd", "forward": "fwd",
    "projection": "proj", "embedding": "embed", "normalization": "norm",
    "position": "pos", "positional": "pos", "computation": "compute",
    "calculation": "compute", "multihead": "mha", "multimodal": "mm",
    "convolution": "conv", "residual": "res", "distribution": "dist",
    "hypernetwork": "hypernet", "aggregation": "agg", "accumulation": "accum",
    "variable": "var", "sequence": "seq", "generation": "gen",
    "preparation": "prep",
}
SLUG_DROP = {"with", "and", "the", "a", "of", "for", "to", "using", "based"}
# Five words is what keeps `gqa-paged-decode-h32-kv8` from becoming the name.
SLUG_MAX_WORDS = 5

# Path segments that hold operators rather than name one. Stripped from the front
# so a trace directory that carries its parent path in its name still yields an
# operator name and not a path -- which the anonymisation gate would reject.
CONTAINER_SEGMENTS = {"root", "home", "users", "user", "workspace", "work",
                      "projects", "project", "src", "repos", "repo", "code",
                      "tmp", "mnt", "data", "opt", "traces", "trace", "kernels"}

# First match wins, so the specific families lead and the broad ones follow.
# `attention` deliberately sits below `ssm-linear-attention`: a gated-delta-net
# kernel mentions attention but is not one.
FAMILY_RULES = (
    ("mask-index", ("mask_prep", "attention_mask", "hybrid_attention_mask",
                    "scatter", "gather", "index_add", "token_repeat",
                    "position_computation", "grid_based_indexing",
                    "cu_seqlens_var")),
    ("moe", ("moe", "expert", "topk_routing", "group_limited")),
    ("rope", ("rope", "rotary", "inverse_frequency",
              "position_embedding_generation", "multi_axis_rope", "yarn")),
    ("norm", ("rmsnorm", "rms_norm", "layernorm", "layer_norm", "groupnorm",
              "group_norm", "instance_normalization", "grn", "altup",
              "modulation")),
    ("ssm-linear-attention", ("mamba", "ssm", "selective_scan", "segsum",
                              "hyena", "gated_delta", "gdn", "linear_attention",
                              "chunk_scan", "fft", "rfft", "residual_coupling",
                              "flow_block")),
    ("decoder-layer", ("decoder_layer", "decoder_complete", "encoder_layer",
                       "transformer_block", "joint_transformer", "full_block",
                       "complete_forward", "layer_stack", "residual_path",
                       "prenorm", "text_decoder_layer",
                       "language_model_decoder")),
    ("conv-vision", ("conv", "vae", "resnet", "unet", "convnext", "patch_embed",
                     "upsample", "pyramid", "downsampling", "denoising",
                     "window_partition")),
    ("attention", ("attention", "attn", "gqa", "mla", "flash_attn", "prefill",
                   "decode", "softmax_dropout", "qk_norm",
                   "two_way_transformer")),
    ("gemm-projection", ("gemm", "matmul", "projection", "lm_head", "linear",
                         "hypernetwork", "cublas", "cutlass_gemm")),
    ("mlp-activation", ("mlp", "swiglu", "gelu", "silu", "ffn", "feedforward",
                        "activation", "softmax")),
)


def normalise(name):
    """Strip the bookkeeping a trace directory carries around the operator.

    Three shapes occur: the campaign counter (`kernel_opt_001_`), a directory
    name flattened from a path (`-root-work-kernel-opt-007-...`), and leading
    punctuation. Container segments have to go or the operator "name" becomes a
    path -- and it names no operator either way.
    """
    name = re.sub(r"^[-_.]+", "", name or "")
    name = re.sub(r"^.*?kernel[-_]opt[-_]\d+[-_]", "", name)
    name = re.sub(r"^kernel_opt_\d+_", "", name)
    name = re.sub(r"^\d+[-_]", "", name)
    name = re.sub(r"[^\w]+", "_", name).strip("_")
    parts = name.split("_")
    while len(parts) > 1 and parts[0].lower() in CONTAINER_SEGMENTS:
        parts.pop(0)
    return "_".join(parts)


def slugify(op_name):
    """`gqa_paged_decode_h32_kv8_d128_ps1` -> `gqa-paged-decode`."""
    op_name = op_name or "unknown"
    for rx, slug in MERGE_RULES:
        if rx.match(op_name):
            return slug
    words = [w for w in re.split(r"[_\-]+", op_name.lower()) if w]
    # A bare number never names an operator: a trace directory called `007` is a
    # campaign counter, and `unknown` is the honest slug for it.
    words = [w for w in words if not w.isdigit()]
    words = [SLUG_ABBREV.get(w, w) for w in words]
    words = [w for w in words if w not in SLUG_DROP]
    return "-".join(words[:SLUG_MAX_WORDS]) or "unknown"


def family_of(op_name):
    """The workload family, or `misc` when nothing matches.

    `misc` is a real answer, not a failure: a trace whose directory names no
    recognisable operator still produces valid records, and filing them under a
    guessed family would make the L1 sibling transfer return the wrong kernels.
    """
    low = (op_name or "").lower()
    for family, keys in FAMILY_RULES:
        for key in keys:
            if key in low:
                return family
    return "misc"


def technique_tag(text, limit=40):
    """A short technique label from free prose, for `retrieval.technique_tags`."""
    words = [w for w in re.split(r"[^a-z0-9]+", (text or "").lower()) if w]
    words = [w for w in words if w not in SLUG_DROP]
    tag = "-".join(words[:4])
    return tag[:limit].rstrip("-") or "unnamed"


def self_test():
    """The behaviours the store's layout depends on."""
    bad = []
    cases = (
        # (raw trace directory name, expected slug, expected family)
        ("kernel_opt_002_gqa_paged_decode_h32_kv8_d128",
         "gqa-paged-decode", "attention"),
        ("kernel_opt_001_example_fused_softmax_h4096",
         "example-fused-softmax-h4096", "mlp-activation"),
        ("flash_attention", "flash-attn", "attention"),
        ("hyena_fft_size_padding_rfft", "hyena-fft-size-padding-rfft",
         "ssm-linear-attention"),
        ("rmsnorm_h8192", "rmsnorm-and-fused-add-rmsnorm", "norm"),
        ("moe_fp8_block_scale_routing_e128", "moe-fp8-block-scale-routing",
         "moe"),
        ("-root-work-traces-kernel-opt-042-causal-conv1d",
         "causal-conv1d", "conv-vision"),
        ("007", "unknown", "misc"),
    )
    for raw, want_slug, want_family in cases:
        name = normalise(raw)
        got = (slugify(name), family_of(name))
        if got != (want_slug, want_family):
            bad.append("%s -> %s, expected %s"
                       % (raw[:44], got, (want_slug, want_family)))

    # Every family this table can emit must be a value the schema allows, or the
    # record is filed in a directory the retrieval engine will never look in.
    declared = {f for f, _keys in FAMILY_RULES} | {"misc"}
    unknown = sorted(declared - set(WORKLOAD_FAMILIES))
    if unknown:
        bad.append("FAMILY_RULES emits %s, absent from WORKLOAD_FAMILIES"
                   % unknown)
    try:
        import json

        import config as c

        enum = json.loads(c.SCHEMA_PATH.read_text())["properties"]["retrieval"][
            "properties"]["generality"]["properties"]["workload_family"]["enum"]
        drifted = sorted(set(WORKLOAD_FAMILIES) ^ set(enum))
        if drifted:
            bad.append("WORKLOAD_FAMILIES has drifted from %s: %s"
                       % (c.SCHEMA_PATH.name, drifted))
    except Exception as exc:                                    # noqa: BLE001
        bad.append("could not check the enum against the schema: %r" % exc)
    return bad, len(cases) + 2


if __name__ == "__main__":
    import sys

    failures, n = self_test()
    for line in failures:
        print("FAIL %s" % line)
    print("families: %d checks, %d failures" % (n, len(failures)))
    sys.exit(1 if failures else 0)
