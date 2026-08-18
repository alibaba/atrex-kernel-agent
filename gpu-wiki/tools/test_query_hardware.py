#!/usr/bin/env python3
"""Tests for the hardware store's lookup tool.

These lock in what makes this store different from the experience wiki: an
address either resolves exactly or fails loudly, a number never travels without
its evidence class, and a declared-missing field returns instructions rather
than a substitute value.
"""
from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import query_hardware as query  # noqa: E402
import hardware_identity  # noqa: E402

STORE = HERE.parent / "hardware_wiki"
STORE_OK = (STORE / "records" / "index.json").is_file()


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = query.main(list(argv))
    except SystemExit as exc:
        code = exc.code
    return code, out.getvalue(), err.getvalue()


def payload(out):
    return json.loads(out)


@unittest.skipUnless(STORE_OK, "store not built")
class AddressingTests(unittest.TestCase):
    def test_identity_table_matches_every_recorded_product(self):
        index = query.load_index(STORE)["records"]
        recorded = {entry["product"] for entry in index
                    if entry["type"] == "spec-sheet" and entry.get("product")}
        self.assertEqual(recorded, set(hardware_identity.RECORDED_PRODUCTS))

    def test_all_product_families_normalize_to_internal_addresses(self):
        examples = {
            "NVIDIA A100 GPU": "a100",
            "NVIDIA L40S accelerator": "l40s",
            "NVIDIA H200 GPU": "h200",
            "NVIDIA B300 GPU": "b300",
            "NVIDIA GB300 GPU": "gb300",
            "NVIDIA RTX-5090 GPU": "rtx5090",
            "NVIDIA RTX PRO 5000 GPU": "rtxpro5000",
            "AMD Instinct MI300X GPU": "mi300x",
            "AMD MI308X accelerator": "mi308x",
            "AMD MI355X GPU": "mi355x",
            "SM_120": "sm120",
            "NVIDIA RTX-5090 GPU": "rtx5090",
        }
        for external, internal in examples.items():
            with self.subTest(external=external):
                self.assertEqual(hardware_identity.normalize_product_name(external),
                                 internal)

    def test_architecture_name_is_not_coerced_to_one_product(self):
        self.assertEqual(hardware_identity.normalize_product_name("gfx942"), "gfx942")
        code, _out, err = run("--product", "gfx942")
        self.assertEqual(code, 2)
        self.assertIn("unknown-product", err)

    def test_distinct_superchip_is_not_borrowed_from_gpu_sheet(self):
        self.assertEqual(hardware_identity.normalize_product_name("GB200"), "gb200")
        code, out, _err = run("--product", "GB200")
        self.assertEqual(code, query.NOT_RECORDED_EXIT)
        self.assertEqual(json.loads(out)["product"], "gb200")

    def test_product_listing_exposes_format_only_normalization(self):
        code, out, _err = run("--list", "products")
        self.assertEqual(code, 0)
        doc = json.loads(out)
        self.assertFalse(doc["normalization"]["case_sensitive"])
        self.assertFalse(doc["normalization"]["identity_translation"])
        self.assertIn("-", doc["normalization"]["ignored_separators"])
        self.assertIn("h100", doc["recognized_without_spec_sheet"])

    def test_unknown_non_product_identifier_is_never_translated(self):
        value = "private-pool-token"
        self.assertEqual(hardware_identity.normalize_product_name(value),
                         "privatepooltoken")
        code, out, err = run("--product", value)
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("unknown-product", err)

    def test_unknown_product_fails_loud_and_lists_known(self):
        code, out, err = run("--product", "zzpart9")
        self.assertEqual(code, 2)
        self.assertIn("unknown-product", err)
        self.assertIn("b300", err)
        self.assertEqual(out, "")

    def test_real_part_without_a_spec_sheet_returns_a_procedure(self):
        """The flow forbids fabricating specs, so a dead end is not an answer."""
        code, out, err = run("--product", "h20")
        self.assertEqual(code, query.NOT_RECORDED_EXIT)
        self.assertIn("not-recorded", err)
        doc = json.loads(out)
        self.assertEqual(doc["status"], "not-recorded")
        self.assertEqual(doc["architecture"], "hopper")
        self.assertIn("do not substitute", doc["do_not"])
        self.assertIn("peak_compute", doc["obtain_instead"])

    def test_not_recorded_answer_carries_no_other_parts_numbers(self):
        _c, out, _e = run("--product", "h20")
        doc = json.loads(out)
        self.assertNotIn("peak_compute", doc)
        self.assertNotIn("facts", doc)

    def test_product_name_is_case_insensitive(self):
        """The caller reads the part name off the driver, not out of this store."""
        code, out, _e = run("--product", "B300", "--field", "peak_compute.bf16.dense")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["field"], "peak_compute.bf16.dense")

    def test_vendor_qualified_recorded_product_is_addressable(self):
        code, out, _err = run("--product", "AMD Instinct MI308X GPU")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["identity"]["product"], "mi308x")

    def test_unknown_dtype_fails_loud(self):
        code, _out, err = run("--product", "b300", "--field", "peak_compute.bfloat16.dense")
        self.assertEqual(code, 2)
        self.assertIn("unknown-dtype", err)

    def test_misspelled_group_fails_loud(self):
        code, _out, err = run("--product", "b300", "--field", "memroy.l2_cache_mb")
        self.assertEqual(code, 2)
        self.assertIn("unknown-field", err)

    def test_unknown_capability_fails_loud(self):
        code, _out, err = run("--capability", "sm_999", "--list", "instructions")
        self.assertEqual(code, 2)
        self.assertIn("unknown-capability", err)

    def test_missing_store_fails_loud(self):
        with TemporaryDirectory() as tmp:
            code, _out, err = run("--store", tmp, "--product", "b300")
            self.assertEqual(code, query.NO_STORE_EXIT)
            self.assertIn("missing-store", err)

    def test_one_question_at_a_time(self):
        code, _out, err = run("--product", "b300", "--feature", "tma")
        self.assertEqual(code, 2)
        self.assertIn("one thing at a time", err)

    def test_field_requires_a_product(self):
        code, _out, err = run("--field", "peak_compute.bf16.dense")
        self.assertEqual(code, 2)
        self.assertIn("--field needs --product", err)

    def test_nothing_asked_is_an_error(self):
        code, _out, err = run()
        self.assertEqual(code, 2)
        self.assertIn("nothing asked", err)


@unittest.skipUnless(STORE_OK, "store not built")
class LookupTests(unittest.TestCase):
    def test_stdout_is_one_json_object(self):
        code, out, _err = run("--product", "b300")
        self.assertEqual(code, 0)
        record = payload(out)                       # would raise if polluted
        self.assertEqual(record["type"], "spec-sheet")

    def test_commentary_goes_to_stderr(self):
        _code, out, err = run("--product", "b300", "--field", "memory.capacity_gb")
        self.assertIn("hit:", err)
        self.assertNotIn("hit:", out)

    def test_field_answer_carries_provenance(self):
        _code, out, _err = run("--product", "b300", "--field", "peak_compute.fp4.dense")
        answer = payload(out)
        self.assertEqual(answer["value"], 13500)
        # Everything seeded from curated third-party docs is provisional; a
        # vendor-graded number only appears once someone records one.
        self.assertIn(answer["provenance"],
                      ("architecture-analysis", "derived-from-system-total",
                       "vendor-published"))

    @unittest.skip("this store records no per-field provenance_overrides")
    def test_per_field_override_beats_group_default(self):
        """L2 size is third-party analysis even though the memory group is published."""
        _c1, out1, _e1 = run("--product", "b300", "--field", "memory.l2_cache_mb")
        _c2, out2, _e2 = run("--product", "b300", "--field", "memory.bandwidth_tb_s")
        self.assertEqual(payload(out1)["provenance"], "architecture-analysis")
        self.assertEqual(payload(out2)["provenance"], "vendor-published")

    def test_declared_missing_field_returns_instructions_not_a_value(self):
        _code, out, _err = run("--product", "b300", "--field",
                               "compute_units.shared_memory_kb_per_sm")
        answer = payload(out)
        self.assertIsNone(answer["value"])
        self.assertIn("device attribute", answer["unavailable"])

    @unittest.skip("this store records no cross-part deltas_vs block")
    def test_deltas_state_the_counterintuitive_regressions(self):
        _code, out, _err = run("--product", "b300", "--vs", "b200")
        changes = {c["field"]: c for c in payload(out)["changes"]}
        self.assertLess(changes["peak_compute.int8.dense"]["delta_pct"], 0)
        self.assertLess(changes["peak_compute.fp64.dense"]["delta_pct"], 0)
        self.assertGreater(changes["peak_compute.fp4.dense"]["delta_pct"], 0)

    def test_unrecorded_comparison_fails_loud(self):
        code, _out, err = run("--product", "b300", "--vs", "mi300x")
        self.assertEqual(code, 2)
        self.assertIn("no-recorded-deltas", err)

    @unittest.skip("sources here state syntax but no closed modifier vocabulary")
    def test_instruction_syntax_is_verbatim_and_modifiers_are_closed(self):
        _code, out, _err = run("--instruction", "tcgen05.ld")
        facts = payload(out)["facts"]
        self.assertTrue(any(s.startswith("tcgen05.ld")
                            for s in facts["syntax"]))
        self.assertEqual(facts["modifiers"]["redOp"], [".min", ".max"])

    def test_instruction_states_where_it_does_not_exist(self):
        _code, out, _err = run("--instruction", "tcgen05.ld")
        avail = payload(out)["identity"]["availability"]
        self.assertTrue(avail["sm_arch"])
        self.assertIsInstance(avail["sm_arch"], list)
        self.assertTrue(avail["sm_arch"],
                        "an instruction must say where it exists")

    def test_feature_lookup(self):
        _code, out, _err = run("--feature", "tma")
        record = payload(out)
        self.assertEqual(record["type"], "arch-feature")
        self.assertTrue(record["facts"].get("parameters"),
                        "a feature must state its parameters")


@unittest.skipUnless(STORE_OK, "store not built")
class NoRankingTests(unittest.TestCase):
    def test_no_worth_or_ranking_anywhere(self):
        """A hardware fact has no importance score and no tier."""
        _code, out, _err = run("--product", "b300")
        self.assertNotIn("worth", payload(out))
        for banned in ("--rank", "--tier", "--min-importance", "--fallback-ratio"):
            code, _o, err = run("--product", "b300", banned)
            self.assertNotEqual(code, 0, banned)
            self.assertIn("unrecognized arguments", err)

    def test_zero_result_never_returns_a_sample(self):
        code, out, err = run("--product", "b999")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("unknown-product", err)


if __name__ == "__main__":
    unittest.main()
