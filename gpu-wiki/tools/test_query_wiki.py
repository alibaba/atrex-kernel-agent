#!/usr/bin/env python3
"""Tests for the kernel-experience retrieval tool.

Focused on the store's own contract -- scope fail-closed, tier/gain filters,
ranking by worth, and the serve projection stripping -- with no dependency on
any host or markdown backend.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import query_wiki as query  # noqa: E402

STORE = HERE.parent / "kernel_wiki"
IMP_RE = re.compile(r"imp=([0-9.]+)")


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = query.main(list(argv))
    except SystemExit as exc:
        code = exc.code
    return code, out.getvalue(), err.getvalue()


def served(out):
    """--emit-json returns an id-keyed envelope; preserve insertion/rank order."""
    doc = json.loads(out)
    return [dict(value, _id=rid) for rid, value in doc["records"].items()]


def result(out):
    """The envelope's self-description: match kind, counts, budget."""
    return json.loads(out)["result"]


def records(out):
    """Compact record views used by ranking tests."""
    return [{"id": e["_id"], "type": e["type"], "payload": e["payload"]}
            for e in served(out)]


def lines(out):
    return [l for l in out.splitlines() if l.startswith("  [")]


_INDEX: dict = {}


def _index() -> list[dict]:
    """The engine-side index; tests use it to check what is NOT served."""
    if "entries" not in _INDEX:
        path = STORE / "records" / "index.json"
        _INDEX["entries"] = json.loads(path.read_text())["records"]
    return _INDEX["entries"]


def index_field(name: str) -> dict:
    return {e["id"]: e.get(name) for e in _index()}


def stored_record(record_id: str) -> dict:
    path = {e["id"]: e["path"] for e in _index()}[record_id]
    return json.loads((STORE / path).read_text())


def all_keys(value) -> set:
    """Every dict key anywhere in a parsed JSON value."""
    found: set = set()
    stack = [value]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            found.update(node)
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return found


STORE_OK = (STORE / "records" / "index.json").is_file()


# ---------------------------------------------------------- corpus discovery
# These tests assert retrieval BEHAVIOUR, not this particular corpus. Two stores
# share this tool and hold different architectures, so the fixtures are derived
# from whichever store is present; a property the corpus cannot exercise is
# skipped rather than silently asserted against the wrong thing.

def _scopes() -> list[dict]:
    return [e["retrieval"]["scope"] for e in _index()] if STORE_OK else []


def _primary_arch() -> str | None:
    """The non-neutral architecture with the most records."""
    counts: dict[str, int] = {}
    for s in _scopes():
        a = s.get("arch")
        if a and a != query.NEUTRAL_ARCH:
            counts[a] = counts.get(a, 0) + 1
    return max(counts, key=counts.get) if counts else None


def _runtime_token(arch: str | None) -> str | None:
    """A token the runtime would report that resolves to this architecture."""
    if not arch:
        return None
    for token, family in query.ARCH_ALIASES.items():
        if family == arch and token.startswith("sm"):
            return token
    return next((t for t, f in query.ARCH_ALIASES.items() if f == arch), None)


def _sibling_arch(arch: str | None) -> str | None:
    """Another architecture of the same vendor, if this store has one."""
    if not arch:
        return None
    vendor = query.ARCH_VENDOR.get(arch)
    others = {s.get("arch") for s in _scopes()
              if s.get("vendor") == vendor and s.get("arch") not in (arch, query.NEUTRAL_ARCH)}
    return sorted(others)[0] if others else None


def _neutral_arch_present(arch: str | None) -> bool:
    vendor = query.ARCH_VENDOR.get(arch or "")
    return any(s.get("arch") == query.NEUTRAL_ARCH
               and s.get("vendor") in (vendor, "generic")
               for s in _scopes())


def vocab_of(key: str) -> set:
    """One dimension of the store's own vocabulary, for corpus-neutral fixtures."""
    return query.vocab(_index())[key] if STORE_OK else set()


def _busiest_dsl(arch: str | None) -> str | None:
    counts: dict[str, int] = {}
    for sc in _scopes():
        d = sc.get("dsl")
        if d and d != "any":
            counts[d] = counts.get(d, 0) + 1
    return max(counts, key=counts.get) if counts else None


def _other_vendor_arch(arch: str | None) -> str | None:
    """An architecture belonging to a DIFFERENT vendor, for leak tests."""
    vendor = query.ARCH_VENDOR.get(arch or "")
    others = {sc.get("arch") for sc in _scopes()
              if sc.get("vendor") not in (vendor, "generic", None)
              and sc.get("arch") not in (None, query.NEUTRAL_ARCH)}
    return sorted(others)[0] if others else None


def _cross_arch_pair() -> tuple[str, str] | None:
    """An (arch, symptom) with no local match but a sibling-architecture one.

    This is the situation the cross-arch widening exists for. Only some corpora
    contain it, so the tests that need it skip rather than assert falsely.
    """
    by_arch: dict[str, set] = {}
    for e in _index():
        sc = e["retrieval"]["scope"]
        sig = e["retrieval"].get("signals", {})
        symptoms = set(sig.get("symptoms") or [])
        if sig.get("bottleneck"):
            symptoms.add(sig["bottleneck"])
        by_arch.setdefault(sc.get("arch"), set()).update(symptoms)
    for arch, symptoms in by_arch.items():
        if not arch or arch == query.NEUTRAL_ARCH:
            continue
        vendor = query.ARCH_VENDOR.get(arch)
        for other, other_symptoms in by_arch.items():
            if other in (arch, None, query.NEUTRAL_ARCH):
                continue
            if query.ARCH_VENDOR.get(other) != vendor:
                continue
            for symptom in sorted(other_symptoms - symptoms):
                return arch, symptom
    return None


ARCH = _primary_arch() if STORE_OK else None
RUNTIME_ARCH = _runtime_token(ARCH)
SIBLING_ARCH = _sibling_arch(ARCH)
HAS_NEUTRAL = _neutral_arch_present(ARCH) if STORE_OK else False
ARCH_OK = STORE_OK and bool(RUNTIME_ARCH)
DSL = _busiest_dsl(ARCH) if STORE_OK else None
OTHER_VENDOR_ARCH = _other_vendor_arch(ARCH) if STORE_OK else None
CROSS_PAIR = _cross_arch_pair() if STORE_OK else None


@unittest.skipUnless(STORE_OK, "store not built")
class ScopeTests(unittest.TestCase):
    def test_unknown_arch_fails_closed(self):
        code, _o, err = run("--arch", "zzarch9")
        self.assertEqual(code, 2)
        self.assertIn("unknown-arch", err)

    @unittest.skipUnless(ARCH_OK, "no architecture alias for this corpus")
    def test_runtime_capability_token_resolves(self):
        """The caller only knows what the runtime told it: sm_90, not "hopper"."""
        code, out, err = run("--arch", RUNTIME_ARCH, "--emit-json")
        self.assertEqual(code, 0)
        self.assertIn("arch=%s" % ARCH, err)
        self.assertEqual(json.loads(out)["query"]["arch"], ARCH)

    def test_known_arch_without_records_is_not_reported_as_a_bad_query(self):
        absent = next((t for t, f in query.ARCH_ALIASES.items()
                       if f not in {s.get("arch") for s in _scopes()}), None)
        if not absent:
            self.skipTest("this corpus covers every known architecture")
        code, _o, err = run("--arch", absent)
        self.assertEqual(code, 2)
        self.assertIn("no-records-for-arch", err)
        self.assertNotIn("unknown-arch", err)

    @unittest.skipUnless(ARCH_OK, "no architecture alias for this corpus")
    def test_alias_and_store_name_agree(self):
        _c, a, _e = run("--arch", RUNTIME_ARCH, "--type", "doc", "--emit-json")
        _c, b, _e = run("--arch", ARCH, "--type", "doc", "--emit-json")
        self.assertEqual([r["id"] for r in records(a)],
                         [r["id"] for r in records(b)])

    def test_unknown_type_fails_closed(self):
        code, _o, err = run("--type", "recipe")
        self.assertEqual(code, 2)
        self.assertIn("unknown-type", err)

    def test_unknown_tier_rejected_by_argparse(self):
        code, _o, _e = run("--tier", "gold")
        self.assertNotEqual(code, 0)

    def test_missing_store_fails_closed(self):
        with TemporaryDirectory() as tmp:
            code, _o, err = run("--json-store", tmp, "gemm")
            self.assertEqual(code, query.NO_STORE_EXIT)
            self.assertIn("missing-store", err)

    def test_dsl_filter_excludes_other_dsl(self):
        code, out, _e = run("--dsl", "triton", "--emit-json", "--limit", "50")
        self.assertEqual(code, 0)
        for e in served(out):
            self.assertIn(e["applies_to"]["dsl"], ("triton", "any"))

    def test_supported_dsl_without_exact_records_reaches_portable_records(self):
        code, out, err = run("--arch", "zwm890p", "--dsl", "tilelang",
                             "--operator", "flash-attention", "--emit-json",
                             "--limit", "50")
        self.assertEqual(code, 0, err)
        self.assertIn("ppu.zwm890p.any.attention.fa-sail",
                      {record["id"] for record in records(out)})

    def test_operator_filter_keeps_operator_agnostic_ppu_documents(self):
        expected = {
            "cuda": "ppu.zwm890p.cuda.any.hggc-sailify",
            "triton": "ppu.zwm890p.triton.any.triton-for-sail",
        }
        for dsl, record_id in expected.items():
            with self.subTest(dsl=dsl):
                code, out, err = run("--arch", "zwm890p", "--dsl", dsl,
                                     "--operator", "flash-attention", "--emit-json",
                                     "--limit", "50")
                self.assertEqual(code, 0, err)
                self.assertIn(record_id, {record["id"] for record in records(out)})

    def test_type_filter_is_exclusive(self):
        code, out, _e = run("--type", "anti-strategy", "--emit-json", "--limit", "30")
        self.assertEqual(code, 0)
        self.assertTrue(records(out))
        for r in records(out):
            self.assertEqual(r["type"], "anti-strategy")


@unittest.skipUnless(STORE_OK, "store not built")
class RankFilterTests(unittest.TestCase):
    def test_filter_only_is_ordered_by_importance(self):
        code, out, _e = run("--type", "technique-card", "--limit", "25")
        self.assertEqual(code, 0)
        scores = [float(m) for m in IMP_RE.findall(out)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_min_importance_drops_low(self):
        code, out, _e = run("--min-importance", "0.9", "--limit", "40")
        self.assertEqual(code, 0)
        for m in IMP_RE.findall(out):
            self.assertGreaterEqual(float(m), 0.9)

    def test_tier_keeps_only_that_standing(self):
        code, out, _e = run("--tier", "cautionary", "--limit", "40")
        self.assertEqual(code, 0)
        self.assertTrue(lines(out))
        for l in lines(out):
            self.assertIn("cautionary", l)

    def test_min_gain_needs_a_number(self):
        gains = index_field("gain_pct")
        if not any(isinstance(v, (int, float)) for v in gains.values()):
            self.skipTest("no record states a measured gain yet; --min-gain has "
                          "nothing to admit until mined records land")
        code, out, _e = run("--min-gain", "25", "--emit-json", "--limit", "40")
        self.assertEqual(code, 0)
        self.assertTrue(records(out))
        for r in records(out):
            self.assertGreaterEqual(gains.get(r["id"]) or -1, 25.0, r["id"])

    def test_negative_weight_rejected(self):
        code, _o, err = run("--weight-importance", "-1", "fusion")
        self.assertEqual(code, 2)

    def test_importance_cannot_outrank_much_stronger_text(self):
        code, out, _e = run("fusion", "--explain", "--limit", "40")
        self.assertEqual(code, 0)
        blends = [float(m.group(1)) for m in re.finditer(r"blend=([0-9.]+)", out)]
        self.assertEqual(blends, sorted(blends, reverse=True))


@unittest.skipUnless(STORE_OK, "store not built")
class ServeProjectionTests(unittest.TestCase):
    def _served(self, *argv):
        code, out, _e = run("--emit-json", *argv)
        self.assertEqual(code, 0)
        entries = served(out)
        self.assertTrue(entries)
        return entries

    def test_entry_has_the_declared_serving_shape(self):
        for e in self._served("--limit", "40"):
            self.assertEqual(sorted(e), ["_id", "applies_to", "match", "payload",
                                         "source", "type"])

    def test_engine_fields_never_served(self):
        code, out, _e = run("--emit-json", "--limit", "40")
        self.assertEqual(code, 0)
        # Structural, not substring: the word "worth" occurs in prose too. Only the
        # layer name is banned recursively -- that also covers worth.track and the
        # score decomposition. Evidence is separately asserted absent below.
        self.assertNotIn("worth", all_keys(json.loads(out)))
        self.assertNotIn("evidence", all_keys(json.loads(out)))
        for e in served(out):
            self.assertNotIn("locator", e["applies_to"], e["_id"])
            self.assertNotIn("links", e["applies_to"], e["_id"])

    def test_payload_has_no_ids_or_paths(self):
        for e in self._served("--limit", "60"):
            blob = json.dumps({"payload": e["payload"]}, ensure_ascii=False)
            impl = e["payload"].get("implementation") or {}
            for key in ("snippet", "dispatch_snippet"):
                if impl.get(key):
                    blob = blob.replace(json.dumps(impl[key], ensure_ascii=False), '""')
            self.assertNotRegex(blob, r"nvidia\.(?:b200|b300|any)\.[a-z0-9-]+\.", e["_id"])
            # This store has no markdown tree, so a page path in an
            # agent-facing layer is a citation the agent can never open.
            self.assertNotRegex(blob, r"[\w/-]+\.md\b", e["_id"])

    def test_no_contributor_or_corpus_path_leaks(self):
        """Assert leak SHAPES, not literal secrets.

        A committed denylist would publish the very names and corpus roots
        it guards. Point ATREX_WIKI_DENYLIST at a file of substrings (one
        per line) to enforce a private list in CI.
        """
        code, out, _e = run("--emit-json", "--limit", "40")
        self.assertEqual(code, 0)
        self.assertNotRegex(out, r"/(?:root|home|Users)/[\w.-]+")
        self.assertNotRegex(out, r"(?<![\w.@])[\w.+-]{2,}@[\w-]+\.[a-z]{2,}\b")
        denylist = os.environ.get("ATREX_WIKI_DENYLIST")
        if denylist and os.path.isfile(denylist):
            with open(denylist, encoding="utf-8") as handle:
                for line in handle:
                    term = line.strip()
                    if term and not term.startswith("#"):
                        self.assertNotIn(term, out, term)

    def test_gain_is_percentage_not_absolute_latency(self):
        """Guards store content: worth.gain is engine-side, so read it from disk."""
        time_unit = re.compile(r"^\s*(?:us|µs|\u03bcs|ms|s|ns|cycles?)\s*$", re.I)
        for e in self._served("--limit", "60"):
            record = stored_record(e["_id"])
            for entry in ((record["worth"].get("gain") or {}).get("metrics") or []):
                self.assertIsNone(time_unit.match(str(entry.get("unit") or "")),
                                  record["id"])
                if entry.get("metric") == "latency":
                    self.assertIsNone(entry.get("before"), record["id"])


@unittest.skipUnless(STORE_OK, "store not built")
class VocabularySpellingTests(unittest.TestCase):
    """A caller's spelling of a store token is not a scope error."""

    def test_punctuation_and_case_are_folded(self):
        dsls = sorted(vocab_of("dsl") - {"any"})
        if not dsls:
            self.skipTest("corpus has no language-scoped records")
        real = dsls[0]
        variants = {real.replace("-", ""), real.replace("-", "_"), real.upper()}
        _c, expected, _e = run("--dsl", real, "--emit-json", "--limit", "5")
        for spelling in variants:
            with self.subTest(spelling=spelling):
                code, got, _e = run("--dsl", spelling, "--emit-json", "--limit", "5")
                self.assertEqual(code, 0, spelling)
                self.assertEqual([r["id"] for r in records(got)],
                                 [r["id"] for r in records(expected)])

    def test_a_genuinely_unknown_token_still_fails_closed(self):
        code, _o, err = run("--dsl", "zznosuchlang")
        self.assertEqual(code, 2)
        self.assertIn("unknown-dsl", err)

    def test_a_rejected_token_comes_back_with_near_misses(self):
        families = sorted(vocab_of("family"))
        if not families:
            self.skipTest("corpus has no operator families")
        stem = families[0].split("-")[0]
        if stem == families[0]:
            self.skipTest("no compound family token to truncate")
        code, _o, err = run("--family", stem)
        self.assertEqual(code, 2)
        self.assertIn("did you mean", err)

    def test_a_long_vocabulary_is_not_dumped_into_the_error(self):
        """One mistyped family printed 157 tokens at a caller that needed 8."""
        families = sorted(vocab_of("family"))
        if len(families) <= query.VOCAB_DUMP_LIMIT:
            self.skipTest("this corpus's vocabulary is short enough to list")
        stem = families[0].split("-")[0]
        code, _o, err = run("--family", stem if stem != families[0] else "zzfam")
        self.assertEqual(code, 2)
        self.assertIn("known values", err)
        self.assertIn("--like", err)
        self.assertLess(len(err), 1200, "the error is itself a context flood")

    def test_list_can_be_filtered_because_a_dump_is_not_readable(self):
        families = sorted(vocab_of("family"))
        if not families:
            self.skipTest("corpus has no operator families")
        needle = families[0][:4]
        code, out, _e = run("--list-family", "--like", needle)
        self.assertEqual(code, 0)
        listed = [l for l in out.splitlines() if l.strip()]
        self.assertTrue(listed)
        for value in listed:
            self.assertIn(needle.lower(), value.lower())

    def test_free_form_signals_are_not_presented_as_selectable_tokens(self):
        """One store records symptoms as whole sentences; say so, do not imply a pick-list."""
        prose = [s for s in vocab_of("symptom") if " " in s]
        code, _o, err = run("--list-symptoms")
        self.assertEqual(code, 0)
        if prose:
            self.assertIn("free-form text", err)
        else:
            self.assertNotIn("free-form text", err)


@unittest.skipUnless(STORE_OK, "store not built")
class EnvelopeTests(unittest.TestCase):
    @unittest.skipUnless(ARCH_OK, "no architecture alias for this corpus")
    def test_fallback_is_labelled_in_the_answer_not_only_on_stderr(self):
        """An agent that captures stdout alone must not read a sample as advice."""
        code, out, err = run("--arch", RUNTIME_ARCH, "zzznosuchterm", "--emit-json")
        self.assertEqual(code, 0)
        res = result(out)
        self.assertEqual(res["kind"], "fallback")
        self.assertIn("NOT query matches", res["note"])
        self.assertIn("NOT query matches", err)

    @unittest.skipUnless(ARCH_OK, "no architecture alias for this corpus")
    def test_matches_are_labelled_as_matches(self):
        _c, out, _e = run("--arch", RUNTIME_ARCH, "--emit-json")
        res = result(out)
        self.assertEqual(res["kind"], "matches")
        self.assertIsNone(res["note"])

    @unittest.skipUnless(ARCH_OK, "no architecture alias for this corpus")
    def test_no_fallback_leaves_an_empty_answer_that_says_so(self):
        _c, out, _e = run("--arch", RUNTIME_ARCH, "zzznosuchterm", "--emit-json",
                          "--no-fallback")
        res = result(out)
        self.assertEqual(res["kind"], "empty")
        self.assertEqual(res["served"], 0)


@unittest.skipUnless(STORE_OK, "store not built")
class SymptomPrecisionTests(unittest.TestCase):
    """A recorded signal outranks a passing mention of the same words."""

    def _indexed_symptom(self):
        for s in sorted(vocab_of("symptom")):
            if " " not in s and len(s) <= 40:
                return s
        return None

    def test_an_indexed_symptom_serves_signal_matches_not_text_matches(self):
        symptom = self._indexed_symptom()
        if not symptom:
            self.skipTest("corpus indexes no token-shaped symptom")
        _c, out, _e = run("--symptom", symptom, "--emit-json", "--limit", "20")
        doc = json.loads(out)
        self.assertEqual(doc["query"]["symptom_match_mode"], "signal")
        for entry in served(out):
            record = stored_record(entry["_id"])
            sig = record["retrieval"]["signals"]
            recorded = set(sig.get("symptoms") or [])
            if sig.get("bottleneck"):
                recorded.add(sig["bottleneck"])
            self.assertIn(symptom, recorded, entry["_id"])

    def test_a_text_only_symptom_still_answers_but_says_so(self):
        _c, out, err = run("--symptom", "zzznosuchsignal", "--emit-json")
        doc = json.loads(out)
        self.assertEqual(doc["query"]["symptom_match_mode"], "text")
        self.assertIn("not an indexed symptom", doc["query"]["symptom_note"])

    def test_a_rejected_symptom_offers_structured_alternatives(self):
        """A caller that guessed wrong wants data it can act on, not just prose."""
        symptom = self._indexed_symptom()
        if not symptom or len(symptom) < 6:
            self.skipTest("no symptom token to truncate")
        _c, out, _e = run("--symptom", symptom[:5], "--emit-json")
        alts = json.loads(out)["query"]["symptom_alternatives"]
        self.assertTrue(alts)
        self.assertTrue(all(isinstance(a, str) for a in alts))


@unittest.skipUnless(STORE_OK, "store not built")
class EmptyResultDiagnosisTests(unittest.TestCase):
    """A zero result must not conflate "no such knowledge" with "impossible filters"."""

    def _over_constrained(self):
        """A scope whose parts each match, but whose combination cannot."""
        types = sorted(vocab_of("type"))
        families = sorted(vocab_of("family"))
        if len(types) < 2 or not families:
            return None
        for family in families[:25]:
            for rtype in types:
                argv = ["--family", family, "--type", rtype, "--emit-json"]
                _c, out, _e = run(*argv)
                if result(out)["kind"] == "empty":
                    return argv
        return None

    def test_an_over_constrained_scope_names_the_guilty_filter(self):
        argv = self._over_constrained()
        if argv is None:
            self.skipTest("this corpus has no empty filter intersection to diagnose")
        _c, out, err = run(*argv)
        res = result(out)
        self.assertEqual(res["kind"], "empty")
        found = res["empty_because"]
        self.assertIn("over-constrained", found["diagnosis"])
        removals = found["single_filter_removals_that_would_match"]
        self.assertTrue(removals)
        for item in removals:
            self.assertGreater(item["would_match"], 0)
            self.assertTrue(item["drop"].startswith("--")
                            or "free-text" in item["drop"])
        self.assertIn("over-constrained", err)

    def test_the_named_removal_actually_matches(self):
        """The advice must be true: dropping that filter must really return records."""
        argv = self._over_constrained()
        if argv is None:
            self.skipTest("this corpus has no empty filter intersection to diagnose")
        _c, out, _e = run(*argv)
        best = result(out)["empty_because"][
            "single_filter_removals_that_would_match"][0]
        flag = best["drop"]
        if not flag.startswith("--"):
            self.skipTest("free-text removal is not a flag")
        reduced = list(argv)
        i = reduced.index(flag)
        del reduced[i:i + 2]
        _c, out2, _e = run(*reduced)
        self.assertEqual(result(out2)["served"], min(best["would_match"],
                                                    query.DEFAULT_LIMIT))

    def test_a_genuinely_absent_subject_is_not_blamed_on_a_filter(self):
        _c, out, _e = run("zzznosuchsubjectanywhere", "--emit-json", "--no-fallback")
        res = result(out)
        self.assertEqual(res["kind"], "empty")
        # No scope filter was set, so there is nothing to drop but the terms.
        removals = (res.get("empty_because") or {}).get(
            "single_filter_removals_that_would_match", [])
        self.assertTrue(all("free-text" in r["drop"] for r in removals), removals)


@unittest.skipUnless(STORE_OK, "store not built")
class NeutralRecallTests(unittest.TestCase):
    @unittest.skipUnless(ARCH_OK and HAS_NEUTRAL,
                         "this corpus has no architecture-neutral records")
    def test_neutral_records_widen_an_arch_scoped_pool(self):
        _c, wide, _e = run("--arch", RUNTIME_ARCH, "--emit-json", "--limit", "60")
        _c, strict, _e = run("--arch", RUNTIME_ARCH, "--strict-arch",
                             "--emit-json", "--limit", "60")
        self.assertGreater(result(wide)["scoped_pool"],
                           result(strict)["scoped_pool"])

    @unittest.skipUnless(ARCH_OK, "no architecture alias for this corpus")
    def test_every_hit_says_how_it_was_reached(self):
        _c, out, _e = run("--arch", RUNTIME_ARCH, "--emit-json", "--limit", "40")
        for e in served(out):
            self.assertIn(e["match"]["arch"], ("exact", "architecture-neutral"))
            if e["match"]["arch"] == "exact":
                scope = e["applies_to"]
                self.assertTrue(scope.get("arch") == ARCH
                                or ARCH in (scope.get("architectures") or []))

    @unittest.skipUnless(OTHER_VENDOR_ARCH, "corpus has only one vendor")
    def test_neutral_recall_never_crosses_the_vendor_line(self):
        want = query.ARCH_VENDOR[OTHER_VENDOR_ARCH]
        _c, out, _e = run("--arch", OTHER_VENDOR_ARCH, "--emit-json", "--limit", "60")
        for e in served(out):
            self.assertIn(e["applies_to"]["vendor"], (want, "generic"),
                          e["_id"])

    @unittest.skipUnless(ARCH_OK, "no architecture alias for this corpus")
    def test_strict_arch_keeps_only_exact_matches(self):
        _c, out, _e = run("--arch", RUNTIME_ARCH, "--strict-arch", "--emit-json",
                          "--limit", "40")
        for e in served(out):
            self.assertEqual(e["match"]["arch"], "exact")


@unittest.skipUnless(STORE_OK, "store not built")
class VolumeTests(unittest.TestCase):
    ARGV = ("--type", "technique-card", "--emit-json", "--limit", "8")

    def test_brief_is_smaller_and_announces_every_omission(self):
        """Trimming only code left prose-heavy records byte-for-byte unchanged."""
        _c, full, _e = run(*self.ARGV)
        _c, brief, _e = run(*self.ARGV, "--brief")
        self.assertLess(len(brief), len(full))
        for e in served(brief):
            payload = e["payload"]
            extra = set(payload) - set(query.BRIEF_KEYS) - {
                "implementation", "omitted_by_brief"}
            self.assertFalse(extra, extra)
            if "omitted_by_brief" in payload:
                self.assertTrue(payload["omitted_by_brief"]["fields"])

    def test_budget_drops_the_weakest_hits_and_reports_it(self):
        _c, out, _e = run(*self.ARGV, "--max-bytes", "9000")
        res = result(out)
        self.assertTrue(res.get("dropped_for_budget") or res.get("over_budget"))

    @unittest.skipUnless(CROSS_PAIR, "corpus has no sibling-architecture gap")
    def test_zero_match_reports_where_the_knowledge_actually_is(self):
        """Empirically, portable knowledge sat under a sibling architecture."""
        arch, symptom = CROSS_PAIR
        _c, out, err = run("--symptom", symptom, "--arch", arch, "--emit-json")
        res = result(out)
        self.assertEqual(res["kind"], "empty")
        found = res["available_elsewhere"]
        self.assertGreater(found["matches_on_other_architectures_of_this_vendor"], 0)
        self.assertIn("--cross-arch", err)
        self.assertIn("do NOT drop", err)

    @unittest.skipUnless(CROSS_PAIR, "corpus has no sibling-architecture gap")
    def test_cross_arch_widening_is_labelled_and_vendor_gated(self):
        arch, symptom = CROSS_PAIR
        want = query.ARCH_VENDOR.get(arch)
        _c, out, _e = run("--symptom", symptom, "--arch", arch,
                          "--cross-arch", "--emit-json")
        self.assertEqual(result(out)["kind"], "matches")
        for e in served(out):
            self.assertEqual(e["match"]["arch"], "other-architecture")
            self.assertIn(e["applies_to"]["vendor"], (want, "generic"))

    @unittest.skipUnless(OTHER_VENDOR_ARCH, "corpus has only one vendor")
    def test_cross_arch_never_crosses_the_vendor_line(self):
        want = query.ARCH_VENDOR[OTHER_VENDOR_ARCH]
        _c, out, _e = run("--arch", OTHER_VENDOR_ARCH, "--cross-arch",
                          "--emit-json", "--limit", "60")
        for e in served(out):
            self.assertIn(e["applies_to"]["vendor"], (want, "generic"),
                          e["_id"])

    def test_vendor_filter_keeps_vendor_neutral_knowledge(self):
        """--vendor X used to DELETE records scoped to no vendor at all."""
        vendors = sorted(v for v in vocab_of("vendor") if v != "generic")
        if not vendors:
            self.skipTest("corpus has no vendor-scoped records")
        want = vendors[0]
        _c, out, _e = run("--vendor", want, "--emit-json", "--limit", "60")
        got = {e["applies_to"]["vendor"] for e in served(out)}
        self.assertTrue(got <= {want, "generic"}, got)

    def test_human_mode_labels_a_fallback_on_stdout(self):
        """Plain mode is what an agent runs first; the label cannot be stderr-only."""
        _c, out, _e = run("zzznosuchterm")
        self.assertIn("NOT MATCHES", out)

    def test_human_mode_empty_scope_is_not_silent(self):
        _c, out, _e = run("zzznosuchterm", "--no-fallback")
        self.assertTrue(out.strip())

    @unittest.skipUnless(ARCH_OK, "no architecture alias for this corpus")
    def test_coverage_separates_an_empty_cell_from_a_full_pool(self):
        """A pool size alone let --dsl any make an empty cell look populated."""
        code, out, _e = run("--arch", RUNTIME_ARCH, "--coverage")
        self.assertEqual(code, 0)
        doc = json.loads(out)
        self.assertEqual(sum(doc["by_reach"].values()), doc["scoped_pool"])
        self.assertIn("exact", doc["by_reach"])

    def test_out_of_vocabulary_symptom_is_announced_not_silently_ignored(self):
        _c, out, err = run("--symptom", "zzznosuchsymptom", "--emit-json")
        self.assertIn("not an indexed symptom", err)
        self.assertIn("not an indexed symptom",
                      json.loads(out)["query"]["symptom_note"])

    def test_indexed_symptom_carries_no_warning(self):
        known = sorted(vocab_of("symptom"))
        if not known:
            self.skipTest("corpus indexes no symptoms")
        _c, out, err = run("--symptom", known[0], "--emit-json")
        self.assertNotIn("not an indexed symptom", err)
        self.assertIsNone(json.loads(out)["query"]["symptom_note"])

    def test_exclude_suppresses_what_the_caller_already_read(self):
        _c, first, _e = run("--emit-json", "--limit", "2")
        seen = [r["id"] for r in records(first)]
        _c, second, _e = run("--emit-json", "--limit", "2",
                             "--exclude", ",".join(seen))
        self.assertFalse(set(r["id"] for r in records(second)) & set(seen))


if __name__ == "__main__":
    unittest.main()
