"""Source lookup contracts, including alias extraction and the NL front door."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import source_search
import query_nl
from test_query_nl import fake_bridge, stub_intent, run_front_door


class SourceSearchTests(unittest.TestCase):
    def test_alias_spelling_lines_and_symlink_mount(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            repo = root / 'repo'
            repo.mkdir()
            code = '\n'.join(['# header', 'def split_kv(x):', '    return x', '# end'])
            (repo / 'attention.py').write_text(code)
            refs = root / 'references'
            refs.mkdir()
            (refs / 'nvidia').symlink_to(repo, target_is_directory=True)
            (repo / 'cycle').symlink_to(repo, target_is_directory=True)
            (root / 'outside.py').write_text('split_kv secret')
            (repo / 'escape.py').symlink_to(root / 'outside.py')
            with patch.dict(os.environ, {}, clear=True):
                result = source_search.search({'mechanisms': [{'name': 'KV splitting', 'aliases': ['split-KV']}]}, refs)
            self.assertEqual(len(result['source_matches']), 1)
            hit = result['source_matches'][0]
            self.assertEqual(hit['path'], 'gpu-wiki/references/nvidia/attention.py')
            self.assertEqual(hit['match_line'], 2)
            self.assertEqual(hit['snippet'], code)
            self.assertEqual(hit['matched_terms'], ['split-KV'])
            self.assertEqual(result['source_search']['files_skipped'], 1)

    def test_source_hits_reach_natural_language_response(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            (repo / 'gdn.py').write_text('def splitKV():\n    return 1\n')
            intent = stub_intent(mechanisms=[{'name': 'split-KV', 'aliases': []}])
            with patch.dict(os.environ, {'ATREX_WIKI_REFERENCE_PATHS': json.dumps({'nvidia': d})}), fake_bridge(intent) as bridge:
                code, out, err = run_front_door('Find a related implementation for KV parallelism')
            self.assertEqual(code, 0, err)
            response = json.loads(out)
            self.assertEqual(len(bridge.prompts), 1)
            self.assertEqual(response['source_matches'][0]['match_line'], 1)
            self.assertEqual(response['source_search']['terms'], ['split-KV'])
            self.assertIn('records', response)

    def test_absence_and_no_mechanism_are_explicit(self):
        with patch.dict(os.environ, {'ATREX_WIKI_REFERENCE_PATHS': '{}'}):
            self.assertEqual(source_search.search({}, Path('.'))['source_search']['status'], 'not_configured')
        for term in ['split-KV', 'split_kv', 'splitKV', 'split kv']:
            self.assertTrue(source_search.pattern(term).search('void split_kv()'))
        with self.assertRaises(ValueError):
            query_nl.strictly_validate_intent(stub_intent(mechanisms=[{'name': 'x', 'aliases': 'bad'}]))
