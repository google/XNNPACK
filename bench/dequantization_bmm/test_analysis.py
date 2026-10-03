# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from summarize import classify, pair_run


class AnalysisTest(unittest.TestCase):

  def rows(self):
    return [
        dict(
            pass_=p,
            rewrite=mode,
            run_us=(20 if mode else 10) * (p + 1),
            invoke_us=30 if mode else 10,
        )
        for p in range(4)
        for mode in (0, 1)
    ]

  def normalized(self):
    rows = self.rows()
    for r in rows:
      r['pass'] = r.pop('pass_')
    return rows

  def test_ratio_direction(self):
    result = pair_run(self.normalized(), 4)
    self.assertEqual(result['run_us_ratio'], 2)
    self.assertEqual(result['invoke_us_ratio'], 3)
    self.assertEqual(classify([2, 2.1, 1.9], 0.05), 'prefer_off')
    self.assertEqual(classify([0.5, 0.6, 0.4], 0.05), 'prefer_on')

  def test_incomplete_and_duplicate(self):
    rows = self.normalized()
    with self.assertRaises(ValueError):
      pair_run(rows[:-1], 4)
    rows[-1] = rows[-2]
    with self.assertRaises(ValueError):
      pair_run(rows, 4)

  def test_uncertainty_and_repetition_count(self):
    self.assertEqual(classify([2, 2], 0.05), 'insufficient_repetitions')
    self.assertEqual(classify([1.2, 0.9, 1.3], 0.05), 'uncertain')
    self.assertEqual(classify([1.02, 1.03, 1.04], 0.05), 'uncertain')

  def test_invalid_timing(self):
    for value in [0, -1, float('nan'), float('inf')]:
      rows = self.normalized()
      rows[0]['run_us'] = value
      with self.assertRaises(ValueError):
        pair_run(rows, 4)

  def test_copied_collection_is_not_an_independent_repetition(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      for name in ['original', 'copy']:
        folder = root / name
        folder.mkdir()
        (folder / 'manifest.json').write_text(
            json.dumps(
                dict(collection_id='same-collection', complete=True, runs=[])
            )
        )
        (folder / 'symbols.txt').write_text('')
        (folder / 'elf.txt').write_text('')
      result = subprocess.run(
          [
              sys.executable,
              str(Path(__file__).with_name('summarize.py')),
              str(root / 'original'),
              str(root / 'copy'),
              '--out',
              str(root / 'summary'),
          ],
          capture_output=True,
          text=True,
      )
      self.assertEqual(result.returncode, 2)
      self.assertIn('duplicate collection', result.stderr)


if __name__ == '__main__':
  unittest.main()
