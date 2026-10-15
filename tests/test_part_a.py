"""Lightweight standard-library checks for the Part A generated artifacts."""
import json
import sqlite3
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class PartAArtifactsTest(unittest.TestCase):
    def test_cleaned_checkpoints(self):
        summary = json.loads((ROOT / 'outputs' / 'part_a_summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['customers']['rows'], 1200)
        self.assertEqual(summary['orders']['rows'], 12309)
        self.assertEqual(summary['orders']['statuses']['completed'], 11110)
        self.assertEqual(summary['orders']['statuses']['cancelled'], 1199)
        self.assertEqual(summary['orders']['completed_revenue_inr'], 2998668.8)

    def test_sqlite_tables_exist(self):
        with sqlite3.connect(ROOT / 'data' / 'instamart.db') as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue({'customers', 'products', 'orders'} <= tables)

if __name__ == '__main__':
    unittest.main()
