import unittest
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from dejacode import load, clean, preprocess, Searcher

class TestBaselinePipeline(unittest.TestCase):
    def test_clean_text(self):
        sample = "<p>All submissions for this problem are available.</p> Hello $X_i$ world! Input: 1 2"
        cleaned = clean(sample)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("All submissions", cleaned)
        self.assertIn("Hello", cleaned)

    def test_preprocess(self):
        raw = pd.DataFrame({
            "QCode": ["TEST1"],
            "Title": ["Test Problem"],
            "statement": ["This is a problem statement long enough to exceed thirty characters."],
            "Tags": ["['dp', 'cook64', 'easy']"],
            "level": ["easy"],
            "link": ["https://codechef.com/problems/TEST1"]
        })
        processed = preprocess(raw)
        self.assertEqual(len(processed), 1)
        self.assertIn("dp", processed.iloc[0]["tags"])
        self.assertNotIn("cook64", processed.iloc[0]["tags"])
        self.assertNotIn("easy", processed.iloc[0]["tags"])

if __name__ == "__main__":
    unittest.main()
