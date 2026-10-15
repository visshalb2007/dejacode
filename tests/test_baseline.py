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

    def test_eval_set_exists_and_valid(self):
        self.assertTrue(os.path.exists("data/eval_set.csv"))
        df = pd.read_csv("data/eval_set.csv")
        self.assertGreaterEqual(len(df), 30)
        self.assertIn("query_qcode", df.columns)
        self.assertIn("similar_qcodes", df.columns)

    def test_recommend_next_progression(self):
        from dejacode import recommend_next
        df = pd.DataFrame({
            "qcode": ["P1", "P2", "P3", "P4"],
            "title": ["Prob 1", "Prob 2", "Prob 3", "Prob 4"],
            "difficulty": ["beginner", "beginner", "easy", "medium"],
            "doc": ["find sum of numbers", "add two integers", "maximum pair sum", "shortest graph path"],
            "tags": [["math"], ["math"], ["math"], ["graph"]],
            "link": ["/p1", "/p2", "/p3", "/p4"]
        })
        emb = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.85, 0.15, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=float)
        # Solved beginner problems P1, P2 -> step +1 should target easy (P3)
        recs = recommend_next(["P1", "P2"], df, emb, k=2, step=1)
        self.assertFalse(recs.empty)
        self.assertNotIn("P1", recs["qcode"].tolist())
        self.assertNotIn("P2", recs["qcode"].tolist())
        self.assertEqual(recs.iloc[0]["qcode"], "P3")
        self.assertEqual(recs.iloc[0]["difficulty"], "easy")

    def test_searcher_modes(self):
        df = pd.DataFrame({
            "qcode": ["A", "B"],
            "title": ["Graph Shortest Path", "Binary Search Array"],
            "doc": ["find shortest path in directed graph", "binary search in sorted array"],
            "difficulty": ["medium", "easy"],
            "tags": [["graph"], ["binary-search"]],
            "link": ["/a", "/b"]
        })
        emb = np.array([[1.0, 0.0], [0.0, 1.0]])
        s = Searcher(df, emb)
        s._q_emb = lambda q: emb[0]
        for mode in ["semantic", "keyword", "hybrid"]:
            res = s.search("shortest path graph", k=2, mode=mode)
            self.assertEqual(len(res), 2)
            self.assertEqual(res.iloc[0]["qcode"], "A")

if __name__ == "__main__":
    unittest.main()
