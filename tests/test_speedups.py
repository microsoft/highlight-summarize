import os
import threading
import unittest
from itertools import combinations
from unittest.mock import patch


os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")


class TestJudgesSharedClient(unittest.TestCase):
    def test_patched_get_completion_uses_imported_client_and_preserves_kwargs(self):
        import highlight_summarize.judges as judges

        captured = {}
        sentinel = object()

        class FakeCompletions:
            def parse(self, **kwargs):
                captured.update(kwargs)
                return sentinel

        class FakeChat:
            completions = FakeCompletions()

        class FakeBeta:
            chat = FakeChat()

        class FakeClient:
            beta = FakeBeta()

        fake_client = FakeClient()

        with patch.object(judges, "client", fake_client):
            result = judges.patched_get_completion(
                messages=[{"role": "user", "content": "hello"}],
                model="gpt-4.1-mini",
                temperature=0.2,
                max_tokens=111,
                seed=7,
                response_format={"type": "json_object"},
            )

        self.assertIs(result, sentinel)
        self.assertEqual(
            captured,
            {
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.2,
                "max_tokens": 111,
                "seed": 7,
                "response_format": {"type": "json_object"},
            },
        )


class TestPairwiseConcurrency(unittest.TestCase):
    def test_pairwise_comparisons_runs_all_unordered_pairs_concurrently_once(self):
        import reproduce.compare as compare

        pipelines = ["p0", "p1", "p2", "p3"]
        expected_pairs = {tuple(sorted(pair)) for pair in combinations(pipelines, 2)}

        called_pairs = set()
        called_lock = threading.Lock()
        barrier = threading.Barrier(len(expected_pairs), timeout=5)

        def fake_pairwise_comparison(run_folder_1, run_folder_2, model_name="gpt-4.1-mini"):
            pair = tuple(sorted((os.path.basename(run_folder_1), os.path.basename(run_folder_2))))
            with called_lock:
                called_pairs.add(pair)
            barrier.wait()

        with patch.object(compare, "pairwise_comparison", side_effect=fake_pairwise_comparison):
            compare.pairwise_comparisons("/tmp/results", pipelines, "fake-model")

        self.assertEqual(called_pairs, expected_pairs)


if __name__ == "__main__":
    unittest.main()