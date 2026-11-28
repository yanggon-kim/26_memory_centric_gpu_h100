import tempfile
from pathlib import Path
import unittest
from shoreline_treatment_analysis import analyze


class Preservation(unittest.TestCase):
    def test_new_output_never_overwrites_historical_aggregate(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'historical';(root/'results').mkdir(parents=True)
            original=root/'results/shoreline_treatment_statistics.json';original.write_text('preserve old artifact')
            output=Path(folder)/'new/results/requalified.json'
            data=analyze(root,output=output)
            self.assertEqual(original.read_text(),'preserve old artifact')
            self.assertTrue(output.exists());self.assertEqual(len(data['comparisons']),22)
            self.assertFalse(any(r['qualified'] for r in data['comparisons']))


if __name__=='__main__':unittest.main()
