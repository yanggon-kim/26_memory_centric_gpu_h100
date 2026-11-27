#!/usr/bin/env python3
import tempfile
from pathlib import Path
import unittest
from h200_ncu_read import parse_csv, resolve_report

LONG = '''"ID","Metric Name","Metric Unit","Metric Value"
"0","dram__bytes_read.sum","byte","1,024"
"0","dram__bytes_write.sum","byte","512"
"0","gpu__time_duration.sum","nsecond","100"
'''
WIDE = '''Slurm preamble
"ID","dram__bytes_read.sum","dram__bytes_write.sum","gpu__time_duration.sum"
"","byte","byte","ns"
"0","1,024","512","100"
'''


class Contract(unittest.TestCase):
    def test_optional_l2_requires_positive_base_unit_observation(self):
        long=LONG+'"0","lts__t_bytes.sum","byte","4096"\n'
        wide=WIDE.replace('"gpu__time_duration.sum"','"gpu__time_duration.sum","lts__t_bytes.sum"').replace('"","byte","byte","ns"','"","byte","byte","ns","byte"').replace('"0","1,024","512","100"','"0","1,024","512","100","4096"')
        self.assertEqual(parse_csv(long,include_l2=True),parse_csv(wide,include_l2=True))
        self.assertEqual(parse_csv(wide,include_l2=True)['0']['lts__t_bytes.sum'],4096)
        for bad in (LONG,WIDE,long.replace('"byte","4096"','"sector","4096"'),wide.replace('"4096"','"nan"')):
            with self.assertRaises(ValueError):parse_csv(bad,include_l2=True)

    def test_equivalent_schemas(self):
        self.assertEqual(parse_csv(LONG), parse_csv(WIDE))
        self.assertEqual(parse_csv(WIDE)['0']['dram__bytes_read.sum'], 1024)

    def test_reject_incomplete_nonfinite_units_duplicates(self):
        bad = [LONG.replace('"512"', '"nan"'), WIDE.replace('"ns"', '"us"'),
               LONG.splitlines()[0]+'\n'+LONG.splitlines()[1]+'\n',
               WIDE+WIDE.splitlines()[-1]+'\n', LONG+LONG.splitlines()[-1]+'\n',
               WIDE.replace('"","byte","byte","ns"\n', '')]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_csv(text)

    def test_report_resolution_retains_compressed_and_rejects_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            plain = Path(directory)/'capture.ncu-rep'
            compressed = plain.with_suffix('.ncu-repz')
            compressed.write_bytes(b'raw fixture')
            self.assertEqual(resolve_report(plain), compressed)
            plain.write_bytes(b'another fixture')
            with self.assertRaises(ValueError):
                resolve_report(plain)


if __name__ == '__main__':
    unittest.main()
