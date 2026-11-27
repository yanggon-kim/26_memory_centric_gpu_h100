#!/usr/bin/env python3
"""Strict CPU reader for legacy long and NCU2026 wide base-unit CSV exports."""
import csv
import io
import math
from pathlib import Path

METRICS = {'dram__bytes_read.sum': {'byte', 'bytes'},
           'dram__bytes_write.sum': {'byte', 'bytes'},
           'gpu__time_duration.sum': {'ns', 'nsecond'}}


def parse_csv(text,include_l2=False):
    metrics=dict(METRICS)
    if include_l2:metrics['lts__t_bytes.sum']={'byte','bytes'}
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.lstrip('\ufeff').startswith('"ID"')), None)
    if start is None:
        raise ValueError('NCU CSV header missing')
    reader = csv.DictReader(io.StringIO('\n'.join(lines[start:])))
    fields = set(reader.fieldnames or [])
    long = {'Metric Name', 'Metric Unit', 'Metric Value'} <= fields
    if not long and not set(metrics) <= fields:
        raise ValueError('Unsupported or incomplete NCU metric schema')
    values = {}
    units_seen = False

    def number(value):
        if not isinstance(value, str):
            raise ValueError('Counter value absent')
        value = float(value.replace(',', ''))
        if not math.isfinite(value) or value < 0:
            raise ValueError('Nonfinite or negative counter')
        return value

    for row in reader:
        rid = row.get('ID')
        if not long and rid == '':
            if units_seen or any(row.get(m) not in allowed for m, allowed in metrics.items()):
                raise ValueError('Missing, duplicate or incompatible wide CSV unit row')
            units_seen = True
            continue
        if not rid:
            continue
        if long:
            metric = row['Metric Name']
            if metric not in metrics:
                continue
            if row['Metric Unit'] not in metrics[metric]:
                raise ValueError('Counter unit incompatible with base units')
            target = values.setdefault(rid, {})
            if metric in target:
                raise ValueError('Duplicate metric observation for one range ID')
            target[metric] = number(row['Metric Value'])
        else:
            if not units_seen or rid in values:
                raise ValueError('Wide CSV unit row absent or duplicate range ID')
            values[rid] = {m: number(row[m]) for m in metrics}
    if not values or any(set(row) != set(metrics) for row in values.values()):
        raise ValueError('Empty or incomplete counter observations')
    return values


def resolve_report(path):
    """Retain the actual compressed/uncompressed artifact name, never rename it."""
    path = Path(path)
    candidates = [path, path.with_suffix('.ncu-repz'), Path(str(path)+'.ncu-repz')]
    present = list(dict.fromkeys(p for p in candidates if p.is_file()))
    if len(present) != 1:
        raise ValueError('Expected exactly one profiler artifact; found '+str(present))
    return present[0]
