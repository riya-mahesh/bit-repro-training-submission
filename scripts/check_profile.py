#!/usr/bin/env python3
"""Reject empty/partial captures; check six expected LeNet convolution launches."""
import re
import sys
from pathlib import Path


def validate(report, log, batch):
    if not re.search(r'^batch ' + str(batch) + r'\s', log, re.M):
        raise ValueError('Capture log does not confirm the requested batch size')
    if '==PROF== Profiling' not in log:
        raise ValueError('No kernel profiling recorded in capture log')
    # Headers contain launch dimensions, Context, Stream and Device, unlike prose rules.
    headers = [s for s in report.splitlines() if 'Context ' in s and 'Stream ' in s and 'Device ' in s]
    expected = [
        ('conv1 forward', r'forward_kernel<\s*(?:\(int\)\s*)?32\s*>'),
        ('conv2 forward', r'forward_kernel<\s*(?:\(int\)\s*)?256\s*>'),
        ('conv1 filter scales', r'filter_scales_kernel.*\(6, 1, 1\)x'),
        ('conv2 filter scales', r'filter_scales_kernel.*\(16, 1, 1\)x'),
        ('conv1 weight gradient', r'weight_grad_kernel.*\(25, 1, 1\)x'),
        ('conv2 weight gradient', r'weight_grad_kernel.*\(150, 1, 1\)x'),
    ]
    if len(headers) != 6:
        raise ValueError(f'Expected six captured launches, found {len(headers)}')
    for name, pattern in expected:
        if sum(bool(re.search(pattern, s)) for s in headers) != 1:
            raise ValueError('Missing or duplicated launch: ' + name)
    if len(re.findall(r'^\s+Duration\s+\S+\s+[\d.,]+', report, re.M)) != 6:
        raise ValueError('Missing duration metrics')


if __name__ == '__main__':
    try:
        validate(Path(sys.argv[1]).read_text(), Path(sys.argv[2]).read_text(), int(sys.argv[3]))
    except (ValueError, OSError) as exc:
        sys.exit('INCOMPLETE PROFILE: ' + str(exc))
    print('PASS: both layers have forward, filter-scale and weight-gradient metrics')
