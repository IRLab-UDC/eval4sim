import argparse
import csv
import json
from pathlib import Path

import numpy as np

DIMENSIONS = ['adherence', 'consistency', 'naturalness']


def mrr_curve(files):
    curves = [np.loadtxt(f, delimiter=',', skiprows=1, ndmin=2) for f in files]
    return curves[0][:, 0], np.mean([c[:, 1] for c in curves], axis=0)


def span_weights(pool_sizes):
    p = np.asarray(pool_sizes, dtype=float)
    inner = (p[2:] - p[:-2]) / 2
    return np.concatenate([[p[1] - p[0]], inner, [p[-1] - p[-2]]])


def curve_similarity(a, b, w):
    return np.sum(w * a * b) / max(np.sum(w * a * a), np.sum(w * b * b))


def relative_similarity(value, reference):
    return 1 - abs(value - reference) / reference


def mean_of(files, key):
    return np.mean([json.loads(f.read_text())[key] for f in files])


def collect(root, dimension, filename):
    corpora = {}
    for corpus_dir in sorted(p for p in (Path(root) / dimension).glob('*') if p.is_dir()):
        files = sorted(corpus_dir.glob(f'*/{filename}'))
        if files:
            corpora[corpus_dir.name] = files
    return corpora


def scores(root, reference):
    results = {}

    curves = {name: mrr_curve(files) for name, files in collect(root, 'adherence', 'mrr.csv').items()}
    if reference in curves:
        pool_sizes, ref = curves[reference]
        weights = span_weights(pool_sizes)
        for name, (_, mrr) in curves.items():
            results.setdefault(name, {})['adherence'] = curve_similarity(ref, mrr, weights)

    for dimension, filename, key in [('consistency', 'metrics.json', 'overall'), ('naturalness', 'summary.json', 'naturalness')]:
        values = {name: mean_of(files, key) for name, files in collect(root, dimension, filename).items()}
        if reference in values:
            for name, value in values.items():
                results.setdefault(name, {})[dimension] = relative_similarity(value, values[reference])

    for row in results.values():
        if all(d in row for d in DIMENSIONS):
            row['e4s'] = np.mean([row[d] for d in DIMENSIONS])
    return results


def main():
    parser = argparse.ArgumentParser(description='Human-similarity scores per dimension and the aggregate E4S score.')
    parser.add_argument('--results', default='results')
    parser.add_argument('--reference', default='personachat')
    parser.add_argument('--output', default='results/e4s.csv')
    args = parser.parse_args()

    results = scores(args.results, args.reference)
    columns = DIMENSIONS + ['e4s']
    rows = sorted(results.items(), key=lambda item: item[1].get('e4s', -1), reverse=True)

    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['corpus'] + columns)
        writer.writerows([name] + [f'{row[c]:.4f}' if c in row else '' for c in columns] for name, row in rows)

    print('| Corpus | ' + ' | '.join(c.capitalize() if c != 'e4s' else 'E4S' for c in columns) + ' |')
    print('|---|' + '---:|' * len(columns))
    for name, row in rows:
        print(f'| {name} | ' + ' | '.join(f'{row[c]:.3f}' if c in row else '–' for c in columns) + ' |')


if __name__ == '__main__':
    main()
