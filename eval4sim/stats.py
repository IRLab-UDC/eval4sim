import argparse
from pathlib import Path

import numpy as np

from eval4sim import corpus


def describe(records):
    personas = [p for r in records for p in (r['persona1'], r['persona2']) if p]
    messages = [t['text'] for r in records for t in r['turns']]
    return [
        len(records),
        np.mean([len(p) for p in personas]),
        np.mean([len(' '.join(p).split()) for p in personas]),
        np.mean([len(r['turns']) for r in records]),
        np.mean([len(m.split()) for m in messages]),
    ]


def main():
    parser = argparse.ArgumentParser(description='Persona and conversation statistics, averaged over the runs of each corpus.')
    parser.add_argument('--corpora', default='data/corpora')
    args = parser.parse_args()

    print('| Corpus | Samples | Persona items | Persona words | Messages | Message words |')
    print('|---|---:|---:|---:|---:|---:|')
    for corpus_dir in sorted(p for p in Path(args.corpora).iterdir() if p.is_dir()):
        runs = [describe(corpus.load(f)) for f in sorted(corpus_dir.glob('*.jsonl'))]
        if runs:
            samples, items, words, turns, length = np.mean(runs, axis=0)
            print(f'| {corpus_dir.name} | {samples:g} | {items:.2f} | {words:.2f} | {turns:.2f} | {length:.2f} |')


if __name__ == '__main__':
    main()
