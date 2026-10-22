import argparse
import json
import random

import numpy as np

from eval4sim import corpus, pan


def halves(text):
    sentences = text.split('. ')
    if len(sentences) < 2:
        return None
    mid = len(sentences) // 2
    first, second = '. '.join(sentences[:mid]).strip(), '. '.join(sentences[mid:]).strip()
    if not first or not second:
        return None
    return [s if s.endswith('.') else s + '.' for s in (first, second)]


def build_pairs(texts, rng):
    same = [pair for pair in map(halves, texts) if pair]
    order = list(range(len(texts)))
    rng.shuffle(order)
    different = [[texts[i], texts[j]] for i, j in zip(order[::2], order[1::2])]

    n = min(len(same), len(different))
    pairs = [(pair, 1) for pair in same[:n]] + [(pair, 0) for pair in different[:n]]
    rng.shuffle(pairs)
    return [pair for pair, _ in pairs], np.array([label for _, label in pairs], dtype=np.float64)


def evaluate(records, seed, train_split):
    rng = random.Random(seed)
    texts = []
    for record in records:
        speakers = [' '.join(corpus.utterances(record, s)).strip() for s in (1, 2)]
        if all(speakers):
            texts.extend(speakers)
    rng.shuffle(texts)
    split = int(len(texts) * train_split)

    model = pan.fit(*build_pairs(texts[:split], rng))
    test_pairs, test_labels = build_pairs(texts[split:], rng)
    return pan.evaluate_all(test_labels, pan.predict(model, test_pairs))


def main():
    parser = argparse.ArgumentParser(description='Consistency: authorship verification of simulated speakers.')
    parser.add_argument('corpora', nargs='+', help='Corpus files, e.g. data/corpora/*/*.jsonl')
    parser.add_argument('--output', default='results/consistency')
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    for path in args.corpora:
        out_dir = corpus.output_dir(path, args.output)
        if (out_dir / 'metrics.json').exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics = evaluate(corpus.load(path), args.seed, args.train_split)
        with open(out_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
