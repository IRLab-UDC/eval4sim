import argparse
import csv
import math
import random
from collections import defaultdict

from colbert import Indexer, Searcher
from colbert.infra import ColBERTConfig, Run, RunConfig
from transformers import AutoTokenizer

from eval4sim import corpus

POOL_SIZES = [1, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000]
MARKERS = {1: '[USER1]', 2: '[USER2]'}


def chunk(turns, tokenizer, max_tokens, target_only):
    chunks, current, size = [], [], 0
    for turn in turns:
        if target_only and turn['speaker'] != 2:
            continue
        message = f"{MARKERS[turn['speaker']]} {' '.join(turn['text'].split())}"
        n = len(tokenizer.encode(message, add_special_tokens=False))
        if current and size + n > max_tokens:
            chunks.append(' '.join(current))
            current, size = [], 0
        current.append(message)
        size += n
    if current:
        chunks.append(' '.join(current))
    return chunks


def build_collection(records, tokenizer, max_tokens, target_only, path):
    owners = []
    with open(path, 'w', encoding='utf-8') as f:
        for conv_id, record in enumerate(records):
            for text in chunk(record['turns'], tokenizer, max_tokens, target_only):
                f.write(f'{len(owners)}\t{text}\n')
                owners.append(conv_id)
    return owners


def search(searcher, owners, chunks, query, pool, k):
    pids = [pid for conv in pool for pid in chunks[conv]]
    best = {}
    if pids:
        for pid, _, score in zip(*searcher.search(query, k=k, pids=pids)):
            best[owners[pid]] = max(best.get(owners[pid], -math.inf), score)
    return best


def evaluate(records, out_dir, args):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    queries = [' '.join(record['persona2']) for record in records]
    weight = math.sin(args.alpha * math.pi / 2)
    views = {name: w for name, w in [('full', 1 - weight), ('target', weight)] if w > 0}

    indexes = {}
    with Run().context(RunConfig(nranks=1, root=str(out_dir), index_root=str(out_dir / 'indexes'))):
        for name, w in views.items():
            collection = out_dir / f'{name}.tsv'
            owners = build_collection(records, tokenizer, args.max_tokens, name == 'target', collection)
            config = ColBERTConfig(doc_maxlen=args.doc_maxlen, nbits=args.nbits, checkpoint=args.checkpoint)
            Indexer(checkpoint=args.checkpoint, config=config).index(name=name, collection=str(collection), overwrite=True)
            chunks = [[] for _ in records]
            for pid, owner in enumerate(owners):
                chunks[owner].append(pid)
            indexes[name] = (Searcher(index=name, verbose=0), owners, chunks, w)

    curve = []
    for pool_size in args.pool_sizes:
        rng = random.Random(args.seed)
        reciprocal_ranks = []
        for qid, query in enumerate(queries):
            others = [c for c in range(len(records)) if c != qid]
            pool = [qid] + rng.sample(others, min(pool_size - 1, len(others)))
            scores = defaultdict(float)
            for searcher, owners, chunks, w in indexes.values():
                for conv, score in search(searcher, owners, chunks, query, pool, args.k).items():
                    scores[conv] += w * score
            ranking = sorted(scores, key=scores.get, reverse=True)
            reciprocal_ranks.append(1 / (ranking.index(qid) + 1) if qid in ranking else 0.0)
        curve.append((pool_size, sum(reciprocal_ranks) / len(reciprocal_ranks)))

    with open(out_dir / 'mrr.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pool_size', 'mrr'])
        writer.writerows(curve)


def main():
    parser = argparse.ArgumentParser(description='Adherence: MRR degradation curve of persona-to-conversation retrieval.')
    parser.add_argument('corpora', nargs='+', help='Corpus files, e.g. data/corpora/*/*.jsonl')
    parser.add_argument('--output', default='results/adherence')
    parser.add_argument('--alpha', type=float, default=1.0, help='0: full index only, 1: target-speaker index only')
    parser.add_argument('--pool_sizes', type=int, nargs='+', default=POOL_SIZES)
    parser.add_argument('--max_conversations', type=int, default=1000)
    parser.add_argument('--max_tokens', type=int, default=480)
    parser.add_argument('--checkpoint', default='colbert-ir/colbertv2.0')
    parser.add_argument('--doc_maxlen', type=int, default=512)
    parser.add_argument('--nbits', type=int, default=8)
    parser.add_argument('--k', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    for path in args.corpora:
        out_dir = corpus.output_dir(path, args.output)
        if (out_dir / 'mrr.csv').exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        records = corpus.load(path)
        if len(records) > args.max_conversations:
            records = random.Random(args.seed).sample(records, args.max_conversations)
        evaluate(records, out_dir, args)


if __name__ == '__main__':
    main()
