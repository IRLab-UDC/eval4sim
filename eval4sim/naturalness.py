import argparse
import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from eval4sim import corpus

MODEL = 'zayn1111/deberta-v3-dnli'
LABELS = ['entailment', 'neutral', 'contradiction']
COHERENCE = {'entailment': 1.0, 'neutral': 0.5, 'contradiction': 0.0}
WEIGHTS = {'cs': 0.565, 'pcr': 0.126, 'scr': 0.309}
CONFIDENCE = 0.7
WINDOW = 5


class DNLI:
    def __init__(self, batch_size, device=None):
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, pairs):
        results = []
        for i in range(0, len(pairs), self.batch_size):
            premises, hypotheses = zip(*pairs[i:i + self.batch_size])
            inputs = self.tokenizer(list(premises), list(hypotheses), return_tensors='pt',
                                    truncation=True, max_length=512, padding=True).to(self.device)
            confidence, label = torch.softmax(self.model(**inputs).logits, dim=-1).max(dim=-1)
            results.extend((LABELS[l], c) for l, c in zip(label.tolist(), confidence.tolist()))
        return results


def contradiction(result):
    label, confidence = result
    return label == 'contradiction' and confidence >= CONFIDENCE


def evaluate_conversation(nli, record):
    turns = [(turn['speaker'], turn['text']) for turn in record['turns']]
    personas = {1: record['persona1'], 2: record['persona2']}

    flow = [(prev, text) for (_, prev), (_, text) in zip(turns, turns[1:])]
    persona = [(trait, text) for speaker, text in turns for trait in personas[speaker]]
    history, windows, own = {}, [], []
    for speaker, text in turns:
        previous = history.setdefault(speaker, [])
        if previous:
            windows.append((len(own), len(own) + len(previous[-WINDOW:])))
            own.extend((p, text) for p in previous[-WINDOW:])
        previous.append(text)

    flow_results, persona_results, own_results = nli(flow), nli(persona), nli(own)
    labels = [label for label, _ in flow_results]
    rates = {label: labels.count(label) / len(labels) if labels else 0.0 for label in LABELS}

    cs = sum(COHERENCE[label] for label in labels) / len(labels) if labels else 1.0
    pcr = sum(map(contradiction, persona_results)) / len(persona) if persona else 0.0
    scr = sum(any(map(contradiction, own_results[a:b])) for a, b in windows) / len(turns) if turns else 0.0
    return {
        'id': record['id'],
        'naturalness': WEIGHTS['cs'] * cs + WEIGHTS['pcr'] * (1 - pcr) + WEIGHTS['scr'] * (1 - scr),
        'coherence_score': cs,
        'persona_contradiction_rate': pcr,
        'self_contradiction_rate': scr,
        'entailment_rate': rates['entailment'],
        'neutral_rate': rates['neutral'],
        'contradiction_rate': rates['contradiction'],
    }


def main():
    parser = argparse.ArgumentParser(description='Naturalness: dialogue NLI profile of simulated conversations.')
    parser.add_argument('corpora', nargs='+', help='Corpus files, e.g. data/corpora/*/*.jsonl')
    parser.add_argument('--output', default='results/naturalness')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    nli = DNLI(args.batch_size, args.device)
    for path in args.corpora:
        out_dir = corpus.output_dir(path, args.output)
        if (out_dir / 'summary.json').exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        scores = [evaluate_conversation(nli, record) for record in corpus.load(path)]
        corpus.save(scores, out_dir / 'conversations.jsonl')
        summary = {metric: sum(s[metric] for s in scores) / len(scores) for metric in scores[0] if metric != 'id'}
        with open(out_dir / 'summary.json', 'w') as f:
            json.dump({'conversations': len(scores), **summary}, f, indent=2)


if __name__ == '__main__':
    main()
