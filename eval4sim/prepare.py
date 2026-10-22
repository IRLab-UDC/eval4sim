import argparse
import csv
import json

from eval4sim import corpus


def from_personachat(path, split):
    with open(path, encoding='utf-8') as f:
        dialogues = json.load(f)[split]

    records = []
    for i, dialogue in enumerate(dialogues):
        last = dialogue['utterances'][-1]
        lines = last['history'] + [last['candidates'][-1]]
        records.append({
            'id': str(i),
            'persona1': [],
            'persona2': [p.strip() for p in dialogue['personality']],
            'turns': [{'speaker': 1 + j % 2, 'text': text.strip()} for j, text in enumerate(lines)],
        })
    return records


def from_spc(path):
    with open(path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    return [{
        'id': str(i),
        'persona1': [p.strip() for p in row['user 1 personas'].split('\n') if p.strip()],
        'persona2': [p.strip() for p in row['user 2 personas'].split('\n') if p.strip()],
        'turns': corpus.parse_turns(row['Best Generated Conversation']),
    } for i, row in enumerate(rows)]


def main():
    parser = argparse.ArgumentParser(description='Convert PersonaChat (.json) or SPC (.csv) into the Eval4Sim corpus format.')
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--split', default='valid', help='PersonaChat split')
    args = parser.parse_args()

    records = from_personachat(args.input, args.split) if args.input.endswith('.json') else from_spc(args.input)
    corpus.save(records, args.output)


if __name__ == '__main__':
    main()
