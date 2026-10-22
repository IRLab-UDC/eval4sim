import json
import re
from pathlib import Path

SPEAKER = re.compile(r'User ([12]):')


def parse_turns(text):
    parts = SPEAKER.split(text)
    turns = []
    for speaker, utterance in zip(parts[1::2], parts[2::2]):
        utterance = re.sub(r'\s+\d+\.$', '', ' '.join(utterance.split())).strip(' "')
        if utterance:
            turns.append({'speaker': int(speaker), 'text': utterance})
    return turns


def load(path):
    with open(path, encoding='utf-8') as f:
        return [record for record in map(json.loads, f) if 'error' not in record]


def save(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def utterances(record, speaker):
    return [turn['text'] for turn in record['turns'] if turn['speaker'] == speaker]


def output_dir(path, root):
    path = Path(path)
    return Path(root) / path.parent.name / path.stem
