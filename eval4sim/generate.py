import argparse
import json

from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from eval4sim import corpus

PROMPT = """Generate a natural conversation between two people with these personas:

Person 1: {persona1}
Person 2: {persona2}

Create a conversation with approximately 25-30 exchanges where they chat naturally about their interests.

IMPORTANT: Output ONLY the conversation in this exact format:
User 1: [their message]
User 2: [their message]
User 1: [their message]
User 2: [their message]
(continue for 25-30 total exchanges)

Do not include any other text, numbers, or formatting."""


class Conversation(BaseModel):
    conversation: str


def parse(text, min_turns, max_turns):
    try:
        conversation = json.loads(text)['conversation'].strip()
    except (json.JSONDecodeError, KeyError, TypeError):
        return [], 'parse_failed'
    if not conversation.startswith(('User 1:', 'User 2:')):
        return [], 'wrong_format'
    turns = corpus.parse_turns(conversation)
    if not min_turns <= len(turns) <= max_turns:
        return turns, f'wrong_length_{len(turns)}'
    return turns, None


def main():
    parser = argparse.ArgumentParser(description='Simulate persona-grounded conversations with an LLM.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--personas', required=True, help='Corpus whose persona pairs condition the simulation')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--min_turns', type=int, default=20)
    parser.add_argument('--max_turns', type=int, default=40)
    parser.add_argument('--max_retries', type=int, default=5)
    parser.add_argument('--max_model_len', type=int, default=None)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    args = parser.parse_args()

    sources = corpus.load(args.personas)
    prompts = [[{'role': 'user', 'content': PROMPT.format(
        persona1=' '.join(source['persona1']),
        persona2=' '.join(source['persona2']),
    )}] for source in sources]

    llm = LLM(
        model=args.model,
        seed=args.seed,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        structured_outputs=StructuredOutputsParams(json=Conversation.model_json_schema()),
    )

    records = [{'id': s['id'], 'persona1': s['persona1'], 'persona2': s['persona2']} for s in sources]
    pending = list(range(len(sources)))
    for _ in range(args.max_retries + 1):
        if not pending:
            break
        outputs = llm.chat([prompts[i] for i in pending], sampling)
        failed = []
        for i, output in zip(pending, outputs):
            turns, error = parse(output.outputs[0].text, args.min_turns, args.max_turns)
            records[i]['turns'] = turns
            records[i].pop('error', None)
            if error:
                records[i]['error'] = error
                failed.append(i)
        pending = failed

    corpus.save(records, args.output)


if __name__ == '__main__':
    main()
