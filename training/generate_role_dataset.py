#!/usr/bin/env python3
"""
Generate fine-tuning data for individual agent roles (not just aggregator).
Each role gets specialized training examples that reinforce its persona.

Usage:
  ANTHROPIC_API_KEY=sk-... python3 generate_role_dataset.py --role reasoner --n 500
  ANTHROPIC_API_KEY=sk-... python3 generate_role_dataset.py --role all --n 300
"""
import anthropic
import json
import random
import argparse
import time
from pathlib import Path

ROLE_TRAINING_PROMPTS = {
    "reasoner": """Generate a training example for a "Reasoner" AI agent. The Reasoner thinks step by step, shows explicit reasoning chains, and prioritizes logical correctness.

Given this query, write the ideal Reasoner response (2-6 sentences, with clear step-by-step logic):

Query: {query}

Respond in JSON: {{"query": "...", "response": "..."}}""",

    "writer": """Generate a training example for a "Writer" AI agent. The Writer produces clear, well-structured, engaging prose. Prioritizes readability and flow.

Given this query, write the ideal Writer response (2-6 sentences, eloquent and clear):

Query: {query}

Respond in JSON: {{"query": "...", "response": "..."}}""",

    "concise": """Generate a training example for a "Concise" AI agent. The Concise agent answers in as few words as possible while being complete and accurate. No fluff.

Given this query, write the ideal Concise response (1-3 sentences max, extremely brief):

Query: {query}

Respond in JSON: {{"query": "...", "response": "..."}}""",

    "critic": """Generate a training example for a "Critic" AI agent. The Critic is a devil's advocate who identifies flaws, edge cases, counterarguments, and unstated assumptions.

Given this query, write the ideal Critic response (2-6 sentences, challenging and analytical):

Query: {query}

Respond in JSON: {{"query": "...", "response": "..."}}""",

    "factchecker": """Generate a training example for a "Factchecker" AI agent. The Factchecker focuses only on verifiable, accurate information. Explicitly flags uncertainty with [uncertain].

Given this query, write the ideal Factchecker response (2-6 sentences, precise and fact-focused):

Query: {query}

Respond in JSON: {{"query": "...", "response": "..."}}""",

    "spotter": """Generate a training example for a "Spotter" (query classifier) AI agent. The Spotter classifies queries into exactly one category.

Given this query, classify it as exactly one of: SIMPLE_FACT, COMPLEX_REASONING, CREATIVE, CODE, MATH, OPINION

Query: {query}

Respond in JSON: {{"query": "...", "classification": "CATEGORY_NAME"}}""",
}

DIVERSE_QUERIES = [
    "What causes the seasons on Earth?",
    "Should we colonize Mars? Why or why not?",
    "How does a neural network learn?",
    "Write a haiku about debugging code.",
    "What is the time complexity of quicksort?",
    "Is free will compatible with determinism?",
    "Explain quantum computing to a five-year-old.",
    "What are the pros and cons of microservices?",
    "How does inflation affect the average person?",
    "What is the most important invention of the 21st century?",
    "Explain the difference between TCP and UDP.",
    "Should social media have age restrictions?",
    "How do vaccines work at the molecular level?",
    "What makes a great leader?",
    "Explain the Fermi paradox.",
    "What is the future of renewable energy?",
    "How does a compiler optimize code?",
    "Is math discovered or invented?",
    "What are the risks of artificial general intelligence?",
    "How does the immune system fight cancer?",
    "What is the P vs NP problem?",
    "Should AI-generated art be copyrightable?",
    "How does HTTPS protect web traffic?",
    "What causes economic recessions?",
    "Is the universe infinite?",
    "How do you design a system for 1 billion users?",
    "What is the Ship of Theseus paradox?",
    "Explain MapReduce in simple terms.",
    "What would happen if the Earth stopped rotating?",
    "How does a hash table handle collisions?",
    "Is meritocracy achievable?",
    "What is the most efficient sorting algorithm?",
    "How do black holes form?",
    "Should programming be taught in elementary school?",
    "What is emergent behavior in complex systems?",
    "How does mRNA differ from DNA?",
    "What are the ethical limits of genetic engineering?",
    "Explain the CAP theorem to a non-engineer.",
    "What would a four-day work week change?",
    "How does blockchain consensus work?",
]

SYSTEM_PROMPTS = {
    "reasoner": "You are a rigorous analytical thinker. Think step by step.",
    "writer": "You are an eloquent communicator. Write clear, well-structured responses.",
    "concise": "You are a master of brevity. Answer in as few words as possible.",
    "critic": "You are a devil's advocate. Identify flaws and counterarguments.",
    "factchecker": "You are a fact-checker. Focus on verifiable, accurate information.",
    "spotter": "Classify the query into EXACTLY one category: SIMPLE_FACT, COMPLEX_REASONING, CREATIVE, CODE, MATH, OPINION",
}


def generate_example(client, role, query):
    prompt = ROLE_TRAINING_PROMPTS[role].format(query=query)
    try:
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        data = json.loads(raw)

        if role == "spotter":
            return {
                "instruction": SYSTEM_PROMPTS[role],
                "input": data.get("query", query),
                "output": data.get("classification", "COMPLEX_REASONING"),
            }
        else:
            return {
                "instruction": SYSTEM_PROMPTS[role],
                "input": data.get("query", query),
                "output": data.get("response", ""),
            }
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", default="all", help="Role to generate for, or 'all'")
    parser.add_argument("--n", type=int, default=300, help="Examples per role")
    parser.add_argument("--out-dir", default=".", help="Output directory")
    args = parser.parse_args()

    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY")

    client = anthropic.Anthropic(api_key=api_key)
    roles = list(ROLE_TRAINING_PROMPTS.keys()) if args.role == "all" else [args.role]

    for role in roles:
        out_path = Path(args.out_dir) / f"data-{role}.jsonl"
        existing = []
        if out_path.exists():
            with open(out_path) as f:
                existing = [json.loads(line) for line in f]
            print(f"Resuming {role}: {len(existing)} existing")

        with open(out_path, "a") as f:
            while len(existing) < args.n:
                query = random.choice(DIVERSE_QUERIES)
                print(f"[{role}] [{len(existing)+1}/{args.n}] {query[:50]}...")

                ex = generate_example(client, role, query)
                if ex and ex.get("output"):
                    f.write(json.dumps(ex) + "\n")
                    f.flush()
                    existing.append(ex)

                time.sleep(0.4)

        print(f"✓ {role}: {len(existing)} examples → {out_path}")

    print(f"\nEstimated cost: ${len(roles) * args.n * 0.003:.2f}")


if __name__ == "__main__":
    main()
