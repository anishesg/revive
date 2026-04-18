#!/usr/bin/env python3
"""
Generate synthetic training data for the REVIVE aggregator model.
Each example: N agent responses → synthesized best answer.
Uses Claude API. Costs ~$3-5 for 750 examples.

Usage:
  ANTHROPIC_API_KEY=sk-... python3 generate_dataset.py --n 750 --out data.jsonl
"""
import anthropic
import json
import random
import argparse
import time
from pathlib import Path

ROLES = ["Reasoner", "Writer", "Concise", "Critic", "Factchecker", "Drafter"]

# Representative queries covering all QueryType categories
QUERY_BANK = [
    # Complex reasoning
    "What are the ethical implications of autonomous weapons systems?",
    "Explain the trolley problem and its implications for self-driving car design.",
    "Should AI systems be granted legal personhood? Argue both sides.",
    "What would a post-scarcity economy actually look like?",
    "Is consciousness a product of computation, or something more?",
    "How does Keynesian economics differ from Modern Monetary Theory?",
    "What is the simulation hypothesis and what evidence exists for or against it?",
    "Can democracy survive social media? What structural changes would help?",
    # Creative
    "Write a short story about the last human librarian.",
    "Describe a city designed entirely around pedestrians, not cars.",
    "What would a world without intellectual property law look like?",
    "Write a pitch for a startup that makes death more humane.",
    # Code/technical
    "Explain the CAP theorem and when you'd sacrifice each guarantee.",
    "What's the difference between optimistic and pessimistic locking?",
    "How does a transformer attention mechanism work, intuitively?",
    "When should you use a B-tree vs a hash index?",
    # Math/science
    "Why does quantum entanglement not allow faster-than-light communication?",
    "Explain Gödel's incompleteness theorems to a non-mathematician.",
    "What is the Monty Hall problem and why is the answer counterintuitive?",
    "How does CRISPR-Cas9 actually edit DNA?",
    # Opinion/analysis
    "Is remote work better or worse for company culture long-term?",
    "Should universities still exist in their current form in 2040?",
    "Is nuclear power necessary to address climate change?",
    "What is the strongest argument against social media algorithms?",
    # Simple (use fewer agents)
    "What is the capital of Australia?",
    "Who invented the telephone?",
    "What does HTTP stand for?",
    "What is the boiling point of water at sea level?",
]

GENERATION_PROMPT = """You are generating training data for an AI aggregation model.

I will give you a user query and ask you to:
1. Write {n_agents} short responses, each from a different AI agent persona
2. Write the ideal synthesized answer that combines the best elements

Respond in this exact JSON format:
{{
  "agent_responses": [
    {{"role": "Reasoner", "response": "..."}},
    {{"role": "Writer", "response": "..."}},
    ... (one per agent)
  ],
  "synthesis": "..."
}}

Rules:
- Each agent response should be 2-6 sentences, reflecting their persona
- Reasoner: logical, step-by-step, explicit reasoning
- Writer: eloquent, well-structured prose
- Concise: as brief as possible while complete
- Critic: identifies flaws, counterarguments, edge cases
- Factchecker: focuses on verifiable facts, flags uncertainty
- Drafter: quick first-pass, covers key points fast
- Synthesis: 3-8 sentences, draws on the best from each, resolves contradictions

Query: {query}
Agents to use: {agents}"""


def generate_example(client: anthropic.Anthropic, query: str, n_agents: int = 4) -> dict | None:
    agents = random.sample(ROLES, min(n_agents, len(ROLES)))

    prompt = GENERATION_PROMPT.format(
        n_agents=n_agents,
        query=query,
        agents=", ".join(agents)
    )

    try:
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",  # cheapest, fast
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = message.content[0].text.strip()

        # Extract JSON from response
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        data = json.loads(raw)

        # Format as training example
        agent_section = "\n\n".join([
            f"[{r['role']}]: {r['response']}"
            for r in data["agent_responses"]
        ])

        return {
            "instruction": (
                "You are the Aggregator of a distributed AI swarm. "
                "Multiple specialized agents have independently answered a user's question. "
                "Synthesize the single best response by combining the strongest elements from each agent, "
                "resolving contradictions (prefer Factchecker and Critic for factual accuracy), "
                "using the Writer's clarity, and the Reasoner's logical structure."
            ),
            "input": f"Original question: {query}\n\nAgent responses:\n\n{agent_section}",
            "output": data["synthesis"]
        }
    except Exception as e:
        print(f"  Error on '{query[:50]}...': {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=750, help="Number of examples to generate")
    parser.add_argument("--out", default="data.jsonl", help="Output file")
    parser.add_argument("--api-key", default=None, help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    import os
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY environment variable")

    client = anthropic.Anthropic(api_key=api_key)
    out_path = Path(args.out)

    # Load existing to resume
    existing = set()
    examples = []
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                ex = json.loads(line)
                examples.append(ex)
                existing.add(ex["input"][:80])
        print(f"Resuming: {len(examples)} examples already written")

    with open(out_path, "a") as f:
        while len(examples) < args.n:
            query = random.choice(QUERY_BANK)
            n_agents = random.choice([3, 4, 4, 5, 5, 6])  # weighted toward 4-5 agents

            print(f"[{len(examples)+1}/{args.n}] {query[:60]}... ({n_agents} agents)")

            ex = generate_example(client, query, n_agents)
            if ex:
                key = ex["input"][:80]
                if key not in existing:
                    f.write(json.dumps(ex) + "\n")
                    f.flush()
                    examples.append(ex)
                    existing.add(key)

            # Rate limit: ~3 req/s on Haiku
            time.sleep(0.4)

    print(f"\nDone. {len(examples)} examples written to {out_path}")
    print(f"Estimated cost: ${len(examples) * 0.004:.2f} (Haiku ~$0.004/example)")


if __name__ == "__main__":
    main()
