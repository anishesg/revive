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
    "What happens to employment when AGI arrives?",
    "Is the multiverse hypothesis scientific or philosophical?",
    "How should we govern AI systems that are smarter than their creators?",
    "What are the second-order effects of universal basic income?",
    "Compare consequentialist and deontological arguments for privacy rights.",
    "What would a truly sustainable global economy look like?",
    "How does game theory explain the failure of climate negotiations?",
    "Should genetic editing of embryos be permitted? Analyze tradeoffs.",
    # Creative
    "Write a short story about the last human librarian.",
    "Describe a city designed entirely around pedestrians, not cars.",
    "What would a world without intellectual property law look like?",
    "Write a pitch for a startup that makes death more humane.",
    "Describe what the internet feels like to an AI.",
    "Write a conversation between two AIs debating consciousness.",
    "Imagine a school system designed from scratch in 2025.",
    "Describe the most beautiful algorithm you can imagine.",
    "Write a poem about distributed computing.",
    "Design a new holiday that celebrates human-AI collaboration.",
    # Code/technical
    "Explain the CAP theorem and when you'd sacrifice each guarantee.",
    "What's the difference between optimistic and pessimistic locking?",
    "How does a transformer attention mechanism work, intuitively?",
    "When should you use a B-tree vs a hash index?",
    "Explain the difference between threads, coroutines, and actors.",
    "How does garbage collection work in Go vs Rust's ownership model?",
    "What is the actor model and when should you use it?",
    "Explain consistent hashing and its use in distributed systems.",
    "How does TLS 1.3 differ from 1.2? What attacks does it prevent?",
    "What are the tradeoffs between REST, GraphQL, and gRPC?",
    "Explain how a bloom filter works and when to use one.",
    "How does WebAssembly achieve near-native performance?",
    "What is the Raft consensus algorithm and how does it handle leader election?",
    "Explain vector databases and why they matter for AI applications.",
    # Math/science
    "Why does quantum entanglement not allow faster-than-light communication?",
    "Explain Gödel's incompleteness theorems to a non-mathematician.",
    "What is the Monty Hall problem and why is the answer counterintuitive?",
    "How does CRISPR-Cas9 actually edit DNA?",
    "Explain the difference between P and NP problems with examples.",
    "What is Bayesian inference and how does it differ from frequentist statistics?",
    "How does the double-slit experiment demonstrate wave-particle duality?",
    "Explain the halting problem and its implications for computing.",
    "What is the mathematical basis of public-key cryptography?",
    "How does gradient descent find minima in high-dimensional spaces?",
    "Explain the birthday paradox and its implications for hash collisions.",
    "What is entropy in information theory vs thermodynamics?",
    # Opinion/analysis
    "Is remote work better or worse for company culture long-term?",
    "Should universities still exist in their current form in 2040?",
    "Is nuclear power necessary to address climate change?",
    "What is the strongest argument against social media algorithms?",
    "Should children learn to code, or will AI make that irrelevant?",
    "Is open-source AI development safer or more dangerous than closed?",
    "Will smartphones exist in 10 years, or will something replace them?",
    "Is meritocracy a myth? What does the evidence say?",
    "Should there be a right to disconnect from work communications?",
    "Is space colonization a realistic goal or a distraction from Earth's problems?",
    # Simple facts (use fewer agents)
    "What is the capital of Australia?",
    "Who invented the telephone?",
    "What does HTTP stand for?",
    "What is the boiling point of water at sea level?",
    "What year did the Berlin Wall fall?",
    "What is the speed of light in meters per second?",
    "Who wrote 1984?",
    "What is the largest planet in our solar system?",
    "What programming language was created by Guido van Rossum?",
    "How many bits are in a byte?",
    # Adversarial: agent disagreement
    "Is Pluto a planet?",
    "Was the moon landing faked?",
    "Is a hot dog a sandwich?",
    "What is the best programming language?",
    "Should you use spaces or tabs for indentation?",
    # Adversarial: one agent gives wrong answer (test factchecker priority)
    "What is the population of Tokyo?",
    "When was the first iPhone released?",
    "What is the distance from the Earth to the Sun?",
    "Who was the first person to walk on the Moon?",
    "How many chromosomes do humans have?",
    # Multi-step reasoning
    "If I have 3 boxes, one with apples, one with oranges, and one with both, and all labels are wrong, and I can pick one fruit from one box, how do I label them correctly?",
    "A bat and a ball cost $1.10 in total. The bat costs $1 more than the ball. What does the ball cost?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
]

GENERATION_PROMPT = """You are generating training data for the REVIVE aggregator — an AI model that runs on a distributed phone swarm. Multiple specialized agents on different devices independently answer a user's question, and the aggregator synthesizes the best answer.

I will give you a user query and ask you to:
1. Write {n_agents} short responses, each from a different AI agent persona
2. Write the ideal synthesized answer that combines the best elements

Each agent runs on a different device with varying capability. The aggregator should weight responses by quality, not just combine them.

Respond in this exact JSON format:
{{
  "agent_responses": [
    {{"role": "Reasoner", "device": "iPhone 15 Pro", "tok_s": 30, "response": "..."}},
    {{"role": "Writer", "device": "Pixel 7", "tok_s": 12, "response": "..."}},
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
- Factchecker: focuses on verifiable facts, flags uncertainty with [uncertain]
- Drafter: quick first-pass, covers key points fast (may be less accurate)
- Devices with higher tok/s tend to produce more thorough responses
- In ~10% of examples, have one agent give a subtly wrong answer — the synthesis should catch and correct it
- Synthesis: 3-8 sentences, draws on the best from each, resolves contradictions
- Prefer Factchecker and Critic for accuracy, Writer for clarity, Reasoner for logic
- If agents disagree, favor the more careful/well-reasoned response over the faster one

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

        agent_section = "\n\n".join([
            f"[{r['role']} — {r.get('device', 'unknown')} @ {r.get('tok_s', 0)} tok/s]: {r['response']}"
            for r in data["agent_responses"]
        ])

        return {
            "instruction": (
                "You are the Aggregator of a distributed AI swarm running across consumer phones, "
                "tablets, and edge devices. Multiple specialized agents have independently answered "
                "a user's question from different devices with varying compute capability. "
                "Synthesize the single best response by: combining the strongest elements from each agent, "
                "resolving contradictions (prefer Factchecker and Critic for factual accuracy), "
                "using the Writer's clarity and the Reasoner's logical structure. "
                "Faster devices (higher tok/s) may produce more thorough answers, but quality matters more than speed. "
                "If any agent's response seems factually wrong, flag or correct it."
            ),
            "input": f"Original question: {query}\n\nAgent responses:\n\n{agent_section}",
            "output": data["synthesis"]
        }
    except Exception as e:
        print(f"  Error on '{query[:50]}...': {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000, help="Number of examples to generate")
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
