# REVIVE

In 2022, the world discarded 5.3 billion phones. Not destroyed, not recycled -- discarded. Sitting in drawers, piling up in e-waste bins, waiting to ship to a facility overseas where the valuable metals get extracted and the rest gets burned. The Global E-Waste Monitor puts total electronic waste at 62 million tonnes that same year, growing five times faster than formal recycling. The part that sticks: roughly 95% of a smartphone's carbon footprint comes from manufacturing, not from use. The transistors, the copper, the rare earth magnets in the vibration motor -- all of that cost was already paid. The device already exists. Its marginal cost of continued operation is just electricity.

Meanwhile, running a real language model locally still requires a GPU most people will never own.

Those two facts are not unrelated. Every discarded iPhone 12 has an A14 Bionic with a dedicated neural engine. Every forgotten Pixel 7 has an ARM Cortex-A78 that can run quantized inference at several tokens per second. A 2019 mid-range Android is not a toy. It has more raw floating-point throughput than a workstation from a decade ago. The cluster is already in people's drawers. Nobody asked it to do anything useful.

REVIVE is our attempt to change that.

---

## What it does

The system runs a single language model across a cluster of consumer devices on the same WiFi. A phone, a tablet, a Raspberry Pi, a laptop. The model gets split into consecutive layer slices, one per device. Each device loads only its chunk into memory. When the cluster generates a token, the computation flows through every device in order -- a lap around a ring -- and the last device in the chain produces the output. The result is numerically identical to what you would get running the full model on one machine.

This is different from the more common approach of giving each device its own complete model and averaging the outputs. Those systems are easier to build, but they cap out at the quality of whatever fits on one device, and averaging language model outputs is a lossy operation. REVIVE does something harder: it actually runs one model, cooperatively, across multiple machines. The devices are not collaborating on a vote. They are collaborating on a computation.

There are two prior projects worth understanding as context. Petals (2022) runs 100B+ models like BLOOM across volunteer GPU clusters over the internet, in a BitTorrent-style network where each participant hosts a subset of layers. It proved that pipeline parallelism at internet scale was feasible, but it requires persistent public network connectivity and is designed for large models on GPU hardware. Exo (2024) takes a peer-to-peer approach with no master node and runs models like Llama 3.1 70B across a heterogeneous mix of Macs, iPhones, and NVIDIA GPUs. It demonstrated that consumer devices could coordinate without a central server. REVIVE differs from both in a few ways that matter. We target Android smartphones as first-class nodes -- not an afterthought alongside Mac hardware -- and we built dedicated hardware-level cluster management using an Arduino Uno as a baseboard management controller, which neither project has. We also built and distilled the model rather than shipping off-the-shelf weights.

---

## The ring topology

Each device runs an HTTP server. On startup, it loads its assigned layers into memory, registers with the cluster controller, and reports its layer range and hardware capabilities. When a query arrives, the first device tokenizes it and runs a forward pass through its slice of the network. The output is a hidden state tensor: a matrix of fp16 floating-point values that encodes everything the model has computed so far. That tensor gets serialized -- raw bytes, with a 4-byte big-endian length prefix and a small JSON header -- and sent over the network to the next device. Which runs its layers on top of it, and forwards the result again. The final device in the ring runs its layers, samples a next token from the output distribution, sends that token ID back to the coordinator, and then the whole cycle repeats for the next token.

The math on network bandwidth is worth spelling out because it surprised us. For a model with hidden dimension H and sequence length S, each stage-boundary transfer is roughly S x H x 2 bytes (the 2 is for fp16). During the autoregressive decoding phase, S = 1 for each new token -- you are only computing one token at a time. For Qwen3-1.7B with H = 2048, that is 1 x 2048 x 2 = 4KB per hop. At 15 tokens per second across three devices, that is less than 1 MB/s of sustained network traffic. A typical home WiFi connection handles 60 MB/s without effort. The bottleneck is the phone doing matrix multiplication, not the wire between phones. This is why the "pipeline parallelism doesn't work on slow networks" folklore is wrong, at least for the inference decode phase on small models. That concern applies to tensor parallelism, which does an all-reduce after every single attention and FFN layer and genuinely needs NVLink or InfiniBand to stay fast. Pipeline parallelism communicates only at stage boundaries, once per forward pass per stage. Completely different communication pattern.

The KV cache lives locally on each device. Each stage manages its own attention keys and values for the sequence; position IDs (absolute, so RoPE works independently per stage) are passed explicitly in the frame header. On non-first stages, the llama.cpp context receives hidden states via the `batch.embd` buffer, not the standard token buffer -- this is the exact mechanism that lets the model run across machines rather than insisting on processing from the token embedding layer.

One subtle bug we caught: on the last stage, after sampling a token, you have to explicitly call `llama_memory_seq_rm` to clear the KV cache for that sequence, otherwise activation state accumulates across generation steps and the outputs drift. The degenerate loop problem -- "Paris. Paris. Paris." -- turned out to be a separate issue, fixed by applying a repetition penalty to the output logits before temperature scaling.

---

## The Arduino

A $10 Arduino Uno is plugged into the host laptop over USB. It runs the cluster. Not inference, control.

The Uno has an ATmega328P running at 16 MHz with 2KB of SRAM and 32KB of flash. The BMC firmware is about 5KB of C. It owns the partition table -- which device covers which layers, who is first in the ring, who is last. Every worker sends a heartbeat over the network every three seconds. The Arduino watches those heartbeats. Two consecutive missed heartbeats (a six-second window) and the worker gets declared dead. The Arduino recalculates the partition, skipping the dead node, and broadcasts the new assignment to surviving workers. Unplug the Arduino and the cluster halts. It has no authority without the BMC.

The protocol is line-oriented ASCII over serial, because that is what fits in 2KB. Commands from the host look like `HB 2 147 35` (heartbeat from worker 2, reporting 14.7 tokens per second and 35 degrees Celsius -- the rate and temperature are integer fixed-point to avoid floats on the MCU). The Arduino responds with events like `DEAD 2` when a worker times out or `PARTITION 0:0:12 1:12:24 3:24:32` when it recalculates the layer split after a failure. The host-to-BMC handshake starts with `HELLO`, the BMC responds with `READY <version>`, and from there it is just heartbeats and partition updates until something changes.

The 2KB SRAM constraint is real and it bit us repeatedly. We initially wanted dynamic routing tables, per-worker retry queues, and in-flight sequence tracking. None of that fits. What fits is a flat array of six worker structs -- id, capability score (uint16, scaled x100), last heartbeat timestamp (uint32 milliseconds), tokens per second (uint16, scaled x10), temperature (int8, Celsius) -- plus a state machine with three states (down, degraded, healthy) and the line-oriented ASCII protocol. The partitioning algorithm itself is a greedy proportional split: compute a capability-weighted fraction of the total layers for each worker, assign that many layers, adjust the last worker to cover whatever's left. Fits in about 60 lines of C.

Simplicity made it reliable. A microcontroller cannot crash the same way a software process can. It does not share a heap with the workers. If the coordinator dies, the Arduino still knows who is alive. There is something genuinely worth having about putting the cluster's control plane on dedicated hardware with a dedicated power rail.

We also built a full software stand-in for the Arduino that runs on the laptop and speaks the exact same wire protocol, exposed over a TCP socket instead of serial. That let us develop the full cluster, including failure injection and recovery testing, without touching physical hardware. Running both the firmware and the simulator against the same test suite was the discipline that kept them in sync.

---

## The fine-tuned model

The model is a distilled, pruned, and quantized version of Qwen3 from Alibaba. Qwen3 is worth talking about because it is genuinely impressive at its size. Qwen3-8B outperforms Qwen2.5-14B on most benchmarks despite being nearly half the parameter count. Qwen3-1.7B scores competitively with models three times its size on reasoning evals. The whole dense lineup -- 0.6B through 32B -- supports 119 languages and 128K context length natively, and the models have a hybrid thinking mode where you can toggle chain-of-thought reasoning on or off per request. The small end of the family was clearly designed to run on constrained hardware, which is exactly what we needed.

We did not just use off-the-shelf Qwen3 weights. We built a training pipeline that produces eight role-specialized models -- Spotter, Drafter, Concise, Reasoner, Writer, Critic, Factchecker, Aggregator -- each fine-tuned for a specific part of the inference pipeline. The training data comes from two sources: a set of examples generated by Claude Haiku (around 750 per role, totaling about $2 in API costs) and a distillation set generated by running Qwen3-4B locally on the same prompts (another 750 per role). We merged and deduplicated those into roughly 1,500 training examples per role.

Fine-tuning used QLoRA via Unsloth. LoRA rank 32, alpha 64, targeting all the attention and FFN projection matrices. Three epochs, learning rate 2e-4, cosine schedule with 5% warmup. The training ran on an AWS g5.xlarge spot instance (an A10G GPU) launched from a shell script, taking about 35 minutes per role at negligible cost. The Unsloth fork of QLoRA cuts GPU memory requirements significantly versus a naive implementation, which is what made this tractable on a single A10G.

After fine-tuning, we apply ShortGPT-style layer pruning (arXiv 2403.17887) to roles that need to run on constrained hardware. The pruning algorithm scores each transformer block by the cosine similarity between its input and output hidden states: if a block barely changes its input, it is not contributing much and can be dropped. For the Spotter role -- which just needs to classify queries into one of six categories -- we drop 40% of the layers. For Drafter and Concise, 25%. Reasoning-heavy roles get no pruning. The calibration set for scoring is role-specific, which matters because the "important" layers differ by task.

Quantization runs in four tiers, each targeting a different device class. Ewaste-tier devices (1-3GB RAM, old phones and Pi 3s) get Q2_K. Budget devices (3-4GB) get Q3_K_S. Standard devices (4-6GB, covering most phones from 2018 onward) get Q4_K_M. Modern devices (6GB+) get Q5_K_M. For the K-quant formats, we use llama.cpp's importance matrix (imatrix) to guide which weights to quantize aggressively -- the imatrix is computed by running the model on role-specific calibration prompts and measuring which weights have the most impact on output. This matters most for Q2 and Q3, where naive quantization loses quality that imatrix-guided quantization preserves.

The final GGUF files are named like `revive-reasoner-qwen3-1.7b-standard-Q4_K_M.gguf`. Each one is self-contained, loads directly into llama.cpp, and runs with no Python runtime on-device.

The refinement layer above the swarm goes one step further. After the on-device cluster finishes generating, the raw chain of thought passes to Claude Sonnet via Bedrock. Sonnet reads what the small models produced and writes a polished final answer grounded in that reasoning. The actual thinking happens on your devices. The polish happens in the cloud, with the small models acting as the reasoning engine and Sonnet acting as the editor.

---

## Four platforms, one protocol

REVIVE runs on iOS (Swift, Metal, llama.cpp compiled as an xcframework), Android (Kotlin, JNI bridge to llama.cpp with ARM NEON vectorization), Raspberry Pi (Python, llama.cpp server subprocess), and macOS (Python, llama.cpp server subprocess with Metal offload on Apple Silicon). All four platforms speak the same OpenAI-compatible HTTP protocol at the worker layer: POST to `/v1/chat/completions`, get back a JSON response with the generated text plus telemetry (tokens per second, thermal state, battery level, memory used, time to first token). The coordinator discovers workers via mDNS (Bonjour on Apple platforms, Android NSD, Avahi on Linux), so devices find each other without any manual configuration.

The iOS binding is the most involved. libLlama.swift initializes the llama.cpp backend, loads the model file from the app's Documents folder, and manages the sampling chain: repetition penalty, then top-k, then top-p, then temperature, then multinomial sampling. Thread count is clamped to `max(1, min(8, processorCount - 2))` to leave room for the OS and the HTTP server. One non-obvious detail: tokenization requires `parse_special = true`, otherwise the ChatML control tokens (`<|im_start|>`, `<|im_end|>`) get tokenized as literal characters instead of the special token IDs the model was trained on. Without that flag, role boundaries disappear and the model never sees where the system prompt ends or where the user message starts.

Performance across devices, measured on Qwen3-1.7B Q4_K_M generating 150 tokens:

MacBook M2 Pro hits 60-80 tokens per second via Metal. iPhone 15 Pro does 25-35. iPhone 14 runs 15-20. A Pixel 8 on CPU-only NEON gets 8-15. Raspberry Pi 5 (the 8GB model) does 3-6. The swarm's aggregate throughput is roughly the sum of its parts -- four iPhones gives you around 100 tokens per second. Power draw is 2-3W per phone during inference, so a ten-phone cluster consumes 20-30W total. An A100 draws 300W just sitting there.

---

## What works right now

Three devices produce the exact same token sequence as the single-machine reference implementation. Getting that correctness check to pass was the first hard milestone and it took a debugging session to get there -- the per-stage KV cache management has edge cases that only show up when you're running real sequences, not toy examples.

The Arduino is real hardware. Not a simulator. The physical board plugged in over USB, running the C firmware. You can open a serial monitor and watch the heartbeat traffic live while the cluster is inferring.

The kill-and-heal demo works. Unplug a phone mid-generation. The Arduino misses two heartbeats from it (six seconds), marks it dead, recomputes the partition, broadcasts the new assignment. The cluster halts cleanly on the in-flight request rather than hanging or producing garbage output. Plug the phone back in. It re-registers. The Arduino broadcasts the restored partition. New queries flow.

There is also a small thing that turned out to matter more than expected: the fan on the Arduino's PWM output spins up when inference starts and spins back down when it finishes. It is cosmetically unnecessary and technically trivial. But a $4 piece of hardware visibly responding to language model inference load, just from a heartbeat telemetry field, is a satisfying thing to demo.

---

## What we learned

Home WiFi is not the constraint. It never was, for pipeline parallelism on small models. The constraint is compute on the phones. The network folklore about needing InfiniBand for distributed inference applies to tensor parallelism, which is a fundamentally different communication pattern. Pipeline parallelism transfers kilobytes per token at stage boundaries. WiFi handles that without trying.

A microcontroller is a good place for a cluster's control plane for the same reason hardware BMCs exist in data centers: it is failure-independent from the machines it manages. An IPMI controller on a server rack does not share memory with the workload. It has its own power rail. It can power-cycle a hung node. The Arduino Uno is a $10 off-the-shelf version of that same idea, and it works for exactly the same reasons.

Small models trained carefully against large models are genuinely useful at their size. DistilQwen2.5 (arXiv 2504.15027) showed this formally: a 7B student trained against a 72B teacher outperforms the non-distilled 7B baseline by measurable margins on instruction-following benchmarks, and the improvement is largest for the smallest student sizes. We did not need a 70B parameter model. We needed a 1.7B model that learned from a 4B.

And five billion discarded phones is not an abstraction. Manufacturing accounts for about 95% of a smartphone's lifetime carbon footprint. Extending a phone's useful life by two years improves its environmental balance by roughly 50%, according to a 2025 study in Communications Earth & Environment. A refurbished phone emits about 91% less CO2 than a new one. The compute these devices contain already cost something to produce. The only question is whether it ever gets used.
