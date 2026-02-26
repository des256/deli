For complex prompts with persona instructions, chat history, memory injection, and tool-trigger parsing at ≤3B parameters, the key  
 constraint is instruction-following fidelity — the model needs to reliably stick to formatting rules and not lose the system prompt as the
context fills with chat history.

Your options ranked

Llama 3.2 3B — Strong instruction following at exactly 3B. Meta explicitly trained it for tool use, so it can produce structured tool-call
outputs with relatively little prompt engineering. Best general-purpose option in your list.

Phi-3-mini — Note this is 3.8B, so it exceeds your 3B limit. If you can stretch to ~4B it's competitive, but if 3B is a hard ceiling it's
out.

Gemma 3 1B — Surprisingly capable for 1B but the gap between 1B and 3B is significant for complex multi-part prompts. You'll see more
prompt drift (forgetting persona rules mid-conversation) and less reliable structured output for tool triggers.

SmolLM 2 1.7B — Designed for efficiency, but instruction following on complex prompts is noticeably weaker than Llama 3.2 3B. Better suited
for simpler single-turn tasks.

Strong alternative to consider

Qwen 2.5 3B — This consistently benchmarks at or above Llama 3.2 3B on instruction following and structured output tasks. It handles
complex system prompts well and has good multilingual support if that matters. At 3B it's probably the strongest option for your use case.

Practical recommendations

1. If 3B fits your hardware: Qwen 2.5 3B or Llama 3.2 3B. Try both — Qwen tends to be better at following formatting/structure rules, Llama
   tends to be more natural in conversation.
2. For tool triggering specifically: Define a simple output format (e.g., [TOOL: name | arg1 | arg2]) rather than full JSON. Small models
   handle single-line markers far more reliably than nested JSON.
3. Chat history management: At 3B you'll want to keep the effective context short — summarize older turns rather than including full
   history. These models degrade noticeably past ~2K tokens of chat context even if the context window is technically larger.
4. Test with your actual prompt: The gap between models is highly prompt-dependent at this scale. A 30-minute eval with your real system
   prompt and a few conversation turns will tell you more than benchmarks.

What hardware are you running on? If it's the Jetson from your recent work, that might further constrain the quantization and model choice.

====

Yes. The jetson has 16gb, but it also needs to host the other models at the same time, so there is roughly space for a 3b model  
 (quantized, likely int8). It does seem that a 2K context is extremely small for this purpose, even with summarizing and swapping  
 chunks to memory.

● You're right, 2K is too tight once you factor in system prompt + memory facts + even a summarized history. Let me be more precise about  
 what to expect.

Realistic context budget at 3B

Most 3B models have a 4K–8K native context window (Qwen 2.5 3B supports 32K nominally, Llama 3.2 3B supports 128K nominally, but actual  
 quality degrades well before those limits on small models). The practical sweet spot is roughly:

┌────────────────────────────────────────┬───────────────────┐
│ Component │ Tokens (estimate) │
├────────────────────────────────────────┼───────────────────┤
│ System prompt (persona, rules, format) │ 300–500 │
├────────────────────────────────────────┼───────────────────┤
│ Memory facts │ 200–400 │
├────────────────────────────────────────┼───────────────────┤
│ Tool definitions/examples │ 100–200 │
├────────────────────────────────────────┼───────────────────┤
│ Chat history │ remaining budget │
├────────────────────────────────────────┼───────────────────┤
│ Generation headroom │ ~200–300 │
└────────────────────────────────────────┴───────────────────┘

With a 4K effective context (where quality stays solid), that leaves ~2.5–3K tokens for chat history — roughly 30–40 conversational turns
if summarized aggressively, or 10–15 turns of raw history. That's workable for a conversational assistant.

The issue isn't the context window size, it's attention quality degradation. At 3B, information in the middle of a long context gets
ignored more than at larger scales (the "lost in the middle" problem). So the architecture of your prompt matters more than the raw token
count:

- System prompt at the start, memory/tool defs near the end (right before recent turns)
- Keep the most recent 3–5 turns verbatim, summarize everything older
- Put memory facts close to the user's latest message, not buried after the system prompt

On the Jetson 16GB constraint

INT8 at 3B is ~3GB VRAM. If your other models (Parakeet ASR, etc.) are already resident, you're right that 3B is roughly the ceiling. One
thing to consider: Qwen 2.5 3B at INT4 (GPTQ/AWQ) drops to ~1.5–2GB with only moderate quality loss for conversational tasks. That could
free headroom if you're tight, or let you fit a slightly larger model.

Recommendation

Go with Qwen 2.5 3B INT8 as your first candidate. It has the best instruction adherence at this size, and its 32K native window means the
architecture was at least trained to handle longer contexts even if you won't push it that far. Llama 3.2 3B is a close second — worth A/B
testing with your actual prompt.

====

At INT4, a 7B model lands around 3.5–4GB — roughly the same VRAM as a 3B at INT8. So it could fit your budget. From the earlier search,  
 there are several options:

┌──────────────────────────────────────────────────────────────────┬──────────────────────────────┐  
 │ Repo │ Notes │  
 ├──────────────────────────────────────────────────────────────────┼──────────────────────────────┤  
 │ keisuke-miyako/Qwen2.5-7B-Instruct-1M-onnx-int4 │ Instruct, 1M context variant │  
 ├──────────────────────────────────────────────────────────────────┼──────────────────────────────┤  
 │ keisuke-miyako/Qwen2.5-7B-onnx-int4 │ Base (non-instruct) │
├──────────────────────────────────────────────────────────────────┼──────────────────────────────┤
│ amd/Qwen2.5-7B-Instruct-awq-g128-int4-asym-bf16-onnx-ryzen-strix │ AWQ, targeted at AMD Ryzen │
├──────────────────────────────────────────────────────────────────┼──────────────────────────────┤
│ GamePlusPlus/Qwen2.5-7B-Instruct-int4-onnx-directml │ DirectML targeted │
└──────────────────────────────────────────────────────────────────┴──────────────────────────────┘

The keisuke-miyako Instruct variant is probably your best bet — same uploader as the 3B, so the export format should be consistent.

The real question is whether 7B INT4 is worth it over 3B INT4 for your use case. At 7B you'd get noticeably better instruction following,
more reliable tool-trigger formatting, and less prompt drift over long conversations. The tradeoff is:

- VRAM: ~3.5–4GB vs ~1.5–2GB for the 3B — that's 2GB less headroom for your other models
- Latency: roughly 2x slower token generation, which matters for a conversational assistant
- KV cache: 7B needs more KV cache memory at runtime, which grows with context length and could push you over budget during longer
  conversations

If your other models (Parakeet, etc.) leave enough room, 7B INT4 would be a meaningful step up in quality for complex prompts. But you'd
want to measure actual peak VRAM usage during inference, not just the static weight size — the KV cache at 4K context on a 7B model adds a
non-trivial amount.
