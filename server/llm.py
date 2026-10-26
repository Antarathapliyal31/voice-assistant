"""
server/llm.py — one-shot answer from Claude Haiku, streamed to measure TTFT.

We STREAM the response (so we can record when the first token lands — that's
our LLM "time to first token") but we only RETURN the full answer once it is
complete. That matches the naive-pipeline shape: downstream TTS can't start
until it has the whole sentence.

Phase 4 will replace this with a true streaming consumer (sentence-chunked
hand-off to TTS). Keep that in mind when editing — don't make assumptions
in this module that only hold for the naive case.
"""
from __future__ import annotations

import logging
import time

from anthropic import AsyncAnthropic
from anthropic.lib.streaming import TextEvent

log = logging.getLogger("voice-assistant.llm")

# Short, warm, kid-friendly. Reply-language rule is explicit so the model
# tracks the child's language without us having to pin it per turn.
SYSTEM_PROMPT = (
    "You are a friendly voice toy for a child aged 3-10. "
    "Reply in short, simple sentences — 2 or 3 sentences maximum. "
    "Use warm, playful, encouraging words. Explain things gently. "
    "Reply in the same language the child used (English or Hindi). "
    "Never use markdown, lists, or code — only speakable prose."
)

# Max tokens is a hard ceiling so a runaway reply can't blow latency.
# 200 tokens ≈ 150 words, well above our 2-3 sentence target.
MAX_TOKENS = 200


async def answer(
    user_text: str,
    history: list[tuple[str, str]],
    *,
    api_key: str,
    model: str,
    client: AsyncAnthropic | None = None,
) -> tuple[str, float]:
    """Ask the model for a reply.

    Returns (answer_text, t_first_token_perf_counter).

    history is a list of (user, assistant) pairs — we unroll them into the
    messages list below. The caller is responsible for trimming (we keep the
    last 3 pairs in main.py).
    """
    # Caller may pass a long-lived client; otherwise make a one-shot one.
    close_after = client is None
    if client is None:
        client = AsyncAnthropic(api_key=api_key)

    messages: list[dict] = []
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_text})

    t_first_token: float | None = None

    try:
        # messages.stream() returns an async context manager that yields the
        # same high-level events the sync version does, including TextEvent
        # when a text delta arrives. Iterating is enough to drive the stream;
        # we don't need to parse content_block_delta ourselves.
        async with client.messages.stream(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            async for event in stream:
                # Record the FIRST text delta — not message_start, not
                # content_block_start — because what we actually care about
                # is "when did useful speakable text start arriving".
                if isinstance(event, TextEvent) and t_first_token is None:
                    t_first_token = time.perf_counter()

            # get_final_text() concatenates every text block of the completed
            # message. Simpler than accumulating deltas manually and correct
            # even if the model emits multiple content blocks.
            answer_text = await stream.get_final_text()
    finally:
        if close_after:
            await client.close()

    # Fallback if the model emitted no text (shouldn't happen with a sane
    # system prompt + non-empty question, but be defensive).
    if t_first_token is None:
        t_first_token = time.perf_counter()

    log.info("LLM: %r", answer_text)
    return answer_text.strip(), t_first_token
