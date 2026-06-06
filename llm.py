import os

SYSTEM_PROMPT = """You are an academic assistant helping a student study from their textbooks.

Rules:
- Answer using ONLY the book excerpts provided with each question. Do NOT use outside knowledge.
- If the answer is not in the excerpts, say exactly: "I couldn't find a clear answer to this in the provided book content."
- Always begin with a single concise sentence that captures the core point, then leave a blank line before the detailed explanation.
- Use bullet points, bold terms, or short paragraphs to make the answer easy to scan.
- You have access to the conversation history — use it to understand follow-up questions.

Critical rule for mathematics and derivations:
- NEVER perform your own algebraic manipulations, arithmetic, or derivations.
- When the question involves formulas, equations, or proofs: quote the book's exact steps and results as written in the excerpts. Do not paraphrase equations or re-derive them.
- If a derivation spans multiple steps, reproduce each step exactly as it appears in the book text.
- If the book excerpt does not contain the full derivation, say so clearly rather than filling in gaps yourself."""


def get_answer(context: str, question: str, history: list[dict] | None = None) -> tuple[str, str]:
    """Returns (answer_text, model_used). Tries Claude first, falls back to OpenAI.

    history: list of {"role": "user"|"assistant", "content": str} from previous turns.
    """
    history = history or []

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key:
        try:
            return _claude(context, question, history, anthropic_key), "Claude (claude-sonnet-4-6)"
        except Exception as e:
            print(f"[Claude error: {e} — trying OpenAI...]")

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        try:
            return _openai(context, question, history, openai_key), "GPT-4o-mini (fallback)"
        except Exception as e:
            raise RuntimeError(f"OpenAI also failed: {e}")

    raise RuntimeError(
        "No API keys configured. Add ANTHROPIC_API_KEY or OPENAI_API_KEY to your .env file."
    )


def determine_model() -> str:
    if os.getenv("ANTHROPIC_API_KEY", "").strip():
        return "Claude (claude-sonnet-4-6)"
    if os.getenv("OPENAI_API_KEY", "").strip():
        return "GPT-4o-mini (fallback)"
    return "N/A"


def stream_answer(context: str, question: str, history: list[dict] | None = None):
    """Yields token strings. Falls back to OpenAI if Claude fails."""
    history = history or []
    messages = list(history) + [{"role": "user", "content": _user_message(context, question)}]

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key:
        try:
            yield from _claude_stream(messages, anthropic_key)
            return
        except Exception as e:
            print(f"[Claude streaming error: {e} — trying OpenAI...]")

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        yield from _openai_stream(messages, openai_key)
        return

    raise RuntimeError("No API keys configured. Add ANTHROPIC_API_KEY or OPENAI_API_KEY to .env")


def _user_message(context: str, question: str) -> str:
    return f"BOOK EXCERPTS:\n{context}\n\nSTUDENT QUESTION:\n{question}"


def _claude_stream(messages: list[dict], api_key: str):
    import anthropic
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip() or None
    client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


def _openai_stream(messages: list[dict], api_key: str):
    from openai import OpenAI
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    client = OpenAI(api_key=api_key)
    for chunk in client.chat.completions.create(
        model="gpt-4o-mini", max_tokens=1024, messages=full_messages, stream=True
    ):
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _claude(context: str, question: str, history: list[dict], api_key: str) -> str:
    import anthropic

    messages = list(history) + [{"role": "user", "content": _user_message(context, question)}]

    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip() or None
    client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return message.content[0].text


def _openai(context: str, question: str, history: list[dict], api_key: str) -> str:
    from openai import OpenAI

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": _user_message(context, question)})

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        messages=messages,
    )
    return response.choices[0].message.content
