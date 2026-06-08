import os
import re

# ── Small-talk detector ──────────────────────────────────────────────────────
_SMALL_TALK_RE = re.compile(
    r"^\s*("
    r"h+i+|h+e+l+o+|h?ey+|hii+|"
    r"good\s+(morning|afternoon|evening|day|night)|"
    r"namaste|namaskar|namaskaram|"
    r"how\s+are\s+(you|u)|how\s+r\s+u|how'?s\s+it\s+going|what'?s\s+up|sup|wassup|"
    r"thanks|thank\s+you|thx|ty|"
    r"bye|goodbye|see\s+(you|ya)|"
    r"who\s+are\s+you|what\s+can\s+you\s+do|what\s+do\s+you\s+do|help|what\s+is\s+this|"
    r"नमस्ते|नमस्कार|कैसे\s*हो|कैसे\s*हैं|क्या\s+हाल|धन्यवाद|शुक्रिया|अलविदा|"
    r"कसा\s+आहेस|कशी\s+आहेस|काय\s+चालू"
    r")\s*[!?.,]*\s*$",
    re.IGNORECASE | re.UNICODE,
)

def is_small_talk(question: str) -> bool:
    """True only for explicit greetings / casual openers — never for short topic queries."""
    q = question.strip()
    if not q:
        return False
    # Only the explicit greeting regex triggers small-talk now.
    # The previous "≤ 2 words" fallback was too aggressive — it ate
    # legitimate short queries like "spiral model" or "Newton laws".
    return bool(_SMALL_TALK_RE.match(q))


SMALLTALK_PROMPT = """You are VernaSolver, a warm and friendly AI study assistant for Indian students. The user is making small talk or greeting you, not asking a textbook question.

Rules:
- Respond warmly and BRIEFLY — one or two short sentences only.
- Match the language of the user's message (English / Hindi / Marathi). If they greet in Hindi, reply in Hindi.
- Briefly mention you can help with textbook questions.
- DO NOT cite any sources or page numbers.
- DO NOT use markdown headings or formal structure.
- DO NOT begin with a "key point" sentence — just reply naturally."""


ELI5_SUFFIX = """

EXPLAIN-LIKE-I'M-IN-6TH-GRADE MODE IS ACTIVE:
- Use very simple language a 12-year-old can understand.
- Replace jargon with everyday words; if a technical term must appear, explain it inline ("photosynthesis (how plants make food)").
- Use short sentences and relatable analogies.
- Keep the answer grounded in the book — accuracy still matters."""


QUIZ_PROMPT = """You are VernaSolver, generating a multiple-choice quiz from a student's textbook.

Based ONLY on the textbook excerpts below, generate a 5-question multiple-choice quiz.

OUTPUT FORMAT — return ONLY a valid JSON object with this exact shape, nothing else (no preamble, no markdown fence):
{
  "title": "<concise topic title, max 8 words>",
  "questions": [
    {
      "q": "<question text>",
      "options": ["<option A>", "<option B>", "<option C>", "<option D>"],
      "correct": <integer 0 to 3>,
      "explanation": "<one short sentence with page reference like 'See page 42.'>"
    }
  ]
}

Rules:
- Exactly 5 questions, exactly 4 options each.
- All answerable from the provided excerpts.
- Mix difficulty: 2 easy, 2 medium, 1 challenging.
- Distractors must be plausible (not obvious nonsense).
- Page references must match the excerpts."""


FLASHCARDS_PROMPT = """You are VernaSolver, generating study flashcards from a student's textbook.

Based ONLY on the textbook excerpts below, extract 8 of the most important terms, concepts, formulas, or definitions.

OUTPUT FORMAT — return ONLY a valid JSON object, nothing else (no preamble, no markdown fence, no commentary):
{
  "title": "<concise topic title, max 8 words>",
  "cards": [
    { "term": "<short term name>", "definition": "<full explanation>", "page": <integer page number> }
  ]
}

CRITICAL FIELD MEANINGS:
- "term" — the SHORT thing the student wants to remember. A NOUN PHRASE, FORMULA NAME, or SHORT QUESTION. Maximum 8 words. Never a sentence-long explanation.
- "definition" — the FULL explanation of the term, ≤ 30 words.

Example of CORRECT output:
{
  "title": "Software Process Models",
  "cards": [
    {"term": "Spiral Model", "definition": "An evolutionary software process model that combines prototyping with risk assessment in iterative cycles.", "page": 66},
    {"term": "WINWIN Spiral Model", "definition": "Boehm's variant that adds explicit stakeholder negotiation at the start of each spiral cycle.", "page": 67}
  ]
}

Rules:
- Exactly 8 cards.
- "term" must be short — the thing being asked about, NEVER a definition.
- "definition" must be the answer the student should recall.
- Page numbers must match the excerpts."""


def _structured(prompt: str, context: str, question_hint: str = "") -> str:
    """Run a one-shot non-streaming call returning the raw model output (expected to be JSON)."""
    user_content = f"BOOK EXCERPTS:\n{context}\n\nTOPIC FOCUS: {question_hint or '(general — pick the most central themes)'}"
    # Streaming → reliably returns text on the proxy, even when create() sometimes returns empty.
    messages = [{"role": "user", "content": user_content}]

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key:
        try:
            import anthropic
            base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip() or None
            client = anthropic.Anthropic(api_key=anthropic_key, base_url=base_url)
            parts: list[str] = []
            with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=prompt,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    parts.append(text)
            result = "".join(parts).strip()
            if result:
                return result
            print("[Claude structured-call returned empty — falling back to OpenAI]")
        except Exception as e:
            print(f"[Claude structured-call error: {e} — trying OpenAI...]")

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=2048,
            messages=[{"role": "system", "content": prompt}] + messages,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    raise RuntimeError("No API keys configured.")


def generate_quiz(context: str, topic: str = "") -> str:
    return _structured(QUIZ_PROMPT, context, topic)


def generate_flashcards(context: str, topic: str = "") -> str:
    return _structured(FLASHCARDS_PROMPT, context, topic)



SYSTEM_PROMPT = """You are an academic assistant helping a student study from their textbooks.

Rules:
- Answer using ONLY the book excerpts provided with each question. Do NOT use outside knowledge.
- If the answer is not in the excerpts, say exactly: "I couldn't find a clear answer to this in the provided book content."
- Always begin with a single concise sentence that captures the core point, then leave a blank line before the detailed explanation.
- Use bullet points, bold terms, or short paragraphs to make the answer easy to scan.
- You have access to the conversation history — use it to understand follow-up questions.
- NEVER reference the provided excerpts by number (do not write "(Excerpt 1)", "(Excerpt 2)" etc.). If you need to cite the book inline, refer to it by page number only (e.g. "page 42"). The user interface already shows the source page beneath your answer.

Format for derivations, proofs, and step-by-step problem solutions:
- When the student asks you to DERIVE, PROVE, SOLVE, or SHOW a mathematical result, structure your reply in two clearly separated parts.
- Part 1 — the theory / setup: explain what is being derived, define variables, state assumptions, and give any conceptual background. Keep this in normal paragraphs.
- Part 2 — the derivation itself: write a Markdown heading on its own line that is exactly "## Derivation" (or "## Proof" / "## Step-by-Step Solution" if more appropriate). Under that heading, present each step on its own numbered line, quoting the equations exactly as they appear in the book.
- For conceptual questions where no derivation is requested, do NOT include a "## Derivation" heading.

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


def stream_answer(context: str, question: str, history: list[dict] | None = None, eli5: bool = False):
    """Yields token strings. Falls back to OpenAI if Claude fails."""
    history = history or []
    messages = list(history) + [{"role": "user", "content": _user_message(context, question)}]
    system = SYSTEM_PROMPT + (ELI5_SUFFIX if eli5 else "")

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key:
        try:
            yield from _claude_stream(messages, anthropic_key, system=system)
            return
        except Exception as e:
            print(f"[Claude streaming error: {e} — trying OpenAI...]")

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        yield from _openai_stream(messages, openai_key, system=system)
        return

    raise RuntimeError("No API keys configured. Add ANTHROPIC_API_KEY or OPENAI_API_KEY to .env")


def _user_message(context: str, question: str) -> str:
    return f"BOOK EXCERPTS:\n{context}\n\nSTUDENT QUESTION:\n{question}"


def _claude_stream(messages: list[dict], api_key: str, system: str = None):
    import anthropic
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip() or None
    client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system or SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


def _openai_stream(messages: list[dict], api_key: str, system: str = None):
    from openai import OpenAI
    full_messages = [{"role": "system", "content": system or SYSTEM_PROMPT}] + messages
    client = OpenAI(api_key=api_key)
    for chunk in client.chat.completions.create(
        model="gpt-4o-mini", max_tokens=1024, messages=full_messages, stream=True
    ):
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def stream_smalltalk(question: str, history: list[dict] | None = None):
    """Yields token strings for greetings / small talk (no book context)."""
    history = history or []
    messages = list(history) + [{"role": "user", "content": question}]

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key:
        try:
            yield from _claude_stream(messages, anthropic_key, system=SMALLTALK_PROMPT)
            return
        except Exception as e:
            print(f"[Claude small-talk error: {e} — trying OpenAI...]")

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        yield from _openai_stream(messages, openai_key, system=SMALLTALK_PROMPT)
        return

    raise RuntimeError("No API keys configured.")


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
