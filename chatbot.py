import sys
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()

import registry
from config import MAX_HISTORY_TURNS
from ingest import ingest_pdf, remove_book
from llm import get_answer
from query import contextualize_query, format_context, format_sources, search

DIVIDER = "─" * 62


@click.group()
def cli():
    """BookBot — Ask questions from your textbooks, grounded in the source."""
    pass


@cli.command()
@click.argument("pdf_path")
@click.option("--subject", "-s", required=True, help="Subject name  (e.g. 'Software Engineering')")
@click.option("--title", "-t", required=True, help="Book title")
@click.option("--author", "-a", required=True, help="Author name")
def ingest(pdf_path, subject, title, author):
    """Ingest a PDF book into the knowledge base."""
    ingest_pdf(pdf_path, subject, title, author)


@cli.command(name="books")
def list_books():
    """List all ingested books grouped by subject."""
    all_books = registry.load()
    if not all_books:
        click.echo("No books ingested yet.\n\nRun:  python chatbot.py ingest <pdf> -s <subject> -t <title> -a <author>")
        return

    from collections import defaultdict
    by_subject: dict[str, list] = defaultdict(list)
    for b in all_books:
        by_subject[b["subject"]].append(b)

    for subject in sorted(by_subject):
        click.echo(f"\n{subject}")
        for b in by_subject[subject]:
            click.echo(f"  • {b['title']}  —  {b['author']}  ({b['pages']} pages, {b['chunks']} chunks)")
    click.echo()


@cli.command()
@click.argument("book_id")
def remove(book_id):
    """Remove a book from the knowledge base (use the book_id shown in `books`)."""
    remove_book(book_id)


@cli.command()
@click.option("--subject", "-s", default=None, help="Subject to query (skips subject selection prompt)")
def ask(subject):
    """Start an interactive Q&A session."""

    subjects = registry.all_subjects()
    if not subjects:
        click.echo("No books ingested yet. Run `python chatbot.py ingest` first.")
        sys.exit(1)

    # Subject selection
    if subject:
        matched = next((s for s in subjects if s.lower() == subject.lower()), None)
        if not matched:
            click.echo(f"Subject '{subject}' not found. Available subjects:")
            for s in subjects:
                click.echo(f"  • {s}")
            sys.exit(1)
        subject = matched
    else:
        click.echo("\nAvailable subjects:")
        for i, s in enumerate(subjects, 1):
            click.echo(f"  [{i}] {s}")
        idx = click.prompt("Select subject", type=int, default=1)
        if not 1 <= idx <= len(subjects):
            click.echo("Invalid selection.")
            sys.exit(1)
        subject = subjects[idx - 1]

    # Book selection
    books = registry.by_subject(subject)
    click.echo(f"\nBooks available for '{subject}':")
    click.echo("  [0] Search ALL books")
    for i, b in enumerate(books, 1):
        click.echo(f"  [{i}] {b['title']}  —  {b['author']}")

    choice = click.prompt("Select book", type=int, default=0)
    if choice == 0:
        selected_book_id = None
        scope_label = f"all books in '{subject}'"
    elif 1 <= choice <= len(books):
        selected_book_id = books[choice - 1]["book_id"]
        scope_label = f"\"{books[choice - 1]['title']}\" by {books[choice - 1]['author']}"
    else:
        click.echo("Invalid selection.")
        sys.exit(1)

    click.echo(f"\nSearching in: {scope_label}")
    click.echo("Type your question or 'exit' to quit.\n")

    history: list[dict] = []

    while True:
        try:
            question = click.prompt("You").strip()
        except (EOFError, KeyboardInterrupt):
            click.echo("\nGoodbye!")
            break

        if question.lower() in ("exit", "quit", "q", "bye"):
            click.echo("Goodbye!")
            break
        if not question:
            continue

        # Expand short follow-up questions before embedding search.
        search_query = contextualize_query(question, history)

        chunks = search(search_query, subject, book_id=selected_book_id)
        if not chunks:
            click.echo("\nNo relevant content found for that question.\n")
            continue

        context = format_context(chunks)
        try:
            answer, model = get_answer(context, question, history=history)
        except RuntimeError as e:
            click.echo(f"\nError: {e}\n")
            continue

        click.echo(f"\n{DIVIDER}")
        click.echo(answer)
        click.echo(f"\nSources:")
        click.echo(format_sources(chunks))
        click.echo(f"[Answered by {model}]")
        click.echo(f"{DIVIDER}\n")

        # Keep history trimmed to MAX_HISTORY_TURNS pairs.
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        if len(history) > MAX_HISTORY_TURNS * 2:
            history = history[-(MAX_HISTORY_TURNS * 2):]


if __name__ == "__main__":
    cli()
