from app.schemas.memory import MemoryChunk, StoredUtterance

ROLE_LABELS = {"me": "나", "customer": "고객", "unknown": "미확인"}
KIND_LABELS = {"meeting": "미팅", "call": "통화"}


def with_context(text: str, company: str, kind: str, occurred_at: str, title: str) -> str:
    """Prefix a chunk with where it came from before embedding.

    Every chunk of a customer then carries the company name, so a question that
    names the company no longer favors the few chunks that happen to say it.
    """
    header = " | ".join([company, f"{occurred_at[:10]} {KIND_LABELS.get(kind, kind)}", title])
    return f"{header}\n{text}"


def format_utterance(utterance: StoredUtterance) -> str:
    return f"[{ROLE_LABELS[utterance.role]}] {utterance.text}"


def chunk_utterances(
    utterances: list[StoredUtterance],
    max_chars: int = 500,
    overlap: int = 1,
) -> list[MemoryChunk]:
    """Group consecutive utterances into search chunks of about max_chars.

    Each chunk repeats the last `overlap` utterances of the previous one so an
    answer split across a boundary still appears whole in one chunk. A single
    utterance longer than max_chars becomes its own chunk.
    """
    lines = [format_utterance(u) for u in utterances]
    chunks: list[MemoryChunk] = []
    start = 0
    while start < len(utterances):
        end = start + 1
        size = len(lines[start])
        while end < len(utterances) and size + 1 + len(lines[end]) <= max_chars:
            size += 1 + len(lines[end])
            end += 1
        window = utterances[start:end]
        chunks.append(
            MemoryChunk(
                text="\n".join(lines[start:end]),
                start=window[0].start,
                end=window[-1].end,
                utterance_ids=[u.id for u in window],
            )
        )
        if end == len(utterances):
            break
        start = max(start + 1, end - overlap)
    return chunks
