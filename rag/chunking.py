from email.mime import text


def chunk_by_paragraph(text: str) -> list[str]:
    paragraphs = text.split("\n\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs