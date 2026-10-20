from email.mime import text


def chunk_by_paragraph(text: str) -> list[str]:
    paragraphs = text.split("\n\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


def create_chunk_records(
    text: str,
    source: str
) -> list[dict]:

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    records = []

    for i, paragraph in enumerate(paragraphs):
        records.append(
            {
                "id": i,
                "text": paragraph,
                "source": source
            }
        )

    return records