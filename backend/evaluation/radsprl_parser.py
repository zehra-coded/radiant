"""
Parser for the Rad-SpRL XML corpus.

Two coordinate systems coexist in this file:

- Character offsets: `Token` elements have cs/cl (character start and
  character length) relative to the document text.
- Token offsets: `Landmark`, `Trajector`, `Diagnosis`, `Hedge`, and
  `Sentence` elements have ts/tl (token start and token length)
  relative to the ordered sequence of Token elements.

To resolve a span's text, we build the token index first, then map
ts/tl to character ranges via the tokens.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

from backend.core.logging import get_logger

logger = get_logger(__name__)

_WANTED_TAGS = frozenset({"Diagnosis", "Hedge", "Landmark", "Trajector"})


@dataclass
class RadSpRLSpan:
    tag: str                 # Diagnosis | Hedge | Landmark | Trajector
    span_type: str | None    # from the `type` attribute, if present
    char_start: int          # character offset in the document text
    char_length: int
    text: str = ""

    @property
    def end(self) -> int:
        return self.char_start + self.char_length


@dataclass
class RadSpRLDocument:
    doc_id: str
    text: str
    spans: list[RadSpRLSpan] = field(default_factory=list)

    def spans_of(self, tag: str) -> list[RadSpRLSpan]:
        return [s for s in self.spans if s.tag == tag]


def parse(xml_path: Path) -> list[RadSpRLDocument]:
    """Parse the Rad-SpRL XML into structured documents."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    if root.tag != "Corpus":
        raise ValueError(f"Expected root <Corpus>, got <{root.tag}>")

    documents = root.findall("Document")
    logger.info("found %d <Document> elements", len(documents))

    docs: list[RadSpRLDocument] = []
    for doc in documents:
        doc_id = doc.get("id", "")
        text_el = doc.find("Text")
        text = text_el.text if text_el is not None and text_el.text else ""

        ann_el = doc.find("Annotations")
        if ann_el is None:
            docs.append(RadSpRLDocument(doc_id=doc_id, text=text, spans=[]))
            continue

        # Build the token index: ordered list of (char_start, char_length).
        tokens: list[tuple[int, int]] = []
        for child in ann_el:
            if child.tag == "Token":
                try:
                    cs = int(child.get("cs", "0"))
                    cl = int(child.get("cl", "0"))
                except ValueError:
                    continue
                tokens.append((cs, cl))

        def token_span_to_char(ts: int, tl: int) -> tuple[int, int] | None:
            """Convert a token start/length to a character start/length."""
            if ts < 0 or tl <= 0 or ts + tl > len(tokens):
                return None
            start_char = tokens[ts][0]
            last_tok = tokens[ts + tl - 1]
            end_char = last_tok[0] + last_tok[1]
            return start_char, end_char - start_char

        spans: list[RadSpRLSpan] = []
        for child in ann_el.iter():
            if child.tag not in _WANTED_TAGS:
                continue
            ts_raw = child.get("ts")
            tl_raw = child.get("tl")
            if ts_raw is None or tl_raw is None:
                continue
            try:
                ts = int(ts_raw)
                tl = int(tl_raw)
            except ValueError:
                continue
            char_range = token_span_to_char(ts, tl)
            if char_range is None:
                continue
            cs, cl = char_range
            span_text = text[cs:cs + cl] if 0 <= cs < len(text) else ""
            spans.append(
                RadSpRLSpan(
                    tag=child.tag,
                    span_type=child.get("type"),
                    char_start=cs,
                    char_length=cl,
                    text=span_text,
                )
            )

        docs.append(RadSpRLDocument(doc_id=doc_id, text=text, spans=spans))

    total_spans = sum(len(d.spans) for d in docs)
    logger.info("parsed %d documents, total %d spans", len(docs), total_spans)
    return docs


def summary(docs: list[RadSpRLDocument]) -> dict:
    counts: dict[str, int] = {}
    for doc in docs:
        for span in doc.spans:
            counts[span.tag] = counts.get(span.tag, 0) + 1
    return counts