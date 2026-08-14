"""Robust report section extraction helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence


TASK_SECTION_NAMES = ("FINDINGS", "IMPRESSION")

_INLINE_KNOWN_HEADING_RE = re.compile(r"(?i)(?<!^)(?<!\n)\b(FINDINGS|IMPRESSION)\s*(?:[:\-–—])")
_KNOWN_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(FINDINGS|IMPRESSION)\s*(?:[:\-–—]\s*)?(.*)$",
    re.IGNORECASE,
)
# Any "Word:"-shaped line, bare (``Lungs and Pleura:``) or inline (``COMPARISON: none.``), with an
# optional leading ATX prefix.  The name is captured non-greedily so it stops at the FIRST colon.
_SECTION_HEADING_RE = re.compile(r"^\s*(?:#{1,6}\s*)?([A-Z][A-Za-z0-9 /&()'.,\-–—]{0,60}?)\s*:\s*(.*)$")

# Report sections that are NOT scoring targets.  When one of these appears it genuinely ends the
# current FINDINGS/IMPRESSION section, so its body must not leak into the scored text.
#
# This list is deliberately CLOSED.  Any *other* ``Word:``-shaped line is body content — typically an
# organ sub-header ("Lungs and Pleura:", "Mediastinum and Heart:") in a structured-style report.  The
# previous implementation treated every unrecognised ``Word:`` line as a section terminator and reset
# the current section to ``None``, which silently DISCARDED the remainder of the section: FINDINGS
# parsed to "" on 16.1% of zero-shot `base` rows (80.9% for gemma-4-E2B-it) while IMPRESSION survived
# because its numbered-list items don't match the heading shape.  Because the loss was silent it read
# as poor model output, deflating base-stage scores and inflating the apparent fine-tuning gain.
# See tests/test_sections.py::test_organ_subheader_does_not_swallow_body.
_NON_TARGET_SECTION_NAMES = frozenset(
    {
        "ADDENDUM",
        "CLINICAL HISTORY",
        "CLINICAL INFORMATION",
        "COMPARISON",
        "COMPARISONS",
        "EXAM",
        "EXAMINATION",
        "HISTORY",
        "INDICATION",
        "NOTIFICATION",
        "PROCEDURE",
        "REASON FOR EXAM",
        "REASON FOR EXAMINATION",
        "RECOMMENDATION",
        "RECOMMENDATIONS",
        "REFERENCE",
        "TECHNIQUE",
        "WET READ",
    }
)

# Markdown ATX prefixes are formatting noise on body lines ("### Lungs:" -> "Lungs:").
_ATX_PREFIX_RE = re.compile(r"^#{1,6}\s*")

# Markdown emphasis runs (``**bold**``, ``*italic*``, ``* bullet``).  Models often wrap headings/body
# in these (e.g. ``**IMPRESSION:**``); left in place the ``*`` leaks into the parsed section text and
# even spawns a stray ``**`` "prelude" that lands in FINDINGS.  We strip asterisk emphasis before parsing.
# The asterisk run must sit at an emphasis boundary — NOT wedged between two alphanumerics — so a ``*``
# used as a "times" operator (``2*3 cm``, ``3*upper limit``) is preserved rather than silently fused into
# bogus numbers.  NOTE: underscores are deliberately NOT stripped — MIMIC de-identification uses ``___``
# placeholders (e.g. "INDICATION: ___ year old woman") that must survive.
# Known limitation: a body sentence that *starts* with an emphasised section word (``*FINDINGS* of ...``)
# loses the emphasis and may then be read as a heading; this shape is not expected in CXR generations.
_MARKDOWN_EMPHASIS_RE = re.compile(r"(?<![A-Za-z0-9])\*+|\*+(?![A-Za-z0-9])")


def _strip_markdown_emphasis(text: str) -> str:
    """Remove markdown asterisk emphasis (``*``/``**``) at emphasis boundaries.

    Preserves ``___`` de-id placeholders (underscores are never touched) and a ``*`` used as a
    multiplication operator between digits/letters (e.g. ``2*3``), which a naive ``\\*+`` strip would
    corrupt into ``23``.
    """
    return _MARKDOWN_EMPHASIS_RE.sub("", text)


@dataclass(frozen=True)
class ReportSections:
    sections: dict[str, str]
    method: str
    headings: tuple[str, ...]


def normalize_report_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_report_texts(texts: Sequence[str]) -> list[str]:
    """Batch version of :func:`normalize_report_text`."""
    return [normalize_report_text(text) for text in texts]


def _split_paragraphs(text: str) -> list[str]:
    return [normalize_report_text(paragraph) for paragraph in re.split(r"\n\s*\n+", text or "") if paragraph.strip()]


def extract_report_sections(text: str) -> ReportSections:
    if not text:
        return ReportSections(sections={name: "" for name in TASK_SECTION_NAMES}, method="empty", headings=())

    text = _strip_markdown_emphasis(text)
    preprocessed_text = _INLINE_KNOWN_HEADING_RE.sub(lambda match: "\n" + match.group(1).upper() + ":", text)
    lines = preprocessed_text.replace("\r\n", "\n").split("\n")

    sections: dict[str, list[str]] = {name: [] for name in TASK_SECTION_NAMES}
    current_section: str | None = None
    first_section: str | None = None
    prelude_lines: list[str] = []
    found_headings: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if current_section is not None:
                sections[current_section].append("")
            elif first_section is None:
                prelude_lines.append("")
            continue

        heading_match = _KNOWN_HEADING_RE.match(line)
        if heading_match:
            current_section = heading_match.group(1).upper()
            if current_section not in found_headings:
                found_headings.append(current_section)
            if first_section is None:
                first_section = current_section
            remainder = heading_match.group(2).strip()
            if remainder:
                sections[current_section].append(remainder)
            continue

        # Only a KNOWN non-target section ends the current section. Every other "Word:" line is
        # body content (organ sub-headers etc.) and must be kept — see _NON_TARGET_SECTION_NAMES.
        section_match = _SECTION_HEADING_RE.match(line)
        if section_match and section_match.group(1).strip().upper() in _NON_TARGET_SECTION_NAMES:
            current_section = None
            continue

        body_line = _ATX_PREFIX_RE.sub("", line)
        if current_section is not None:
            sections[current_section].append(body_line)
        elif first_section is None:
            prelude_lines.append(body_line)

    if not found_headings:
        paragraphs = _split_paragraphs(text)
        if len(paragraphs) >= 2:
            return ReportSections(
                sections={"FINDINGS": paragraphs[0], "IMPRESSION": paragraphs[-1]},
                method="heuristic",
                headings=(),
            )
        if len(paragraphs) == 1:
            return ReportSections(
                sections={"FINDINGS": paragraphs[0], "IMPRESSION": ""}, method="heuristic", headings=()
            )
        return ReportSections(sections={name: "" for name in TASK_SECTION_NAMES}, method="empty", headings=())

    prelude_text = normalize_report_text(" ".join(part for part in prelude_lines if part))
    if prelude_text:
        if not sections["FINDINGS"]:
            sections["FINDINGS"].insert(0, prelude_text)
        elif not sections["IMPRESSION"]:
            sections["IMPRESSION"].insert(0, prelude_text)

    normalized_sections = {name: normalize_report_text(" ".join(parts)) for name, parts in sections.items()}
    return ReportSections(sections=normalized_sections, method="explicit", headings=tuple(found_headings))


def extract_section(source: str | Mapping[str, str], section_name: str) -> str:
    section_key = section_name.upper()
    if isinstance(source, Mapping):
        return normalize_report_text(str(source.get(section_key, "")))

    return extract_report_sections(str(source)).sections.get(section_key, "")
