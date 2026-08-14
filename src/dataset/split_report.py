import re
import functools

from .constants import LABELS, REPORTS_BASE_PATH

# Build a regex pattern to find each section
LABELS_PATTERN = r"(?=(" + "|".join(LABELS) + r")\s*:)"  # lookahead for section names


@functools.lru_cache(maxsize=1024)
def split_report(report_path_extension: str) -> dict:
    with open(REPORTS_BASE_PATH + report_path_extension, "r") as file:
        report_content = file.read()

    sections_split = re.split(LABELS_PATTERN, report_content)

    sections = {}
    for i in range(1, len(sections_split), 2):
        header = sections_split[i].strip()
        content = re.sub(r"\s+", " ", sections_split[i + 1]).strip()
        sections[header] = content

    return {name: sections.get(name, "") for name in LABELS}
