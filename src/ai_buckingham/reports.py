"""Read completed system cards for notebooks, Streamlit and static reports."""
from __future__ import annotations

import ast
import base64
from dataclasses import dataclass, field
from functools import lru_cache
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import tempfile

from .artifacts import run_figures


@dataclass
class _Element:
    tag: str
    attrs: list[tuple[str, str | None]] = field(default_factory=list)
    children: list = field(default_factory=list)

    def classes(self):
        return dict(self.attrs).get("class", "").split()

    def text(self):
        return "".join(child.text() if isinstance(child, _Element) else html.unescape(child)
                        for child in self.children).strip()

    def find(self, class_name):
        if class_name in self.classes():
            return self
        for child in self.children:
            if isinstance(child, _Element):
                found = child.find(class_name)
                if found is not None:
                    return found
        return None

    def find_all(self, class_name):
        found = [self] if class_name in self.classes() else []
        for child in self.children:
            if isinstance(child, _Element):
                found.extend(child.find_all(class_name))
        return found


_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


class _CardParser(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=False)
        self.root = _Element("div")
        self.stack = [self.root]
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        element = _Element(tag, attrs)
        self.stack[-1].children.append(element)
        if tag not in _VOID:
            self.stack.append(element)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(_Element(tag, attrs))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)

    def handle_entityref(self, name):
        self.handle_data("&" + name + ";")

    def handle_charref(self, name):
        self.handle_data("&#" + name + ";")


_OMITTED_DETAILS = {
    "Ledger identity and advanced structured diagnostics",
    "Raw Analyzer and compiled contract",
    "Advanced structured diagnostics",
    "Raw Checker response and program gate",
    "Actual retrieval queries and reference records",
    "Exact evidence supplied to Checker",
    "Exact feedback returned to Analyzer",
}


_PLANNING_FIELDS = (
    "Protocol", "Similarity", "Output optimization",
    "Similarity decision · relative MI threshold", "Input constraints",
    "Output constraints", "Manual overrides",
)
_SCREENING_FIELDS = (
    "Decision", "Screening k", "MI · fixed output", "MI · joint output",
    "Relative MI loss without output scaling", "Relative MI threshold",
    "Screening cosine threshold", "Fixed basis powers during screening",
    "Final input constraints", "Final output constraints",
)
_REDUNDANT_COPY = {
    "Rows and columns follow the labels above; zero rows may be omitted. Counts are computed from this matrix, not inferred separately by the model.",
    "Checker decisions are model judgments. The program gate is reported separately above; ? denotes an unknown coordinate, never zero.",
    "A candidate representation was optimized and checked against restart and held-out evidence.",
    "Physical exponents are display-normalized; basis-coordinate powers are the raw optimizer coefficients.",
    "Count selection, parsimony checks, and formula refinement are summarized below.",
    "The physical and basis-coordinate forms are scale-equivalent representations of the selected group.",
    "Final relations are copied directly from the numerical state. Each relation is shown both in physical variables and in the optimized Buckingham-Pi basis coordinates.",
    "These are basis generators, not yet the final governing groups selected by MI/CMA-ES.",
}

_READER_LABELS = {
    "Program-compiled dimensional matrix": "Dimensional matrix",
    "Physical input ledger": "Physical quantities and dimensions",
    "CSV input": "Input variable",
    "Named SI dimensions": "SI dimensions",
    "Checker decision": "Dimensional-checking decision",
    "No structured Checker reason was returned.": "No dimensional-checking explanation is available.",
    "? means an unknown coordinate; it is never replaced by zero. A value of 0 is known absence of that SI base.": "? denotes an unspecified SI exponent.",
}


def _promote_metrics(card, fields):
    """Move retained archived detail rows into the card's existing metric grid."""
    grid = card.find("daa-metrics")
    if grid is None:
        return
    values = {}
    for metric in grid.find_all("daa-metric"):
        key, value = metric.find("daa-metric-key"), metric.find("daa-metric-value")
        if key is not None and value is not None:
            values[" ".join(key.text().split())] = value.children
    for detail in card.find_all("daa-details"):
        tables = [child for child in detail.children if isinstance(child, _Element) and child.tag == "table"]
        for table in tables:
            for row in table.children:
                if not isinstance(row, _Element) or row.tag != "tr":
                    continue
                cells = [child for child in row.children if isinstance(child, _Element) and child.tag == "td"]
                if len(cells) == 2:
                    values[" ".join(cells[0].text().split())] = cells[1].children
    # A failed or partial stage may only have its status and failure message.
    if fields[0] not in values:
        return
    grid.attrs = [("class", "daa-metrics daa-metrics-wide")]
    grid.children = [
        _Element("div", [("class", "daa-metric")], [
            _Element("div", [("class", "daa-metric-key")], [html.escape(key)]),
            _Element("div", [("class", "daa-metric-value")], values[key]),
        ])
        for key in fields if key in values
    ]
    card.children = [child for child in card.children
                     if not isinstance(child, _Element)
                     or not {"daa-summary", "daa-note", "daa-details"}.intersection(child.classes())]


def _readable(element):
    """Keep card content while making every included detail immediately visible."""
    if isinstance(element, str):
        return html.escape(_READER_LABELS[html.unescape(element)]) if html.unescape(element) in _READER_LABELS else element
    if element.tag in {"script", "iframe", "style"} or "daa-card-label" in element.classes():
        return ""
    text = " ".join(element.text().split())
    if "daa-metric" in element.classes():
        key = element.find("daa-metric-key")
        if key is not None and key.text() == "Output inference":
            return ""
    if text in _REDUNDANT_COPY or ("daa-diagnostic-note" in element.classes() and text.startswith("Termination:")):
        return ""
    if element.tag == "tr":
        cells = [child for child in element.children if isinstance(child, _Element) and child.tag in {"td", "th"}]
        if cells and cells[0].text() == "Restart stability":
            return ""
    if element.tag == "details":
        summary = next((child for child in element.children
                        if isinstance(child, _Element) and child.tag == "summary"), None)
        if summary is not None and summary.text() in _OMITTED_DETAILS:
            return ""
        body = "".join(_readable(child) for child in element.children if child is not summary)
        if not body.strip():
            return ""
        heading = _readable(summary) if summary is not None else ""
        return '<section class="daa-flat-details">' + heading + body + "</section>"
    tag = "h4" if element.tag == "summary" else element.tag
    attrs = [(key, value) for key, value in element.attrs if not key.lower().startswith("on")]
    attributes = "".join(" " + key + ('="' + html.escape(value, quote=True) + '"' if value is not None else "")
                         for key, value in attrs)
    start = "<" + tag + attributes + ">"
    if tag in _VOID:
        return start
    body = "".join(_readable(child) for child in element.children)
    if tag in {"table", "thead", "tbody", "tr"} and not body.strip():
        return ""
    return start + body + "</" + tag + ">"


_STAGES = [
    ("results", "Selected dimensionless relations"),
    ("formulation", "Problem formulation"),
    ("dimensions", "Physical consistency and dimensional basis"),
    ("planning", "Analysis protocol"),
    ("regimes", "Physical-regime identification"),
    ("learning", "Dimensionless learning and simplification"),
]


def report_sections(run_dir: Path | str) -> list[dict]:
    """Return the last recorded card for each scientific stage and local scope.

    Reading cards never runs a model, optimizer or recorder, and never writes to
    the run. The last snapshot preserves failed or partial states as recorded.
    """
    source = Path(run_dir) / "evidence" / "display_bundles.json"
    bundles = json.loads(source.read_text(encoding="utf-8")) if source.is_file() else []
    cards = {}
    for bundle in bundles:
        parsed = _CardParser(bundle.get("text/html", ""))
        for card in parsed.root.find_all("daa-card"):
            title_element = card.find("daa-card-title")
            if title_element is None:
                continue
            title = " ".join(title_element.text().split())
            if title == "Selected dimensionless relations":
                stage = "results"
            elif title == "Problem formulation":
                stage = "formulation"
            elif title in {"Physical consistency", "Dimensional basis"}:
                stage = "dimensions"
            elif title == "Analysis protocol":
                stage = "planning"
                _promote_metrics(card, _PLANNING_FIELDS)
            elif title in {"Mechanism evidence", "Physical-regime identification"}:
                stage = "regimes"
                title_element.children = ["Physical-regime identification"]
            elif title.startswith("Similarity-class screening") or "candidate k=" in title or title in {"global", "Local model selection"} or title.startswith("regime_"):
                stage = "learning"
                if title.startswith("Similarity-class screening"):
                    _promote_metrics(card, _SCREENING_FIELDS)
            else:
                continue
            cards[title] = (stage, _readable(card))
    sections = []
    for stage, title in _STAGES:
        content = "".join(content for group, content in cards.values() if group == stage)
        if content:
            sections.append({"id": stage, "title": title,
                             "html": '<div class="daa-shell">' + content + "</div>"})
    return sections


@lru_cache(maxsize=1)
def report_styles() -> str:
    """Reuse the system card stylesheet without importing its numerical code."""
    module = ast.parse(Path(__file__).with_name("presentation.py").read_text(encoding="utf-8"))
    assignment = next(node for node in module.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "_AGENT_EDITORIAL_CSS"
                              for target in node.targets))
    styles = ast.literal_eval(assignment.value)
    return styles + """<style>
    .daa-flat-details {margin:1rem 0;}
    .daa-flat-details > h4 {margin:.75rem 0 .4rem;font-size:1rem;}
    .daa-shell table {width:100%;border-collapse:collapse;}
    .daa-shell td,.daa-shell th {overflow-wrap:anywhere;vertical-align:top;}
    .daa-shell .daa-front-scroll {overflow-x:auto;}
    .daa-report-figure {margin:1.5rem 0;}
    .daa-report-figure img {width:100%;height:auto;}
    .daa-report-stage > summary, .daa-report-stage h3 {overflow-wrap:anywhere;}
    .daa-report-stage {margin:1.25rem 0;border:1px solid #d9dee5;padding:1rem;}
    .daa-report-stage > summary {font:600 1.15rem Georgia,serif;cursor:pointer;}
    </style>"""


def _run_title(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    case_id = manifest["case_id"]
    metadata = Path(__file__).resolve().parents[2] / "docs" / "paper_cases.json"
    if metadata.is_file():
        cases = json.loads(metadata.read_text(encoding="utf-8"))["cases"]
        return next((case["title"] for case in cases if case["id"] == case_id), case_id)
    return case_id


def report_figures(run_dir: Path | str) -> list[Path]:
    """Prefer saved plot PNGs over captured copies in a run's presentation."""
    images = [path for path in run_figures(run_dir) if path.suffix.lower() == ".png"]
    originals = [path for path in images if not path.name.startswith("display_")]
    return originals or images


def render_report(run_dir: Path | str) -> str:
    """Render the selected run's scientific cards and figures as standalone HTML."""
    run_dir = Path(run_dir)
    title = html.escape(_run_title(run_dir))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    status = manifest.get("status", "unknown")
    run_status = ('<p class="daa-run-status">Run status: ' + html.escape(str(status)) + '</p>'
                  if status != "completed" else "")
    sections = report_sections(run_dir)
    content = [section["html"] for section in sections if section["id"] == "results"]
    if not sections:
        content.append("<p>Scientific stage cards are unavailable for this run.</p>")
    for figure in report_figures(run_dir):
        encoded = base64.b64encode(figure.read_bytes()).decode("ascii")
        content.append('<figure class="daa-report-figure"><img src="data:image/png;base64,'
                       + encoded + '" alt="' + html.escape(figure.stem.replace("_", " "), quote=True)
                       + '"></figure>')
    for section in sections:
        if section["id"] != "results":
            content.append('<details class="daa-report-stage" id="' + section["id"] + '"><summary>'
                           + html.escape(section["title"]) + '</summary>' + section["html"] + '</details>')
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>' + title + '</title>' + report_styles()
            + '<style>body{font:16px/1.5 Arial,sans-serif;margin:2rem auto;max-width:1180px;padding:0 1.25rem;color:#17202a}h1{font-family:Georgia,serif;font-size:2rem;overflow-wrap:anywhere}</style>'
            '</head><body><h1>' + title + '</h1>' + run_status + "".join(content) + "</body></html>")


def export_report(run_dir: Path | str, output_path: Path | str) -> Path:
    """Write the derived reader report atomically, preserving the raw report."""
    run_dir = Path(run_dir).resolve()
    output_path = Path(output_path)
    if output_path.resolve() == run_dir / "report.html":
        raise ValueError("The original run report cannot be replaced by a reader report")
    page = render_report(run_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=output_path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(page)
        os.replace(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output_path
