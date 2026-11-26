"""Build static case pages from figures and saved structured results.

Only the Python standard library is used. The builder reads published files;
it does not import the scientific engine or execute a numerical analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
from urllib.parse import quote, unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = Path(__file__).with_name("pages")
REPOSITORY = "https://github.com/ChengAo21/ai-buckingham"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_source(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError(f"Missing or non-portable published source: {relative}")
    return path


def github(path: str, *, repository: str, branch: str, directory=False) -> str:
    kind = "tree" if directory else "blob"
    return f"{repository}/{kind}/{quote(branch, safe='')}/{quote(path, safe='/')}"


def link(url: str, label: str, class_name: str = "") -> str:
    attribute = f' class="{html.escape(class_name)}"' if class_name else ""
    return f'<a{attribute} href="{html.escape(url, quote=True)}">{html.escape(label)}</a>'


def formula(value: str) -> str:
    escaped = html.escape(value)
    return re.sub(r"\^\{([^{}]+)\}", r"<sup>\1</sup>", escaped).replace(" * ", " · ")


def selected_metrics(state: dict) -> list[dict]:
    metrics = []
    for scope in state["workflow_result"]["local_workflows"]:
        if scope.get("status") != "success":
            raise ValueError("A selected case contains an unsuccessful local workflow")
        final = scope["final_solution"]
        scope_id = scope["scope_id"]
        metrics.append({
            "scope_id": scope_id,
            "scope": "Global" if scope_id == "global" else scope_id.replace("_", " ").capitalize(),
            "samples": scope["n_samples"],
            "groups": scope["locked_count"],
            "mutual_information": final["mutual_information"],
            "epsilon": final["normalized_information_score"],
            "conditional_test_r2": final["gpr_validation"]["test_r2_mean"],
            "input_formulas": final["input_formulas"],
            "output_formula": final["output_dimensionless_formula"],
        })
    if not metrics:
        raise ValueError("A selected case has no structured numerical results")
    return metrics


def metric_table(metrics: list[dict]) -> str:
    rows = "".join(
        f'<tr><th scope="row">{html.escape(item["scope"])}</th>'
        f'<td>{item["samples"]:,}</td><td>{item["groups"]}</td>'
        f'<td>{item["mutual_information"]:.3f}</td><td>{item["epsilon"]:.3f}</td>'
        f'<td>{item["conditional_test_r2"]:.3f}</td></tr>'
        for item in metrics
    )
    return ('<div class="table-scroll"><table class="metric-table">'
            '<thead><tr><th scope="col">Scope</th><th scope="col">Samples</th>'
            '<th scope="col">Groups</th><th scope="col">MI</th>'
            '<th scope="col">ε</th><th scope="col">Test R²</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>')


def selected_relations(metrics: list[dict]) -> str:
    rows = []
    for item in metrics:
        inputs = "".join(
            f'<div class="formula"><span>Input {number}</span>{formula(value)}</div>'
            for number, value in enumerate(item["input_formulas"], 1)
        )
        rows.append(f'<div class="formula-row"><h4>{html.escape(item["scope"])}</h4>'
                    + inputs
                    + f'<div class="formula"><span>Output</span>{formula(item["output_formula"])}</div></div>')
    return "".join(rows)


def case_figure(entry: dict, prefix: str = "") -> str:
    image = html.escape(prefix + entry["image"], quote=True)
    title = html.escape(entry["title"], quote=True)
    return (f'<figure class="case-figure"><a href="{image}">'
            f'<img src="{image}" alt="{title}" width="1600" height="760" loading="lazy" decoding="async">'
            '</a></figure>')


def case_article(entry: dict) -> str:
    actions = "".join([
        link(entry["full_card"], "Result card", "button"),
        link(entry["notebook"], "Notebook", "button button-outline"),
        link(entry["evidence"], "Data", "button button-outline"),
    ])
    return f'''<article class="case-panel" id="case-{entry["case_id"]}">
  <div class="case-heading"><h3>{html.escape(entry["title"])}</h3></div>
  {case_figure(entry)}
  <div class="case-actions">{actions}</div>
  <details class="result-details"><summary>Numerical results and selected relations</summary>{metric_table(entry["metrics"])}{selected_relations(entry["metrics"])}</details>
</article>'''


def public_card(entry: dict, paper: dict, repository: str) -> str:
    title = html.escape(entry["title"])
    source_links = [
        link(entry["notebook"], "Notebook"),
        link(entry["dataset"], "Dataset · CSV"),
        link(entry["state"], "State · JSON"),
        link(entry["config"], "Case settings · JSON"),
        link(entry["evidence"], "Run data"),
    ]
    if entry["optimizer_trace"]:
        source_links.append(link(entry["optimizer_trace"], "Optimizer trace · JSONL"))
    tables = "".join(f'<li>{link(item["url"], item["name"])}</li>' for item in entry["tables"])
    table_links = f'<h3>Tables</h3><ul class="source-files">{tables}</ul>' if tables else ""
    return f'''<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} · AI Buckingham</title><link rel="stylesheet" href="../assets/style.css"></head>
<body>
  <a class="skip-link" href="#main">Skip to results</a>
  <header class="site-header"><div class="header-inner"><a class="wordmark" href="../index.html">AI Buckingham</a><nav aria-label="Main navigation"><a href="../index.html#cases">Cases</a>{link(paper["url"], "Paper")}{link(repository, "GitHub")}</nav></div></header>
  <main class="card-page" id="main">
    <header class="card-heading"><p class="case-meta">{html.escape(entry["tag"])}</p><h1>{title}</h1><p>{link('../index.html#case-' + entry['case_id'], 'All cases')}</p></header>
    {case_figure(entry, "../")}
    <section class="card-section"><h2>Numerical results</h2>{metric_table(entry["metrics"])}</section>
    <section class="card-section"><h2>Selected relations</h2>{selected_relations(entry["metrics"])}</section>
    <section class="card-section"><h2>Data and settings</h2><div class="source-links">{"".join(source_links)}</div>{table_links}</section>
  </main>
</body>
</html>
'''


def remove_previous_generated_files(output: Path) -> None:
    """Remove only paths recorded by a previous invocation of this builder."""
    manifest = output / ".pages-build.json"
    if not manifest.is_file():
        return
    for relative in read_json(manifest).get("files", []):
        path = (output / relative).resolve()
        if not path.is_relative_to(output.resolve()) or path == output.resolve():
            raise ValueError("Previous Pages build contains an unsafe output path")
        if path.is_file():
            path.unlink()


def build(output: Path, repository: str, branch: str) -> dict:
    output = output.resolve()
    if not output.is_relative_to(ROOT.resolve()) or output == ROOT.resolve():
        raise ValueError("Pages output must be a dedicated directory inside this repository")
    if output.parts[len(ROOT.resolve().parts)] in {
        "src", "scripts", "results", "configs", "data", "knowledge", "reproduction",
        "notebooks", "docs", "tests", "environments", ".git", ".github",
    }:
        raise ValueError("Pages output cannot overwrite published source or evidence")
    specification = read_json(local_source("docs/paper_cases.json"))
    paper = specification["paper"]
    groups = specification["groups"]
    cases = specification["cases"]
    group_ids = {group["id"] for group in groups}
    case_ids = {case["id"] for case in cases}
    if len(cases) != 10 or len(case_ids) != len(cases) or len(group_ids) != len(groups):
        raise ValueError("Paper metadata must identify ten cases and unique groups")
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in case_ids | group_ids):
        raise ValueError("Paper metadata contains an invalid case or group identifier")
    if any(case["group"] not in group_ids for case in cases):
        raise ValueError("A paper case has no matching group")
    index = read_json(local_source("results/index.json"))
    data_manifest = read_json(local_source("data/manifest.json"))
    by_id = {case["case_id"]: case for case in index["cases"]}
    if set(by_id) != case_ids:
        raise ValueError("Paper metadata and the result index identify different cases")
    assets = [
        (local_source("docs/assets/workflow.png"), "assets/workflow.png"),
        (local_source("docs/assets/manifest.json"), "assets/manifest.json"),
        (local_source("docs/assets/cases/manifest.json"), "assets/cases/manifest.json"),
    ]
    source_files = [local_source("docs/paper_cases.json"), local_source("results/index.json")]
    entries = []
    total_generations = 0
    for published in cases:
        case_id = published["id"]
        case = by_id[case_id]
        selected = [run for run in case["runs"] if run["run_id"] == case["selected_run"]]
        if len(selected) != 1 or selected[0]["status"] != "completed":
            raise ValueError(f"Case {case_id} does not select one completed run")
        run = selected[0]
        state_path = local_source(f"{run['directory']}/state.json")
        config_relative = f"configs/cases/{case_id}.json"
        config_path = local_source(config_relative)
        config = read_json(config_path)
        dataset = local_source(config["dataset"]).relative_to(ROOT).as_posix()
        image_relative = f"docs/assets/cases/{case_id}.png"
        if published["image"] != image_relative:
            raise ValueError(f"Unexpected case image path for {case_id}")
        image = f"assets/cases/{case_id}.png"
        assets.append((local_source(image_relative), image))
        source_files.extend([state_path, config_path])
        trace_relative = f"{run['directory']}/evidence/optimization_trace.jsonl"
        trace = github(trace_relative, repository=repository, branch=branch) if (ROOT / trace_relative).is_file() else None
        tables = [{"name": path.name, "url": github(path.relative_to(ROOT).as_posix(), repository=repository, branch=branch)}
                  for path in sorted((state_path.parent / "tables").glob("*.csv"))]
        group = next(item for item in groups if item["id"] == published["group"])
        entries.append({
            "case_id": case_id, "title": published["title"], "short_name": published["title"],
            "group": published["group"], "tag": group["title"], "section": published["section"],
            "figure": published["figure"], "image": image, "samples": config["data_metadata"]["rows"],
            "run_id": run["run_id"], "full_card": f"cards/{case_id}.html",
            "notebook": github(case["notebook"], repository=repository, branch=branch),
            "evidence": github(run["directory"], repository=repository, branch=branch, directory=True),
            "dataset": github(dataset, repository=repository, branch=branch),
            "state": github(state_path.relative_to(ROOT).as_posix(), repository=repository, branch=branch),
            "config": github(config_relative, repository=repository, branch=branch),
            "optimizer_trace": trace, "tables": tables,
            "figures": [{"file": image, "source": image_relative, "name": case_id}],
            "metrics": selected_metrics(read_json(state_path)),
        })
        total_generations += int(run.get("optimizer_generations", 0))
    payload = {
        "repository": repository, "branch": branch, "paper": paper, "groups": groups, "cases": entries,
        "observation_count": sum(item["rows"] for item in data_manifest["datasets"]),
        "optimizer_generations": total_generations,
        "source_index_sha256": digest(ROOT / "results/index.json"),
        "source_summary_sha256": digest(local_source("results/summary.csv")),
    }
    sections = []
    for group in groups:
        articles = "\n".join(case_article(entry) for entry in entries if entry["group"] == group["id"])
        sections.append(f'<section class="case-group" id="{group["id"]}" aria-labelledby="title-{group["id"]}">'
                        f'<h2 id="title-{group["id"]}">{html.escape(group["title"])}</h2>'
                        f'<div class="case-grid">{articles}</div></section>')
    authors = paper.get("authors", ["Ao Cheng", "Lei Zhang", "Guowei He"])
    author_line = ", ".join(authors[:-1]) + " and " + authors[-1] if len(authors) > 1 else authors[0]
    replacements = {
        "__PAPER_TITLE__": html.escape(paper["title"]),
        "__AUTHORS__": html.escape(author_line),
        "__PAPER_URL__": html.escape(paper["url"], quote=True),
        "__REPO_URL__": html.escape(repository, quote=True),
        "__BRANCH__": quote(branch, safe=""),
        "__CASE_GROUPS__": "\n".join(sections),
        "__WORKFLOW_CAPTION__": html.escape(paper.get("workflow_caption", "Schematic of the AI Buckingham workflow.")),
        "__PAGE_DATA__": json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"),
    }
    template = (TEMPLATES / "index.html").read_text(encoding="utf-8")
    for token, value in replacements.items():
        template = template.replace(token, value)
    if re.search(r"__[A-Z_]+__", template):
        raise ValueError("Unresolved Pages template token")
    remove_previous_generated_files(output)
    output.mkdir(parents=True, exist_ok=True)
    produced = []
    source_records = []

    def write(relative: str, content: str) -> None:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8", newline="\n")
        produced.append(relative)

    for source, relative in assets:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        produced.append(relative)
        source_records.append({"source": source.relative_to(ROOT).as_posix(), "output": relative,
                               "sha256": digest(source), "bytes": source.stat().st_size})
    for source in source_files:
        source_records.append({"source": source.relative_to(ROOT).as_posix(),
                               "sha256": digest(source), "bytes": source.stat().st_size})
    write("index.html", template)
    for entry in entries:
        write(entry["full_card"], public_card(entry, paper, repository))
    for name in ("style.css", "app.js"):
        write(f"assets/{name}", (TEMPLATES / name).read_text(encoding="utf-8"))
    write("data/cases.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    write(".nojekyll", "")
    write("build-info.json", json.dumps({
        "source_index_sha256": payload["source_index_sha256"],
        "source_summary_sha256": payload["source_summary_sha256"],
        "case_count": len(entries), "observation_count": payload["observation_count"],
        "sources": source_records, "method": "Case figures and saved structured results.",
    }, indent=2) + "\n")
    produced.append(".pages-build.json")
    (output / ".pages-build.json").write_text(json.dumps({"files": sorted(produced)}, indent=2) + "\n", encoding="utf-8")
    checked = check_local_links(output)
    repository_targets = sum(check_repository_targets(document, repository, branch) for document in output.rglob("*.html"))
    return {"output": output.relative_to(ROOT).as_posix(), "cases": len(entries), "files": len(produced),
            "bytes": sum((output / relative).stat().st_size for relative in produced),
            "local_links_checked": checked, "repository_targets_checked": repository_targets}


def check_repository_targets(document: Path, repository: str, branch: str) -> int:
    """Validate repository links against the checkout without network requests."""
    prefix = repository.rstrip("/") + "/"
    checked = 0
    for ref in re.findall(r'href="([^"]+)"', document.read_text(encoding="utf-8")):
        if not ref.startswith(prefix):
            continue
        suffix = ref[len(prefix):]
        for kind in ("blob", "tree"):
            marker = f"{kind}/{quote(branch, safe='')}/"
            if suffix.startswith(marker):
                relative = unquote(urlsplit(suffix[len(marker):]).path)
                target = (ROOT / relative).resolve()
                if not target.is_relative_to(ROOT.resolve()) or not target.exists():
                    raise ValueError(f"Broken repository Pages link: {ref}")
                if kind == "blob" and not target.is_file():
                    raise ValueError(f"Expected a repository file: {ref}")
                if kind == "tree" and not target.is_dir():
                    raise ValueError(f"Expected a repository directory: {ref}")
                checked += 1
    return checked


def check_local_links(output: Path) -> int:
    """Check local files and anchors without fetching deployment endpoints."""
    from html.parser import HTMLParser

    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.refs = []
            self.ids = set()

        def handle_starttag(self, tag, attrs):
            values = dict(attrs)
            if "id" in values:
                self.ids.add(values["id"])
            for attribute in ("src", "href"):
                if attribute in values:
                    self.refs.append(values[attribute])

    documents = {}
    for document in output.rglob("*.html"):
        parser = Links()
        parser.feed(document.read_text(encoding="utf-8"))
        documents[document.resolve()] = parser
    checked = 0
    for document, parser in documents.items():
        for ref in parser.refs:
            parsed = urlsplit(ref)
            if parsed.scheme or parsed.netloc:
                continue
            target = (document.parent / unquote(parsed.path)).resolve() if parsed.path else document
            if not target.is_relative_to(output.resolve()) or not target.is_file():
                raise ValueError(f"Broken local Pages reference in {document.name}: {ref}")
            if parsed.fragment and target in documents and unquote(parsed.fragment) not in documents[target].ids:
                raise ValueError(f"Broken HTML anchor in {document.name}: {ref}")
            checked += 1
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "site")
    parser.add_argument("--repository-url", default=REPOSITORY)
    parser.add_argument("--branch", default="main")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    print(json.dumps(build(output, args.repository_url.rstrip("/"), args.branch), indent=2))


if __name__ == "__main__":
    main()
