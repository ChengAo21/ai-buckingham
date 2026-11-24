"""Build the static GitHub Pages evidence explorer from reviewed saved results.

This entry point uses only the Python standard library. It never imports the
agent, initializes models, recalculates scientific results, or accesses a network.
"""
from __future__ import annotations

import argparse
import csv
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
CASE_ORDER = (
    "mhd", "pressure_gradient", "rayleigh_benard", "laser_keyhole", "blasius",
    "sandgrain", "velocity_scaling", "goldenfeld", "cess_channel", "rough_pipe",
)
SHORT_NAMES = {
    "mhd": "MHD duct flow", "pressure_gradient": "Pressure-gradient friction",
    "rayleigh_benard": "Rayleigh–Bénard convection", "laser_keyhole": "Laser keyhole",
    "blasius": "Blasius boundary layer", "sandgrain": "Sand-grain roughness",
    "velocity_scaling": "Compressible velocity", "goldenfeld": "Goldenfeld similarity",
    "cess_channel": "Cess channel flow", "rough_pipe": "Rough circular pipe",
}
DESCRIPTIONS = {
    "mhd": "A pressure–viscous coordinate emerges while magnetic variables are removed during simplification.",
    "pressure_gradient": "One coordinate combines viscosity, integral-profile energy and boundary-layer thicknesses.",
    "rayleigh_benard": "An invertible Rayleigh-number coordinate organizes the convection measurements.",
    "laser_keyhole": "The learned coordinate recovers a power-equivalent Keyhole-number representation.",
    "blasius": "The selected coordinate recovers the classical similarity variable for a laminar boundary layer.",
    "sandgrain": "Two learned groups retain complementary roughness information, evaluated on the recorded source-stratified 83/21 split.",
    "velocity_scaling": "Type II similarity rescales both the wall-normal coordinate and the normalized mean velocity.",
    "goldenfeld": "Joint coordinate learning and output scaling recover an incomplete-similarity rough-pipe representation.",
    "cess_channel": "Three local relations distinguish outer, intermediate and near-wall dependence.",
    "rough_pipe": "Gradient-based identification separates Reynolds-number and roughness-dominated dependencies.",
}
ASSETS = (
    "hero.svg", "workflow.svg", "mhd_mi_landscape.png", "cess_regimes.png",
    "rough_pipe_regimes.png", "sandgrain_predictions.png", "velocity_transformations.png",
    "manifest.json",
)


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


def formula(value: str) -> str:
    escaped = html.escape(value)
    return re.sub(r"\^\{([^{}]+)\}", r"<sup>\1</sup>", escaped).replace(" * ", " · ")


def scope_name(case_id: str, scope: str) -> str:
    names = {
        "cess_channel": {"regime_0": "Outer", "regime_1": "Intermediate", "regime_2": "Near wall"},
        "rough_pipe": {"regime_0": "Roughness dominated", "regime_1": "Reynolds dominated"},
    }
    return names.get(case_id, {}).get(scope, "Global")


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
    index = read_json(local_source("results/index.json"))
    manifest = read_json(local_source("data/manifest.json"))
    benchmark = read_json(local_source("reproduction/rag_ablation/package_manifest.json"))
    with local_source("results/summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        summary = list(csv.DictReader(stream))
    by_id = {case["case_id"]: case for case in index["cases"]}
    if set(by_id) != set(CASE_ORDER):
        raise ValueError("The Pages explorer requires the ten published cases")
    sources = []
    for name in ASSETS:
        sources.append(local_source(f"docs/assets/{name}"))
    remove_previous_generated_files(output)
    output.mkdir(parents=True, exist_ok=True)
    produced: list[str] = []
    source_records: list[dict] = []

    def write(relative: str, content: str) -> None:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8", newline="\n")
        produced.append(relative)

    def copy(source: Path, relative: str) -> None:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        produced.append(relative)
        source_records.append({
            "source": source.relative_to(ROOT).as_posix(), "output": relative,
            "sha256": digest(source), "bytes": source.stat().st_size,
        })

    for source in sources:
        copy(source, f"assets/{source.name}")
    entries = []
    articles = []
    selectors = []
    total_generations = 0
    for case_id in CASE_ORDER:
        case = by_id[case_id]
        selected = [run for run in case["runs"] if run["run_id"] == case["selected_run"]]
        if len(selected) != 1 or selected[0]["status"] != "completed":
            raise ValueError(f"Case {case_id} does not select one completed run")
        run = selected[0]
        metrics = [row for row in summary if row["case_id"] == case_id]
        if not metrics or any(row["run_id"] != run["run_id"] for row in metrics):
            raise ValueError(f"Summary/index mismatch for {case_id}")
        config = read_json(local_source(f"configs/cases/{case_id}.json"))
        sample_count = config["data_metadata"]["rows"]
        card = f"cards/{case_id}.html"
        copy(local_source(run["report"]), card)
        figures = []
        for relative in run["figures"]:
            if relative.lower().endswith(".png"):
                source = local_source(relative)
                target = f"assets/cases/{case_id}/{source.name}"
                copy(source, target)
                figures.append({"file": target, "source": relative, "name": source.stem})
        if not figures:
            raise ValueError(f"Case {case_id} has no recorded PNG figure")
        primary = next((figure for figure in figures if figure["name"].endswith("regime_partition")), figures[0])
        total_generations += int(run.get("optimizer_generations", 0))
        multiple = len(metrics) > 1
        second_kind = any(row["similarity_type"] == "second_kind" for row in metrics)
        tag = "Multiple regimes" if multiple else "Type II similarity" if second_kind else "Type I similarity"
        notebook = github(case["notebook"], repository=repository, branch=branch)
        evidence = github(run["directory"], repository=repository, branch=branch, directory=True)
        trace_relative = f"{run['directory']}/evidence/optimization_trace.jsonl"
        trace = github(trace_relative, repository=repository, branch=branch) if (ROOT / trace_relative).is_file() else None
        entry = {
            "case_id": case_id, "title": case["title"], "short_name": SHORT_NAMES[case_id],
            "description": DESCRIPTIONS[case_id], "samples": sample_count, "tag": tag,
            "run_id": run["run_id"], "full_card": card, "notebook": notebook,
            "evidence": evidence, "optimizer_trace": trace, "figures": figures,
            "metrics": [{
                "scope": scope_name(case_id, row["scope_id"]), "samples": int(row["local_samples"]),
                "groups": int(row["group_count"]), "mutual_information": float(row["mutual_information"]),
                "epsilon": float(row["information_score"]), "conditional_test_r2": float(row["conditional_test_r2"]),
                "input_formulas": json.loads(row["input_formulas"]), "output_formula": row["output_formula"],
            } for row in metrics],
        }
        entries.append(entry)
        selectors.append(f'<a class="case-option" href="#case-{case_id}" role="tab" id="tab-{case_id}" aria-controls="case-{case_id}" aria-selected="false">{html.escape(SHORT_NAMES[case_id])}</a>')
        metric_rows = "".join(
            f'<tr><th scope="row">{html.escape(metric["scope"])}</th><td>{metric["samples"]:,}</td><td>{metric["groups"]}</td><td>{metric["mutual_information"]:.3f}</td><td>{metric["epsilon"]:.3f}</td><td>{metric["conditional_test_r2"]:.3f}</td></tr>'
            for metric in entry["metrics"]
        )
        formula_rows = "".join(
            f'<div class="formula-row"><strong>{html.escape(metric["scope"])}</strong>'
            + "".join(f'<div class="formula">{formula(value)}</div>' for value in metric["input_formulas"])
            + f'<div class="output-formula">Response: {formula(metric["output_formula"])}</div></div>'
            for metric in entry["metrics"]
        )
        extras = [figure for figure in figures if figure is not primary]
        extra_html = ""
        if extras:
            extra_html = '<details class="more-figures"><summary>More recorded figures</summary>' + "".join(
                f'<figure><a href="{figure["file"]}" target="_blank" rel="noopener"><img src="{figure["file"]}" loading="lazy" decoding="async" alt="Additional recorded evidence for {html.escape(SHORT_NAMES[case_id])}"></a></figure>'
                for figure in extras
            ) + '</details>'
        trace_html = f'<a href="{trace}">Recorded optimizer trace</a>' if trace else ''
        articles.append(f'''<article class="case-panel" id="case-{case_id}" role="tabpanel" aria-labelledby="tab-{case_id}" data-case="{case_id}">
  <div class="case-heading"><div><p class="eyebrow">{html.escape(tag)} <span>·</span> {sample_count:,} observations</p><h3>{html.escape(case["title"])}</h3></div><a class="text-link" href="{evidence}">Run evidence</a></div>
  <p class="case-description">{html.escape(DESCRIPTIONS[case_id])}</p>
  <figure class="case-figure"><a href="{primary["file"]}" target="_blank" rel="noopener"><img src="{primary["file"]}" loading="lazy" decoding="async" alt="Actual recorded result for {html.escape(SHORT_NAMES[case_id])}"></a><figcaption>Original recorded plot. Open the image for full resolution.</figcaption></figure>
  <div class="table-scroll"><table class="metric-table"><caption>Selected representation · recorded numerical results</caption><thead><tr><th scope="col">Scope</th><th scope="col">Samples</th><th scope="col">Groups</th><th scope="col">MI</th><th scope="col">ε</th><th scope="col">Test R²</th></tr></thead><tbody>{metric_rows}</tbody></table></div>
  <details class="formulas"><summary>Selected dimensionless relations</summary>{formula_rows}</details>
  <div class="case-actions"><a class="button" href="{card}">Read full result card</a><a class="button button-outline" href="{notebook}">Open executed notebook</a></div>
  <div class="case-text-links">{trace_html}<a href="{github(f'configs/cases/{case_id}.json', repository=repository, branch=branch)}">Exact case settings</a></div>
  {extra_html}
</article>''')
    payload = {
        "repository": repository, "branch": branch, "cases": entries,
        "observation_count": sum(item["rows"] for item in manifest["datasets"]),
        "optimizer_generations": total_generations,
        "source_index_sha256": digest(ROOT / "results/index.json"),
        "source_summary_sha256": digest(ROOT / "results/summary.csv"),
    }
    template = (TEMPLATES / "index.html").read_text(encoding="utf-8")
    replacements = {
        "__REPO_URL__": html.escape(repository),
        "__BRANCH__": quote(branch, safe=""),
        "__CASE_COUNT__": str(len(entries)),
        "__TRIAL_COUNT__": f'{benchmark["trial_count"]:,}',
        "__CHECKER_ROUNDS__": f'{benchmark["checker_rounds"]:,}',
        "__OBSERVATION_COUNT__": f'{payload["observation_count"]:,}',
        "__GENERATION_COUNT__": f'{total_generations:,}',
        "__CASE_SELECTORS__": "\n".join(selectors),
        "__CASE_ARTICLES__": "\n".join(articles),
        "__PAGE_DATA__": json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    if re.search(r"__[A-Z_]+__", template):
        raise ValueError("Unresolved Pages template token")
    write("index.html", template)
    for name in ("style.css", "app.js"):
        write(f"assets/{name}", (TEMPLATES / name).read_text(encoding="utf-8"))
    write("data/cases.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    write(".nojekyll", "")
    write("build-info.json", json.dumps({
        "source_index_sha256": payload["source_index_sha256"],
        "source_summary_sha256": payload["source_summary_sha256"],
        "case_count": len(entries), "observation_count": payload["observation_count"],
        "sources": source_records, "method": "Static copy of reviewed saved evidence; no scientific execution or API calls.",
    }, indent=2) + "\n")
    produced.extend([".pages-build.json"])
    (output / ".pages-build.json").write_text(json.dumps({"files": sorted(produced)}, indent=2) + "\n", encoding="utf-8")
    checked = check_local_links(output)
    repository_targets = check_repository_targets(output / "index.html", repository, branch)
    size = sum((output / relative).stat().st_size for relative in produced)
    return {"output": output.relative_to(ROOT).as_posix(), "cases": len(entries), "files": len(produced), "bytes": size, "local_links_checked": checked, "repository_targets_checked": repository_targets}


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
    """Check references without fetching GitHub or the deployment endpoint."""
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

    checked = 0
    for document in output.rglob("*.html"):
        parser = Links()
        parser.feed(document.read_text(encoding="utf-8"))
        for ref in parser.refs:
            parsed = urlsplit(ref)
            if parsed.scheme or parsed.netloc:
                continue
            if not parsed.path:
                if parsed.fragment and parsed.fragment not in parser.ids:
                    raise ValueError(f"Broken HTML anchor in {document.name}: {ref}")
            else:
                target = (document.parent / parsed.path).resolve()
                if not target.is_relative_to(output.resolve()) or not target.is_file():
                    raise ValueError(f"Broken local Pages reference in {document.name}: {ref}")
            checked += 1
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "site")
    parser.add_argument("--repository-url", default=REPOSITORY)
    parser.add_argument("--branch", default="main")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build(output, args.repository_url.rstrip("/"), args.branch)
    print(json.dumps(report, indent=2))
    if report["bytes"] > 20 * 1024 * 1024:
        print("Note: Pages output exceeds the preferred 20 MiB presentation budget.")


if __name__ == "__main__":
    main()
