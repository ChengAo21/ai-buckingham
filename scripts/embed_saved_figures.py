"""Embed saved PNG figures in notebooks and reports."""
from pathlib import Path
import base64
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import nbformat
    from ai_buckingham.artifacts import saved_figures_html, sha256_file
    count = 0
    for path in sorted((ROOT / "notebooks" / "cases").glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        metadata = nb.metadata.get("ai_buckingham", {})
        if metadata.get("execution") != "completed" or not metadata.get("run_directory"):
            continue
        run_dir = ROOT / metadata["run_directory"]
        figures = [p for p in sorted((run_dir / "figures").glob("*.png")) if not p.name.startswith("display_")]
        if not figures:
            continue
        nb.cells = [cell for cell in nb.cells if cell.metadata.get("ai_buckingham_role") != "saved_figure_gallery"]
        gallery = nbformat.v4.new_markdown_cell("## Figures\n\n" + "\n\n".join(f"**{p.name}**\n\n![{p.stem}](attachment:{p.name})" for p in figures))
        gallery.metadata["ai_buckingham_role"] = "saved_figure_gallery"
        gallery.attachments = {p.name: {"image/png": base64.b64encode(p.read_bytes()).decode("ascii")} for p in figures}
        gallery.metadata["source_sha256"] = {p.name: sha256_file(p) for p in figures}
        nb.cells.append(gallery)
        nbformat.write(nb, path)
        nbformat.write(nb, run_dir / "notebook.ipynb")
        report = run_dir / "report.html"
        if report.exists():
            text = report.read_text(encoding="utf-8")
            if 'id="saved-figures"' not in text:
                report.write_text(text.replace("</body>", saved_figures_html(run_dir) + "</body>"), encoding="utf-8")
        count += 1
    print(json.dumps({"notebooks_with_embedded_saved_figures": count}))


if __name__ == "__main__":
    main()
