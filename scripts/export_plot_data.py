"""Export figure source tables from completed runs without calling APIs."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from ai_buckingham.plot_data import export_plot_data
    count = 0
    for path in sorted((ROOT / "results" / "cases").glob("*/*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            continue
        state = json.loads((path.parent / "state.json").read_text(encoding="utf-8"))
        result = export_plot_data(state, path.parent)
        print(json.dumps({"case": manifest["case_id"], "tables": len(result["files"]), "warnings": result["warnings"]}))
        count += 1
    return count


if __name__ == "__main__":
    main()
