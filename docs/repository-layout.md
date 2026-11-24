# Repository layout and naming

The reusable implementation lives in `src/ai_buckingham/`. Notebooks and the web app call that implementation. Case IDs are shared by configuration files, dataset names, notebook names and run directories. Module names use lower-case snake_case; callable and state-field compatibility aliases are documented in `refactoring.md`.

`data/datasets/` is the authoritative input set. Its bytes and CSV spellings are preserved. `data/generation/` contains the supplied generators and provenance. `knowledge/` holds the two separately cited knowledge bases and read-only embedding indexes. The former development `Dataset` directory is not distributed.

`configs/cases/` preserves each exact prompt and manual override. Reference dimensions are metadata for validation, not forced answers. Shared numerical defaults remain in `config.py` and the existing numerical routines, avoiding a second editable copy that can drift from the implementation.

`results/cases/<case_id>/<run_id>/` holds included fixed runs. `generated/` is ignored and reserved for future user runs, caches and reproduction outputs. A result manifest prevents overwriting a completed or partial run. To change settings, create a new run.

Within a run, `plots/` preserves the original routines' output paths referenced by the recorded state; `figures/` contains exact copies for the shared viewer and report interface. The case notebook is also copied into its run directory so the public latest notebook and every retained attempt remain traceable.

`reproduction/` preserves the original scientific evidence packages and exposes safe entry points. Internal historical IDs and file bytes are retained even when their spelling differs from current naming conventions. In particular, the velocity comparison's historical CSV uses a different column order and must not be replaced with the central CSV.

The repository intentionally excludes manuscripts, manuscript source archives, reference-paper collections, local credentials, local Python environments and Git history. PDF files inside numerical result directories are generated figures, not reference publications.
