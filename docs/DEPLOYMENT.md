# Upload and deployment

Use the **public release folder**, or extract `ai-buckingham-public-v0.5.0.zip` and enter its inner `ai-buckingham/` directory. The repository root must contain `app.py`, `README.md`, `src/` and `docs/`.

The public release includes selected original figure images, the Streamlit preview, source, data and numerical results. Full manuscript and supplementary documents, editable slides, authoring notebooks, credentials and internal delivery files are excluded. Private authoring materials remain outside this upload directory.

## Upload with Git Bash

Create an empty GitHub repository named `ai-buckingham`. Leave GitHub's README, gitignore and license initialization disabled because these files are already included. Open **Git Bash Here** in the public release folder and run:

```bash
git init -b main
git add .
git status --short
git commit -m "Release AI Buckingham source, results and Streamlit explorer"
git remote add origin https://github.com/YOUR_USERNAME/ai-buckingham.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub owner name. If the extracted folder is already initialized, skip `git init`. If a remote already exists, inspect `git remote -v` before changing it. Sign in to GitHub when prompted. See [GitHub's command-line upload guide](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github).

If Git requests an author identity, configure this repository and repeat the commit:

```bash
git config user.name "YOUR_NAME"
git config user.email "YOUR_GITHUB_EMAIL"
```

The README already presents the Streamlit app, original framework artwork, selected results, configuration and reproducibility information. `PUBLIC_RELEASE.json` inventories the distributed files and their SHA-256 checksums. `LICENSE` records the current rights notice; the authors can replace it with their chosen code license.

## GitHub Pages

After upload, choose **Settings → Pages → Build and deployment → Source: GitHub Actions**. The included `.github/workflows/pages.yml` publishes `docs/`. Open **Actions → Publish research page → Run workflow** for the initial deployment. This follows [GitHub's custom-workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

The static site features the Streamlit screenshot and deployment path, selected original figures, all nine systems and interactive numerical plots. It does not host the Python app. All asset paths are relative, so repository Pages paths work. Preview locally with:

```bash
python -m http.server 8765 --directory docs
```

## Streamlit Community Cloud

Connect the uploaded repository, choose branch **main** and entry point **app.py**. Root `requirements.txt` installs the package with its app dependencies. The offline explorer starts without secrets. Follow the [official Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app).

For API runs, put runtime values into **Advanced settings → Secrets**, using `.streamlit/secrets.toml.example` as the schema. Actual secrets and `.env` are ignored. The app supports provider endpoints, chat and embedding models, and opt-in tracing. Provider calls start only when an autonomous run is requested.

After deployment, add the actual Streamlit app URL to the README's Streamlit section and replace the research page's deployment button with that URL. No hosted app URL is claimed in this local release.

Full-budget numerical discovery can exceed free-host resources. Run expensive searches locally and inspect their exported JSON in the app.

## Rebuild the public assets

```bash
python scripts/build_research_assets.py
python scripts/recompute_regimes.py cess pipe
python scripts/build_regime_learning.py
python scripts/build_research_assets.py
python -m pytest -q
python scripts/validate_repository.py
```

The asset builder reuses existing landscape files and does not copy manuscript documents. If an objective or dataset changes, regenerate its corresponding landscape and trajectory records together. Original CSVs and reference values are not overwritten.

`build_regime_learning.py` rebuilds all five regime-specific surfaces and paths from the current Cess/pipe partitions. Each grid and search stores row IDs and a matching subset fingerprint. `build_discovery_map.py` rebuilds the workflow from packaged original artwork crops, without requiring the private PowerPoint file.

To create a fresh public package and upload folder:

```bash
python scripts/package_release.py \
  --output ../ai-buckingham-public.zip \
  --directory ../ai-buckingham-public
```

The builder uses an explicit allowlist, scans for credential patterns and refuses to overwrite an existing archive or nonempty upload directory. Validate a fresh exported tree with:

```bash
python scripts/validate_repository.py --public
```
