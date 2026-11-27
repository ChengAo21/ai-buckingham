# Static results explorer

The [results explorer](https://chengao21.github.io/ai-buckingham/) presents the ten cases in two columns, with full result cards and reproduction instructions. The [Streamlit application](https://ai-buckingham.streamlit.app/) provides interactive results and analysis controls.

## Build and preview

From the repository root, use Python 3.12:

```bash
python scripts/build_pages.py
python -m http.server 8000 --bind 127.0.0.1 --directory site
```

Open [localhost:8000](http://127.0.0.1:8000/). The builder reads the included case index, summaries and figures, then writes the static files to `site/`.

The page uses the final case figures listed in `docs/paper_cases.json`.

## GitHub Pages deployment

In **Settings → Pages → Build and deployment**, set **Source** to **GitHub Actions**.

`.github/workflows/pages.yml` builds and publishes `site/` when `main` changes or when a maintainer starts the workflow manually. Check the current **Publish research page** run for the build and deployment status.

See GitHub's [Pages workflow guidance](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site#publishing-with-a-custom-github-actions-workflow) for deployment configuration.
