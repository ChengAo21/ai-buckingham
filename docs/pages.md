# Research page on GitHub Pages

The public research page is served at [chengao21.github.io/ai-buckingham](https://chengao21.github.io/ai-buckingham/). It presents the project and selected recorded results. The separate [hosted Streamlit application](https://ai-buckingham.streamlit.app/) provides the interactive workflow and live-analysis interface. Its local setup and operating details are described in [app.md](app.md).

## Build and publication

`.github/workflows/pages.yml` builds and publishes the site when `main` changes or when a maintainer runs the workflow manually. The build uses Python 3.12 and the standard library:

```bash
python scripts/build_pages.py
```

The builder reads the committed result index and summaries, and stages the static page and selected assets under `site/`. It does not run the scientific engine or call model APIs. The Pages artifact contains only `site/`; the workflow does not send the entire research repository through Jekyll.

Build steps have repository-read access. Only the deployment job receives `pages: write` and `id-token: write`. Deployment requires the successful build job, uses the `github-pages` environment, and publishes only from `main`. Existing deployment history remains available in GitHub Actions.

## Repository setting

In the repository's **Settings → Pages → Build and deployment**, set **Source** to **GitHub Actions**. The corresponding REST API value is `build_type: workflow`. A branch-based publishing source would start a different build pipeline and should not be used with this workflow.

After publication, check the latest **Publish research page** run at the current commit. Both **Build static research page** and **Deploy research page** should succeed, and the deployment environment should link to the public page. A historical failed Pages run remains historical evidence; use the latest deployment and commit when checking the current status.

## Local preview

```bash
python scripts/build_pages.py
python -m http.server 8000 --bind 127.0.0.1 --directory site
```

Open [localhost:8000](http://127.0.0.1:8000/) to inspect the page. Regenerate the static output after changing the showcase content or committed results, then publish through the same workflow.

The workflow follows GitHub's official [custom Pages workflow guidance](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site#publishing-with-a-custom-github-actions-workflow) and uses the official [Pages artifact action](https://github.com/actions/upload-pages-artifact) and [deployment action](https://github.com/actions/deploy-pages).
