# Credentials and sensitive output

Keep provider keys in environment variables, an untracked `.env`, or Streamlit
Secrets. Tracing is opt-in. Rotate any credentials that have appeared in messages
or previous source files before release. Never include keys in issue reports.

The supplied chat credentials are not stored in this repository. Exported runs
remove credential-like dictionary keys; review free-text provider output before
sharing it. Configure endpoint URLs only for providers you intend to use.

For a sensitive report, contact the corresponding authors using the manuscript's
contact information, rather than opening a public issue containing the secret.
