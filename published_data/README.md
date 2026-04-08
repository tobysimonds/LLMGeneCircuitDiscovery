# Published Data

This directory contains the minimal tracked data needed to reproduce the current PDAC blog/demo site without relying on ignored local `artifacts/` output.

Contents:

- `pdac_run/`
  - The published run bundle used to rebuild the blog site.
  - Includes summary data, graph JSONs, benchmark outputs, knockout rankings, and the two DEG graph PNGs.
- `blog_site/post_bundle.json`
  - The exact generated blog-site data bundle used by the current static site.

To rebuild the blog site from this tracked data:

```bash
uv run build_blog_site.py \
  --run-dir published_data/pdac_run \
  --output-dir artifacts/pdac-blog-site-20260321
```

To serve the rebuilt site locally:

```bash
uv run python -m http.server 8130 --directory artifacts/pdac-blog-site-20260321
```
