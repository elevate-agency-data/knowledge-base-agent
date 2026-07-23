# Brand assets

Drop each brand's official logo here and point its profile at it via
`logo_path` in `shared/brand.py`.

Expected files (supply the real, licensed assets — none are shipped in git):

| Profile  | Expected file            | `logo_path` in brand.py          |
|----------|--------------------------|----------------------------------|
| hermes   | `hermes_logo.png`        | `ui/assets/hermes_logo.png`      |
| indica   | `indica_logo.png`        | (set if/when available)          |
| lacoste  | `lacoste_logo.png`       | (set if/when available)          |
| activate | `activate_logo.png`      | (set if/when available)          |

- Formats: PNG (transparent background preferred) or SVG.
- Recommended height ~56px in the header; the source can be larger.
- If the file is absent, the UI falls back to a styled text wordmark — no
  broken image, no placeholder trademark.
