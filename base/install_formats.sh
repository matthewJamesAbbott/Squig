#!/usr/bin/env bash
# install_formats.sh — one-shot dependencies for the extended book formats
# in the DDC pipeline (run once, idempotent).
#
#   pip (marker_env) : striprtf   mammoth   pychm
#   apt              : djvulibre-bin   calibre   libchm-dev
#
# What they enable (code lands in booktext.py / converters.py):
#   RTF            striprtf -> text path
#   DOCX           mammoth  -> text + images path
#   HTML / MHT / FB2   stdlib + lxml (already present)
#   CHM            pychm -> text path
#   MOBI / AZW / AZW3 / KFX / LIT (Kindle)  calibre ebook-convert -> EPUB path
#   DOC / ODT      LibreOffice headless (ALREADY installed) -> DOCX path
#   DJVU           djvups (PS) + ps2pdf (ghostscript, ALREADY installed)
#                  -> PDF -> marker OCR path (batched, pypdf merge)
#   CBZ / CBR      zip / 7z / unrar (ALREADY installed) -> PIL -> PDF path
#
# Usage:  bash install_formats.sh      (prompts for the sudo password)
set -euo pipefail

PY="${PIPELINE_PYTHON:-$HOME/miniconda3/envs/marker_env/bin/python}"
[[ -x "$PY" ]] || { echo "missing $PY"; exit 1; }

SUDO=""
if [[ $EUID -ne 0 ]]; then
  SUDO="sudo"
  command -v sudo >/dev/null || { echo "run as root or install sudo"; exit 1; }
fi

echo "== [1/3] apt: djvulibre-bin (djvups, djvused) + calibre (ebook-convert) + libchm-dev (pychm build dep) =="
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends djvulibre-bin calibre libchm-dev

echo "== [2/3] pip packages into marker_env: striprtf mammoth pychm =="
"$PY" -m pip install --quiet striprtf mammoth pychm

echo "== [3/3] verify =="
"$PY" - <<'EOF'
import striprtf, mammoth, chm, pypdf
from striprtf.striprtf import rtf_to_text
print("  python ok: striprtf", getattr(striprtf, "__version__", "?"),
      ", mammoth", getattr(mammoth, "__version__", "?"),
      ", chm(pychm)", chm.__version__, ", pypdf", pypdf.__version__)
# functional micro-check: strip a real RTF fragment
s = rtf_to_text(r"{\rtf1\ansi Hello {\b world} \par next para}")
assert "Hello" in s and "next para" in s, s
print("  striprtf functional:", repr(s[:40]))
# functional micro-check: PIL multi-page PDF (the comic bundler)
from PIL import Image
import io, pathlib
ims = [Image.new("RGB", (64, 64), c) for c in
       ((255, 0, 0), (0, 255, 0), (0, 0, 255))]
p = pathlib.Path("/tmp/opencode_fmt_test.pdf")
ims[0].save(p, "PDF", save_all=True, append_images=ims[1:])
assert p.stat().st_size > 500
p.unlink()
print("  PIL pdf bundler functional")
EOF

for c in ebook-convert djvups djvused ps2pdf soffice unrar 7z; do
  printf "  %-15s %s\n" "$c" "$(command -v "$c" || echo MISSING)"
done

# functional micro-check: calibre round-trip (html -> mobi)
T="$(mktemp -d /tmp/opencode/fmt_test.XXXXXX)"
trap 'rm -rf "$T"' EXIT
cat > "$T/sample.html" <<'HTML'
<html><body><h1>Sample</h1><p>Calibre round-trip check.</p></body></html>
HTML
( cd "$T" && ebook-convert sample.html sample.mobi >/dev/null \
    && file sample.mobi | grep -qi "Microsoft" \
    && echo "  ebook-convert functional: sample.mobi built" )

echo
echo "All installed. New formats ready for booktext.py / converters.py."
