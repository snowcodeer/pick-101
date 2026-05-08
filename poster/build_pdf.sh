#!/usr/bin/env bash
# Build A0 PDF (841mm x 1189mm) from poster.html using headless Chrome.
# Requires Google Chrome or Chromium.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
HTML="$HERE/poster.html"
PDF="$HERE/poster_A0.pdf"

# Find Chrome / Chromium
if [[ -x "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ]]; then
    CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
elif command -v google-chrome >/dev/null 2>&1; then
    CHROME="$(command -v google-chrome)"
elif command -v chromium >/dev/null 2>&1; then
    CHROME="$(command -v chromium)"
else
    echo "ERROR: Google Chrome or Chromium not found." >&2
    echo "Install Chrome, or open poster.html in a browser and use" >&2
    echo "  File > Print > Save as PDF, paper size = A0 (841mm x 1189mm)" >&2
    exit 1
fi

echo "Using: $CHROME"
echo "Rendering: $HTML"
echo "Output:    $PDF"

"$CHROME" \
    --headless \
    --disable-gpu \
    --no-pdf-header-footer \
    --no-margins \
    --print-to-pdf="$PDF" \
    --print-to-pdf-no-header \
    --virtual-time-budget=10000 \
    "file://$HTML"

echo
echo "Done. PDF written to: $PDF"
echo "Verify dimensions with: pdfinfo \"$PDF\" | grep 'Page size'"
echo "Expected: 2383.94 x 3370.39 pts (A0)"
