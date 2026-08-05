#!/bin/bash
# render_pdf.sh -- prints report/report.html to report/PMVC_WNM_Evidence_Report.pdf
# using the pre-installed headless Chromium (no LaTeX toolchain in this environment).
set -e
cd "$(dirname "$0")"
CHROME=/opt/pw-browsers/chromium-1194/chrome-linux/chrome

"$CHROME" --headless --disable-gpu --no-sandbox \
  --print-to-pdf=report/PMVC_WNM_Evidence_Report.pdf \
  --print-to-pdf-no-header \
  --no-pdf-header-footer \
  --run-all-compositor-stages-before-draw \
  --virtual-time-budget=10000 \
  "file://$(pwd)/report/report.html" 2>&1 | grep -v "dbus\|ERROR:object_proxy\|ERROR:bus.cc" || true

ls -la report/PMVC_WNM_Evidence_Report.pdf
