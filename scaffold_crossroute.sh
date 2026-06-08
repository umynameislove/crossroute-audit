#!/usr/bin/env bash
# Recreate the CrossRoute-Audit project skeleton (idempotent).
# Usage: bash scaffold_crossroute.sh [TARGET_DIR]   (default: current directory)
set -euo pipefail
ROOT="${1:-.}"
cd "$ROOT"

# Package layout
mkdir -p crossroute_audit/{model_adapters,instrumentation,interventions,controls,attribution,metrics,io,dashboard}
mkdir -p schemas data/manifest data/images configs synthetic tests runs

# Package markers
for d in crossroute_audit crossroute_audit/*/ ; do
  [ -f "${d%/}/__init__.py" ] || echo '"""CrossRoute-Audit package."""' > "${d%/}/__init__.py"
done

# Keep empty directories under version control
touch runs/.gitkeep data/images/.gitkeep

echo "Skeleton created at: $ROOT"
