#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${COPAW_WORKSPACE_DIR:-/app/working/workspaces/default}"

rm -f "$WORKSPACE/BOOTSTRAP.md"

echo "setup complete: skills and local data deployed via environment/"
