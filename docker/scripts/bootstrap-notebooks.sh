#!/usr/bin/env bash

# This hook is sourced by the Jupyter Docker Stacks startup script, so keep
# strict shell options isolated to avoid mutating the parent shell state.
(
  set -euo pipefail

  NB_USER_NAME="${NB_USER:-jovyan}"
  SRC_DIR="/opt/llm-spine/notebooks"
  PROJECT_DIR="/home/${NB_USER_NAME}/llm_train"
  FALLBACK_DIR="/home/${NB_USER_NAME}/notebooks"

  if [ -d "$SRC_DIR" ]; then
    if [ -d "$PROJECT_DIR" ]; then
      DEST_DIR="${PROJECT_DIR}/notebooks"
    else
      DEST_DIR="$FALLBACK_DIR"
    fi

    mkdir -p "$DEST_DIR"

    shopt -s nullglob
    for nb in "$SRC_DIR"/*.ipynb; do
      nb_name="$(basename "$nb")"
      if [ ! -e "$DEST_DIR/$nb_name" ]; then
        cp "$nb" "$DEST_DIR/$nb_name"
      fi
    done
  fi
)

