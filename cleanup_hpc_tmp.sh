#!/bin/bash
VENV=/tmp/luhi161d-chest-press-cxr-venv
if [[ -d "$VENV" ]]; then
    rm -rf "$VENV"
    echo "Removed $VENV"
else
    echo "$VENV does not exist"
fi
