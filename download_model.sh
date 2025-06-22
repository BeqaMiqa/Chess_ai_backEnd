#!/usr/bin/env bash
set -e

mkdir -p model
curl -L \
  "https://github.com/BeqaMiqa/Chess_ai_backEnd/releases/download/v1.0/board_level_model.h5" \
  --output model/board_level_model.h5

