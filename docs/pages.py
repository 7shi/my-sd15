"""docs/ の各 .md ファイルにあるページナビゲーション行を更新する。

"ページ：" で始まる行を
  ページ：[1](01_quickstart.md) | [2](02_overview.md) | ...
の形式に書き換える。カレントディレクトリは問わず、このスクリプトのある
ディレクトリ（docs/）を基準にして動作する。
"""

import re
from pathlib import Path

script_dir = Path(__file__).parent
md_files = sorted(script_dir.glob("[0-9][0-9]_*.md"))

# ページリンク文字列を組み立てる（自分自身はリンクなしの数字のみ）
def make_nav(current):
    parts = []
    for path in md_files:
        num = path.name[:2]
        parts.append(f"**{num}**" if path == current else f"[{num}]({path.name})")
    return "ページ：" + " | ".join(parts)

# 各ファイルの "ページ：" 行を置換する
for path in md_files:
    nav_line = make_nav(path)
    text = path.read_text(encoding="utf-8")
    new_text = re.sub(r"^ページ：.*$", nav_line, text, flags=re.MULTILINE)
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
        print(f"updated: {path.name}")
    else:
        print(f"no change: {path.name}")
