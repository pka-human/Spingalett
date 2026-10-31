#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
"""Writes Images/SpeedLight.svg and SpeedDark.svg, the README's chart, from the table in CHANGELOG.md's 1.6.0."""
from pathlib import Path

# Tokens a second on an RTX 4050 Laptop GPU: (model, Spingalett on CUDA, llama.cpp on Vulkan), prompt and generation.
PROMPT = [("Qwen3-0.6B Q8_0", 16447, 12120), ("Qwen3-1.7B Q8_0", 8117, 6543), ("Qwen3-4B Q4_K_M", 3601, 2974),
          ("Llama-3.2-1B Q4_K_M", 12682, 10473), ("Qwen2.5-0.5B Q4_K_M", 22215, 16265)]
TOKENS = [("Qwen3-0.6B Q8_0", 222, 207), ("Qwen3-1.7B Q8_0", 90.7, 90.2), ("Qwen3-4B Q4_K_M", 64.9, 65.2),
          ("Llama-3.2-1B Q4_K_M", 196, 198), ("Qwen2.5-0.5B Q4_K_M", 326, 290)]
THEMES = {
    "Light": {"ours": "#2a78d6", "rival": "#eb6834", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df"},
    "Dark": {"ours": "#3987e5", "rival": "#d95926", "text": "#f0efec", "muted": "#c3c2b7", "grid": "#30363d"},
}
WIDTH, LABELS, PANEL, GAP, RATIO = 880, 150, 300, 40, 45
TOP, ROW, BAR = 74, 40, 11
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"


def bar(x, y, length, color):
    """A bar from its baseline, its data end rounded by 4 pixels."""
    r = min(4, length)
    return (f'<path d="M{x} {y}h{length - r:.1f}a{r} {r} 0 0 1 {r} {r}v{BAR - 2 * r}a{r} {r} 0 0 1 -{r} {r}'
            f'h-{length - r:.1f}z" fill="{color}"/>')


def panel(x, title, rows, top, ticks, t):
    bottom = TOP + ROW * len(rows)
    out = [f'<text x="{x}" y="{TOP - 16}" font-size="14" font-weight="600" fill="{t["text"]}">{title}</text>']
    for v in ticks:
        gx = x + PANEL * v / top
        out.append(f'<line x1="{gx:.1f}" y1="{TOP - 6}" x2="{gx:.1f}" y2="{bottom}" stroke="{t["grid"]}"/>')
        out.append(f'<text x="{gx:.1f}" y="{bottom + 16}" font-size="11" text-anchor="middle" '
                   f'fill="{t["muted"]}">{v:,}</text>')
    for i, (_, ours, rival) in enumerate(rows):
        y = TOP + ROW * i + 4
        out.append(bar(x, y, PANEL * ours / top, t["ours"]))
        out.append(bar(x, y + BAR + 2, PANEL * rival / top, t["rival"]))
        end = x + PANEL * max(ours, rival) / top + 7
        out.append(f'<text x="{end:.1f}" y="{y + BAR + 5}" font-size="12" font-weight="600" '
                   f'fill="{t["text"]}">{ours / rival:.2f}×</text>')
    return out


def chart(t):
    height = TOP + ROW * len(PROMPT) + 28
    second = LABELS + PANEL + RATIO + GAP - 25
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" '
           f'font-family="{FONT}" role="img" aria-labelledby="t d">',
           '<title id="t">Spingalett on CUDA against llama.cpp on Vulkan, tokens a second</title>',
           '<desc id="d">Five GGUF models on an RTX 4050 Laptop GPU. Spingalett processes prompts 1.21 to 1.37 times as '
           'fast; generation is level, 0.99 to 1.12 times. The README\'s table holds the numbers.</desc>',
           f'<rect x="{LABELS}" y="6" width="12" height="12" rx="3" fill="{t["ours"]}"/>',
           f'<text x="{LABELS + 18}" y="16" font-size="13" fill="{t["text"]}">Spingalett on CUDA</text>',
           f'<rect x="{LABELS + 160}" y="6" width="12" height="12" rx="3" fill="{t["rival"]}"/>',
           f'<text x="{LABELS + 178}" y="16" font-size="13" fill="{t["text"]}">llama.cpp on Vulkan</text>']
    for i, (name, _, _) in enumerate(PROMPT):
        out.append(f'<text x="{LABELS - 12}" y="{TOP + ROW * i + 20}" font-size="12" text-anchor="end" '
                   f'fill="{t["muted"]}">{name}</text>')
    out += panel(LABELS, "Prompt of 512 tokens, tokens/s", PROMPT, 25000, [0, 5000, 10000, 15000, 20000, 25000], t)
    out += panel(second, "Generation of 128 tokens, tokens/s", TOKENS, 400, [0, 100, 200, 300, 400], t)
    return "\n".join(out + ["</svg>", ""])


for name, theme in THEMES.items():
    (Path(__file__).parent / f"Speed{name}.svg").write_text(chart(theme), encoding="utf-8")
