"""Package paper/latex_v2/ into a zip for Overleaf upload.

Overleaf accepts a zip of the project root (containing main .tex + .sty + figures/).
"""
import zipfile, sys, io
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'paper' / 'latex_v2'
OUT = ROOT / 'paper' / 'agent_count_neurips_2026_overleaf.zip'

# Files to include (avoid junk like .DS_Store, *.aux, etc.)
INCLUDE_EXT = {'.tex', '.sty', '.bib', '.pdf', '.png', '.cls', '.bst'}
EXCLUDE_NAMES = {'.DS_Store', 'Thumbs.db'}

# Exclude the unused legacy teaser variants — Overleaf doesn't need them
LEGACY_TEASERS = {
    'fig0_teaser.pdf', 'fig0_teaser.png',
    'fig0_teaser_v2.pdf', 'fig0_teaser_v2.png',
    'fig0_teaser_v3.pdf', 'fig0_teaser_v3.png',
    'fig0_teaser_v4.pdf', 'fig0_teaser_v4.png',
}

with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    n = 0
    total_bytes = 0
    for p in sorted(SRC.rglob('*')):
        if not p.is_file(): continue
        if p.name in EXCLUDE_NAMES: continue
        if p.name in LEGACY_TEASERS: continue
        if p.suffix.lower() not in INCLUDE_EXT: continue
        rel = p.relative_to(SRC)
        zf.write(p, str(rel))
        n += 1
        total_bytes += p.stat().st_size
        print(f'  + {rel}  ({p.stat().st_size//1024} KB)')

print(f'\nWrote {OUT}')
print(f'  {n} files, {total_bytes/1024/1024:.1f} MB uncompressed')
print(f'  zip size: {OUT.stat().st_size/1024/1024:.1f} MB')

print('\n=== Overleaf upload steps ===')
print('1. https://www.overleaf.com → New Project → Upload Project')
print(f'2. Drag {OUT.name} into the upload area')
print('3. Main file: agent_count_neurips_2026.tex (Overleaf usually auto-detects)')
print('4. Compiler: pdfLaTeX (default). Click "Recompile".')
print('5. If compile fails for missing packages, leave them — neurips_2026.sty bundles what it needs.')
