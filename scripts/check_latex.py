"""Sanity check for LaTeX file structure."""
import re, sys, os
from collections import Counter

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

f = '<paper_package>/paper/latex_v2/agent_count_neurips_2026.tex'
text = open(f, encoding='utf-8').read()

# Check begin/end balance
begins = re.findall(r'\\begin\{(\w+)\}', text)
ends = re.findall(r'\\end\{(\w+)\}', text)
b, e = Counter(begins), Counter(ends)
mismatches = [k for k in set(b) | set(e) if b[k] != e[k]]
print(f'\\begin total: {sum(b.values())}, \\end total: {sum(e.values())}')
if mismatches:
    print('Mismatches:')
    for k in mismatches:
        print(f'  {k}: begin={b[k]}, end={e[k]}')
else:
    print('begin/end balanced OK')

# Citations
cites = re.findall(r'\\cite[ptn]?\{([^}]+)\}', text)
all_cite_keys = set()
for c in cites:
    for k in c.split(','):
        all_cite_keys.add(k.strip())
bibs = re.findall(r'\\bibitem\[[^\]]*\]\{([^}]+)\}', text)

print(f'\nCitations used ({len(all_cite_keys)}): {sorted(all_cite_keys)}')
print(f'Bibitems defined ({len(bibs)}): {sorted(bibs)}')
missing = all_cite_keys - set(bibs)
extra = set(bibs) - all_cite_keys
if missing:
    print(f'MISSING bibitems for: {missing}')
if extra:
    print(f'Unused bibitems: {extra}')
if not missing and not extra:
    print('Citations balanced OK')

# Figure references
figs = re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}', text)
print(f'\nFigures referenced ({len(figs)}):')
base = os.path.dirname(f)
for p in figs:
    cands = [os.path.join(base, p), os.path.join(base, p + '.pdf'),
             os.path.join(base, p + '.png')]
    found = next((c for c in cands if os.path.exists(c)), None)
    print(f'  {p}: {"OK -> " + found if found else "MISSING"}')

# label/ref consistency
labels = set(re.findall(r'\\label\{([^}]+)\}', text))
refs = set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', text))
print(f'\nLabels: {len(labels)}, refs: {len(refs)}')
unused_labels = labels - refs
broken_refs = refs - labels
if broken_refs:
    print(f'Broken refs (no matching label): {broken_refs}')
if unused_labels:
    print(f'Unused labels: {unused_labels}')

# basic stats
lines = text.count('\n')
words = len(re.findall(r'\b\w+\b', text))
print(f'\nLines: {lines}, words ~{words}')
print(f'TODO marks: {text.count("[TODO")}')
