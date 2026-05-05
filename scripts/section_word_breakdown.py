"""Section-by-section word count for compression targeting."""
import sys, io, re
from pathlib import Path
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

p = Path(r'<paper_package>\paper\latex_v2\agent_count_neurips_2026.tex')
text = p.read_text(encoding='utf-8')
doc_start = text.find(r'\begin{document}')
bib_start = text.find(r'\bibliographystyle')

pat = re.compile(r'\\section\*?\{([^}]+)\}')
all_matches = list(pat.finditer(text))
body_secs = [m for m in all_matches if doc_start <= m.start() < bib_start]
print(f'Body sections: {len(body_secs)}')

def wc(s):
    s = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?', ' ', s)
    s = re.sub(r'\$[^$]*\$', 'X', s)
    s = re.sub(r'%.*', '', s)
    return len(s.split())

starts = [(m.start(), m.group(1)) for m in body_secs]
starts.append((bib_start, '<END>'))
pre = text[doc_start:starts[0][0]] if starts else text[doc_start:bib_start]
print(f'\n{"Section":50s}  {"Words":>6s}')
print('-' * 62)
total = wc(pre)
print(f'  {"<abstract+intro+contribs>":48s}  {total:>6d}')
for i in range(len(starts)-1):
    nm = starts[i][1]
    body_i = text[starts[i][0]:starts[i+1][0]]
    w = wc(body_i)
    total += w
    print(f'  {nm[:48]:48s}  {w:>6d}')
print('-' * 62)
print(f'  {"BODY TOTAL":48s}  {total:>6d}')
