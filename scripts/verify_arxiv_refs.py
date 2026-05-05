"""Verify all arXiv IDs in the paper bibliography against the public arXiv API.

For each \\bibitem{key} entry that contains an arXiv ID:
  - Extract the ID and the bibliography's claim (title, authors, year)
  - Query http://export.arxiv.org/api/query?id_list=<id>
  - Parse Atom XML response (no extra deps)
  - Compare title, first author, year
  - Flag MISMATCH if any field disagrees materially

Output:
  analysis/arxiv_verification.csv  — per-citation result table
  stdout — human-readable summary
"""
from __future__ import annotations
import re, sys, io, csv, urllib.request, urllib.parse, time, ssl
from pathlib import Path
from xml.etree import ElementTree as ET

# Bypass SSL verification for arXiv public API (Windows certifi store issue).
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PAPER = Path(r'<paper_package>\paper\latex_v2\agent_count_neurips_2026.tex')
OUT_CSV = Path(r'<paper_package>\analysis\arxiv_verification.csv')

ARXIV_ID_RE = re.compile(r'arXiv:\s*(\d{4}\.\d{4,5})', re.IGNORECASE)
BIBITEM_RE = re.compile(
    r'\\bibitem\[([^\]]+)\]\{([^}]+)\}(.*?)(?=(?:\\bibitem|\\end\{thebibliography\}))',
    re.DOTALL)

ATOM_NS = {'a': 'http://www.w3.org/2005/Atom',
           'arxiv': 'http://arxiv.org/schemas/atom'}


def fetch_arxiv(arxiv_id: str) -> dict | None:
    """Return dict with title/authors/year/published or None on miss."""
    url = f'https://export.arxiv.org/api/query?id_list={arxiv_id}'
    try:
        with urllib.request.urlopen(url, timeout=20, context=SSL_CTX) as f:
            data = f.read().decode('utf-8', errors='replace')
    except Exception as e:
        return {'_error': str(e)}
    try:
        root = ET.fromstring(data)
    except Exception as e:
        return {'_error': f'parse: {e}'}
    entry = root.find('a:entry', ATOM_NS)
    if entry is None:
        return None
    title = (entry.findtext('a:title', namespaces=ATOM_NS) or '').strip()
    title = re.sub(r'\s+', ' ', title)
    pub = (entry.findtext('a:published', namespaces=ATOM_NS) or '').strip()
    year = pub[:4] if pub else ''
    authors = []
    for a in entry.findall('a:author', ATOM_NS):
        n = a.findtext('a:name', namespaces=ATOM_NS)
        if n: authors.append(n.strip())
    primary = entry.find('arxiv:primary_category', ATOM_NS)
    cat = primary.get('term') if primary is not None else ''
    return {'title': title, 'first_author': authors[0] if authors else '',
            'all_authors': authors, 'year': year, 'category': cat}


def normalize(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', (s or '').lower())


def parse_bibitem(label: str, key: str, body: str):
    """Extract claimed title, year, author surnames, and arxiv id from bibitem body."""
    body = body.strip()
    # try to extract arxiv id from anywhere in body
    m = ARXIV_ID_RE.search(body)
    arxiv_id = m.group(1) if m else None
    # try to extract a year from the label e.g. "Smith et al.(2024)" or 2024 inline
    year_m = re.search(r'\(?(\d{4})\)?', label)
    label_year = year_m.group(1) if year_m else ''
    # Extract title — usually \newblock\nTitle.\n\newblock or until next \newblock
    # Use \newblock as separator
    blocks = re.split(r'\\newblock', body)
    # blocks[0] = authors, blocks[1] = title, blocks[2..] = venue/year/notes
    authors_raw = blocks[0].strip() if len(blocks) > 0 else ''
    title_raw = blocks[1].strip() if len(blocks) > 1 else ''
    title_raw = title_raw.rstrip('.').strip()
    return {'key': key, 'label': label, 'arxiv_id': arxiv_id,
            'claimed_year': label_year,
            'claimed_title': title_raw,
            'claimed_authors_raw': authors_raw}


def compare(claim: dict, fact: dict | None) -> tuple[str, list[str]]:
    """Return (verdict, reasons)."""
    if fact is None: return ('UNKNOWN_ID', ['arxiv id not found'])
    if '_error' in fact: return ('FETCH_ERROR', [fact['_error']])
    reasons = []
    # Year check
    cy = claim['claimed_year']
    fy = fact.get('year', '')
    if cy and fy and cy != fy:
        # arxiv "year" can be the v1 year; allow ±1
        try:
            if abs(int(cy) - int(fy)) > 1:
                reasons.append(f'year {cy} vs arxiv {fy}')
        except ValueError:
            pass
    # Title check (token overlap)
    ct = normalize(claim['claimed_title'])
    ft = normalize(fact.get('title', ''))
    if ct and ft:
        # require overlap of at least 60% chars in shorter
        shorter, longer = (ct, ft) if len(ct) <= len(ft) else (ft, ct)
        # quick heuristic: shared trigram count
        def trigrams(s): return set(s[i:i+3] for i in range(len(s)-2))
        ts, tl = trigrams(shorter), trigrams(longer)
        overlap = len(ts & tl) / max(len(ts), 1)
        if overlap < 0.5:
            reasons.append(f'title mismatch: claimed≈{shorter[:40]!r} vs arxiv≈{ft[:40]!r}')
    # Author surname check (loose)
    cas = normalize(claim['claimed_authors_raw'])
    fa = normalize(fact.get('first_author', ''))
    # split first author into surname (last word)
    if fa:
        # get last name part
        first_full = fact.get('first_author', '')
        last_name = first_full.split()[-1] if first_full else ''
        if last_name and normalize(last_name) not in cas:
            # not a hard fail since claimed authors string may be abbreviated
            # but flag only if also no etal-style match
            reasons.append(f'first-author surname {last_name!r} not found in claim')
    if reasons: return ('MISMATCH', reasons)
    return ('OK', [])


def main():
    text = PAPER.read_text(encoding='utf-8')
    bibs = BIBITEM_RE.findall(text)
    print(f'Found {len(bibs)} bibitems')

    rows = []
    for label, key, body in bibs:
        claim = parse_bibitem(label, key, body)
        if not claim['arxiv_id']:
            rows.append({**claim, 'verdict': 'NO_ARXIV', 'arxiv_title': '',
                         'arxiv_first_author': '', 'arxiv_year': '', 'reasons': ''})
            continue
        print(f'  fetching {key} -> arXiv:{claim["arxiv_id"]} ...', end=' ', flush=True)
        fact = fetch_arxiv(claim['arxiv_id'])
        time.sleep(3.0)  # arXiv rate limit politeness
        verdict, reasons = compare(claim, fact)
        print(verdict)
        rows.append({
            **claim,
            'verdict': verdict,
            'arxiv_title': (fact or {}).get('title', '')[:120],
            'arxiv_first_author': (fact or {}).get('first_author', ''),
            'arxiv_year': (fact or {}).get('year', ''),
            'reasons': ' | '.join(reasons),
        })

    OUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows: w.writerow(r)
    print(f'\nWrote {OUT_CSV}')

    # Summary
    print('\n=== SUMMARY ===')
    from collections import Counter
    counts = Counter(r['verdict'] for r in rows)
    for v, c in counts.most_common():
        print(f'  {v}: {c}')
    print('\n=== PROBLEM ENTRIES ===')
    for r in rows:
        if r['verdict'] not in ('OK', 'NO_ARXIV'):
            print(f'\n  [{r["verdict"]}]  key={r["key"]}  arxiv={r["arxiv_id"]}')
            print(f'    claimed_title: {r["claimed_title"][:120]}')
            print(f'    arxiv_title:   {r["arxiv_title"]}')
            print(f'    arxiv_first_author: {r["arxiv_first_author"]}')
            print(f'    reasons: {r["reasons"]}')


if __name__ == '__main__':
    main()
