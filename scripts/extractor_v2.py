"""Robust v2 answer extractor for math/reasoning Tier-2 files.

Priority order (highest first):
  1. \\boxed{X}  (LaTeX, last occurrence)
  2. "Final Answer: X" / "the answer is X" / "answer: X"  (last occurrence)
  3. Last "**X**" markdown bold
  4. Last number / letter in text

For numeric answers (math): take the LAST number in the chosen scope.
For letter answers (mcq): take the FIRST valid letter in the chosen scope.

This module is shared between sanity-check, paper-figures, and
human-spot-check scripts so the extractor is the single source of truth.
"""
from __future__ import annotations
import re

NUM_RE = re.compile(r'-?\d+(?:[,]\d{3})*(?:\.\d+)?')
BOXED_RE = re.compile(r'\\boxed\{([^{}]+)\}')
# Look for the last occurrence of an answer cue and capture up to the next newline / period
ANSWER_CUE_RE = re.compile(
    r'(?:final\s+answer|answer\s+is|answer:|answer\s*=)\s*[:*\-]*\s*([^\n]{0,120})',
    re.IGNORECASE)
BOLD_RE = re.compile(r'\*\*([^*\n]{1,120})\*\*')


def normalise_number(s) -> str:
    """Take the LAST number in the string, return canonical form."""
    if s is None: return ''
    s = str(s).replace('\\$', '').replace('$', '')
    nums = NUM_RE.findall(s)
    if not nums:
        return ''
    n = nums[-1].replace(',', '')
    if '.' in n:
        n = n.rstrip('0').rstrip('.') or '0'
    # strip leading zeros except "0" itself
    if n.startswith('-'):
        sign, n = '-', n[1:]
    else:
        sign = ''
    n = n.lstrip('0') or '0'
    return sign + n


def extract_math(text: str) -> str:
    """Extract numeric final answer."""
    if not isinstance(text, str): return ''
    # 1. boxed
    bm = BOXED_RE.findall(text)
    if bm:
        n = normalise_number(bm[-1])
        if n: return n
    # 2. answer cue (LAST occurrence)
    matches = list(ANSWER_CUE_RE.finditer(text))
    if matches:
        n = normalise_number(matches[-1].group(1))
        if n: return n
    # 3. last bold
    bolds = BOLD_RE.findall(text)
    for b in reversed(bolds):
        n = normalise_number(b)
        if n: return n
    # 4. last number in text
    n = normalise_number(text)
    return n


def _find_letter(scope: str, k: int = 4) -> str:
    valid = 'ABCD'[:k]
    # Prefer first standalone letter token at start
    m = re.search(rf'^[\s*]*\(?([{valid}{valid.lower()}])\)?(?:\b|$)', scope.strip())
    if m: return m.group(1).upper()
    # Otherwise first letter token anywhere
    m = re.search(rf'\b([{valid}{valid.lower()}])\b', scope)
    if m: return m.group(1).upper()
    return ''


def extract_letter(text: str, k: int = 4) -> str:
    if not isinstance(text, str): return ''
    # 1. boxed
    bm = BOXED_RE.findall(text)
    for b in reversed(bm):
        L = _find_letter(b, k)
        if L: return L
    # 2. answer cue (LAST)
    matches = list(ANSWER_CUE_RE.finditer(text))
    for m in reversed(matches):
        L = _find_letter(m.group(1), k)
        if L: return L
    # 3. last bold
    bolds = BOLD_RE.findall(text)
    for b in reversed(bolds):
        L = _find_letter(b, k)
        if L: return L
    # 4. last lone letter token in whole text
    valid = 'ABCD'[:k]
    letters = re.findall(rf'\b([{valid}{valid.lower()}])\b', text)
    if letters: return letters[-1].upper()
    return ''


def is_correct_math(text: str, gold) -> bool:
    pred = extract_math(text)
    g = normalise_number(gold)
    return bool(pred) and pred == g


def is_correct_letter(text: str, gold, k: int = 4) -> bool:
    pred = extract_letter(text, k)
    g = str(gold).strip().upper()[:1] if gold else ''
    return bool(pred) and pred == g


# Quick self-test
if __name__ == '__main__':
    cases = [
        ('Per week, **35 × 7 = 245**. **Final Answer: 245 text messages per week.** 245 text messages per week', 245, True),
        ('The answer is $\\boxed{12}$', 12, True),
        ('Step 1: 6 + 4 = 10. Step 2: 10 * 12 = 120. Final answer: 120.', 120, True),
        ('After verification, **answer = 96**', 96, True),
        ('Step 1: 95% of 40 = 38. Step 2: 40 - 3 = 37. The highest is 38. Hannah needs 39.', 39, True),
    ]
    print('--- math self-test ---')
    for txt, gold, exp in cases:
        got = extract_math(txt)
        ok = (got == normalise_number(gold)) == exp
        print(f'  {"OK" if ok else "FAIL":4s}  gold={gold} got={got!r}')

    cases_l = [
        ('After analysis, the answer is **C. tropical.**', 'C', True),
        ('Final answer: D', 'D', True),
        ('Option A is the correct one.', 'A', True),
        ('I believe option B because... actually wait, the answer is D', 'D', True),
    ]
    print('--- letter self-test ---')
    for txt, gold, exp in cases_l:
        got = extract_letter(txt)
        ok = (got == gold) == exp
        print(f'  {"OK" if ok else "FAIL":4s}  gold={gold} got={got!r}')
