"""AIME/AMC grading fix: hendrycks is_equiv treats '016' and '16' as different, but AIME answers are integers 0-999
and leading zeros are common. aime_equiv = is_equiv OR (both parse as integers and are equal)."""
import re
from lm_eval.tasks.hendrycks_math.utils import is_equiv
_INT = re.compile(r'^[+-]?\d+(\.0*)?$')
def _as_int(s):
    t = str(s).strip().strip('$').replace(',', '').replace(' ', '')
    t = re.sub(r'^\\text\{(.*)\}$', r'\1', t)
    return int(t.split('.')[0]) if _INT.match(t) else None   # parse as int directly (no float round trip, avoids overflow on very long integers)
def aime_equiv(gold, pred):
    if is_equiv(str(gold), str(pred)): return True
    a, b = _as_int(gold), _as_int(pred)
    return a is not None and b is not None and a == b
