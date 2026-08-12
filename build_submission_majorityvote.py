# -*- coding: utf-8 -*-
"""Cải thiện Outcome (70% điểm) GPU-FREE bằng self-consistency majority vote.

Giữ NGUYÊN case_evidence + law_evidence của submission hiện tại (đã tối ưu: 17.8 chunk/case
E_i=1.0; law top-14). CHỈ thay `prediction`: thay vì lấy `prediction` (decider=
derived_from_claim_outcomes), lấy MAJORITY của `derived_label_votes` (3 vote self-consistency,
Wang 2022), tie-break bằng `holistic_label`, fallback về prediction cũ.

Đo trên PUBLIC (fullchunks/pado Qwen3.5-9B, có gold):
  prediction hiện tại        : 4-label 0.580 / binary 0.740
  majority(votes)+holistic_tb: 4-label 0.620 / binary 0.780   <-- dùng cái này
Đây là kỹ thuật chuẩn (không cherry-pick), đổi 8/60 case private.

Chạy: py -X utf8 build_submission_majorityvote.py
"""
import json, sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent
LAB = ['A_WIN', 'B_WIN', 'PARTIAL_A_WIN', 'PARTIAL_B_WIN']
EXISTING_SUB = ROOT / 'private/[submission] UTEHY_NLU.json'
PRED = ROOT / 'private/predictions_qwen3-5-9b_seed-2026.json'
OUT = ROOT / 'private/submission_UTEHY_NLU_majorityvote.json'


def majority_vote(votes, tiebreak, fallback):
    """THẬN TRỌNG: chỉ override khi có 1 nhãn chiếm đa số TUYỆT ĐỐI (unique max, >=2 vote).
    Khi HÒA phiếu -> giữ nguyên prediction cũ (không tung đồng xu bằng tie-break).
    Trên public bản này = bản full (0.620/0.780) nhưng đổi ÍT hơn -> rủi ro thấp hơn."""
    c = Counter(v for v in (votes or []) if v in LAB)
    if not c:
        return fallback
    mx = max(c.values())
    cand = [l for l in LAB if c[l] == mx]
    if len(cand) == 1 and mx >= 2:
        return cand[0]
    return fallback


def main():
    sub = json.loads(EXISTING_SUB.read_text(encoding='utf-8'))
    pred = json.loads(PRED.read_text(encoding='utf-8'))

    changed = 0
    new = []
    for s in sub:
        cid = s['case_id']
        r = pred.get(cid, {})
        old = s['prediction']
        mv = majority_vote(r.get('derived_label_votes'), r.get('holistic_label'), old)
        if mv != old:
            changed += 1
        new.append({
            'case_id': cid,
            'prediction': mv,
            'case_evidence': s.get('case_evidence', []),   # GIỮ NGUYÊN
            'law_evidence': s.get('law_evidence', []),      # GIỮ NGUYÊN
        })

    assert len(new) == 60
    assert len({x['case_id'] for x in new}) == 60
    assert all(x['prediction'] in LAB for x in new)

    tmp = OUT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(OUT)
    print(f'ĐÃ GHI {OUT.name} (60 case, đổi {changed} prediction vs bản cũ)')
    print('  dist mới:', dict(Counter(x['prediction'] for x in new)))
    print('  dist cũ :', dict(Counter(s['prediction'] for s in sub)))
    print('  avg case_evidence:', round(sum(len(x['case_evidence']) for x in new)/60, 1))
    print('  avg law_evidence :', round(sum(len(x['law_evidence']) for x in new)/60, 1))


if __name__ == '__main__':
    main()
