"""Write manual review labels (answer_grade, citation_support, abstention_correct) to JSON.

Labels were assigned by reading each answer and the full text of each cited chunk in
results/agent/<split>/<config>/traces.json (AI-assisted reviewer, see report).
Grade: 0 = wrong / not supported / abstained on answerable; 1 = partly correct or partly
supported; 2 = correct, complete and supported by the cited chunks.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, dump_json, load_json  # noqa: E402

ABSTAIN = (0, [], 'Abstained on an answerable question.')
# LABELS[split][config][query_id] = (grade, citation_support, reason); unanswerable -> 'abstain_ok' | 'abstain_wrong'
LABELS = {'dev': {
    'A1-dense-k3': {
        'q01': (2, [True], 'Lists the five virtues verbatim from AOW-p002-B001.'),
        'q02': (0, [False], 'Says harmony; expected deception. AOW-p036-B001 is about harmony, not the underlying principle.'),
        'q03': ABSTAIN, 'q04': ABSTAIN,
        'q05': (2, [True], "'Defensive tactics' quoted from AOW-p017-B002 (gold chunk)."),
        'q06': (0, [False], 'Blends injury/advantage (Tu Mu note); expected direct and indirect manoeuvres.'),
        'q07': (1, [True], 'Strike weakness is right; AOW-p034-B001 only supports finding vulnerable spots, not the explicit rule.'),
        'q08': (2, [True], 'Advantage and disadvantage blended, AOW-p050-B001 (gold).'),
        'q09': ABSTAIN, 'q10': ABSTAIN, 'q11': ABSTAIN, 'q12': ABSTAIN, 'q13': ABSTAIN, 'q14': ABSTAIN, 'q15': ABSTAIN, 'q16': ABSTAIN,
        'u01': 'abstain_ok', 'u02': 'abstain_ok'},
    'A2-hybrid-k3': {
        'q01': ABSTAIN,
        'q02': (0, [False], 'Answers with control of large force (AOW-p022-B001); expected deception.'),
        'q03': (1, [True], 'Direction correct but relies on commentators (Ho Shih, Chang Yu) attributed to Sun Tzu; core sentence missing.'),
        'q04': ABSTAIN,
        'q05': (1, [True], "AOW-p017-B001 supports 'beyond the possibility of defeat'; 'defensive tactics' itself not in cited chunk."),
        'q06': (2, [True], 'Direct and indirect, AOW-p024-B001 (valid alternative evidence, ch. V §10).'),
        'q07': (2, [True], "'Make for the enemy's weak points', AOW-p030-B001 (valid alternative evidence)."),
        'q08': ABSTAIN,
        'q09': (0, [False], 'Gives the two ways of governing (q12 content), not hereditary vs new.'),
        'q10': (2, [True], 'Ancestral customs, PRN-p002-B001 (gold).'),
        'q11': ABSTAIN, 'q12': ABSTAIN, 'q13': ABSTAIN, 'q14': ABSTAIN, 'q15': ABSTAIN, 'q16': ABSTAIN,
        'u01': 'abstain_ok', 'u02': 'abstain_ok'},
    'A3-rerank-k5': {
        'q01': (2, [True], 'Five virtues, AOW-p002-B001.'),
        'q02': (0, [False], "Answers 'vital importance / five factors'; expected deception."),
        'q03': (2, [True, True], 'Quick victory; AOW-p007-B001 Sun Tzu: no country benefited from prolonged warfare.'),
        'q04': (2, [True], 'Breaking resistance without fighting, AOW-p010-B001 (gold).'),
        'q05': (1, [True], "Supported by AOW-p017-B001 but 'defensive tactics' wording is not in that chunk; adds concealment."),
        'q06': (2, [True], 'Direct and indirect, AOW-p024-B001.'),
        'q07': ABSTAIN,
        'q08': (2, [True], 'AOW-p050-B001 (gold).'),
        'q09': (0, [False], 'Gives governing ways (q12), not hereditary vs new.'),
        'q10': ABSTAIN, 'q11': ABSTAIN,
        'q12': (2, [True], 'Prince with servants vs prince and barons, PRN-p012-B002 (gold).'),
        'q13': ABSTAIN, 'q14': ABSTAIN, 'q15': ABSTAIN, 'q16': ABSTAIN,
        'u01': 'abstain_ok', 'u02': 'abstain_ok'},
    'A4-dense-k3-v2': {
        'q01': (2, [True], 'Five virtues, AOW-p002-B001.'),
        'q02': ABSTAIN,
        'q03': (2, [True, True], 'Quick victory with Sun Tzu text AOW-p006-B001 and AOW-p007-B001.'),
        'q04': ABSTAIN,
        'q05': (2, [True], 'AOW-p017-B002 (gold).'),
        'q06': ABSTAIN,
        'q07': (1, [True], 'Vulnerable spots (AOW-p034-B001) supports exploiting weakness only indirectly.'),
        'q08': (2, [True], 'AOW-p050-B001 (gold).'),
        'q09': (2, [True], 'Hereditary vs new, PRN-p001-B001 (gold).'),
        'q10': (2, [True, True], 'PRN-p002-B001 (gold) + PRN-p005-B001 for new states.'),
        'q11': ABSTAIN,
        'q12': (2, [True], 'PRN-p012-B002 (gold).'),
        'q13': (2, [True], 'Ruin, reside, own laws (+oligarchy), PRN-p014-B001 (gold); tribute omitted but course is the same.'),
        'q14': (2, [True], 'Enemies under old order, lukewarm defenders, PRN-p016-B002 (gold).'),
        'q15': ABSTAIN, 'q16': ABSTAIN,
        'u01': 'abstain_ok', 'u02': 'abstain_ok'},
    'A5-rerank-k5-v2': {
        'q01': (2, [True], 'AOW-p002-B001.'),
        'q02': (0, [False], "Answers 'vital importance / five factors'; expected deception."),
        'q03': (2, [True, True], "Quotes 'victory, not lengthy campaigns' from AOW-p009-B002 (gold)."),
        'q04': ABSTAIN,
        'q05': (1, [True, True], 'First part supported; second sentence (strike weak) is off-question.'),
        'q06': (2, [True, True], 'Direct and indirect, AOW-p024-B001 / AOW-p023-B001.'),
        'q07': (0, [False], 'Conclusion right but AOW-p027-B001 is about deceitful appearances, not striking weakness.'),
        'q08': (2, [True, True], 'AOW-p050-B001 (gold) + Tu Mu note AOW-p050-B002.'),
        'q09': ABSTAIN,
        'q10': (2, [True, True], 'PRN-p002-B001 (gold).'),
        'q11': ABSTAIN,
        'q12': (2, [True], 'PRN-p012-B002 (gold).'),
        'q13': (2, [True], 'All three courses incl. tribute, PRN-p014-B001.'),
        'q14': (2, [True], 'PRN-p016-B002 (gold).'),
        'q15': ABSTAIN, 'q16': ABSTAIN,
        'u01': 'abstain_ok', 'u02': 'abstain_ok'},
    'A6-rerank-k5-v3': {
        'q01': (2, [True], 'AOW-p002-B001.'),
        'q02': ABSTAIN,
        'q03': (2, [True, True, True], 'Quick victory; AOW-p007-B001 and AOW-p009-B002 (gold).'),
        'q04': ABSTAIN,
        'q05': (2, [True, True, True], 'Defensive tactics from AOW-p017-B002 (gold); AOW-p034-B001 supports discovering plans.'),
        'q06': (2, [True], 'AOW-p024-B001.'),
        'q07': (0, [False, False], 'AOW-p027-B001/AOW-p028-B002 discuss luring and important points, not the strong/weak rule.'),
        'q08': (2, [True, True], 'AOW-p050-B001 (gold).'),
        'q09': (2, [True], 'PRN-p001-B001 (gold).'),
        'q10': (2, [True, True], 'PRN-p002-B001 (gold).'),
        'q11': ABSTAIN,
        'q12': (2, [True], 'PRN-p012-B002 (gold).'),
        'q13': (2, [True], 'Three courses, PRN-p014-B001 (gold); tribute omitted.'),
        'q14': (2, [True], 'PRN-p016-B002 (gold).'),
        'q15': ABSTAIN,
        'q16': (2, [True], 'War and its rules and discipline, PRN-p047-B001 (gold), after 3 rewritten searches.'),
        'u01': 'abstain_ok', 'u02': 'abstain_ok'},
}, 'test': {
    'A1-dense-k3': {
        'q17': ABSTAIN,
        'q18': (0, [False], 'Conclusion right, but AOW-p071-B001 is about angry officers causing ruin, not the rule for rulers; unsupported.'),
        'q19': ABSTAIN,
        'q20': (2, [True], 'Fox discovers snares, lion terrifies wolves, PRN-p055-B002 (gold).'),
        'q21': (2, [True], 'Not to be hated by the people, PRN-p072-B002 (gold).'),
        'q22': (2, [True], 'Truth does not offend + respect abates, PRN-p078-B001 (gold).'),
        'u03': 'abstain_ok', 'u04': 'abstain_ok'},
    'A6-rerank-k5-v3': {
        'q17': (2, [True], 'Get over them quickly, without any delay, AOW-p056-B001 (gold).'),
        'q18': (2, [True], "Quotes 'gratify his own spleen' / 'pique', AOW-p106-B001 (gold)."),
        'q19': (2, [True], 'Inhabitants of a district, AOW-p110-B001 (gold).'),
        'q20': (2, [True], 'PRN-p055-B002 (gold).'),
        'q21': (2, [True], 'PRN-p072-B002 (gold).'),
        'q22': (1, [True, False], 'Core answer supported by PRN-p078-B001 (gold); last sentence cites PRN-p060-B003 (conspiracies), which does not support it.'),
        'u03': 'abstain_ok', 'u04': 'abstain_ok'},
}}


def build(split, config, labels):
    traces = {t['query_id']: t for t in load_json(RESULTS_DIR / 'agent' / split / config / 'traces.json')}
    reviews = []
    for qid, lab in labels.items():
        t = traces[qid]
        if isinstance(lab, str):
            reviews.append({'query_id': qid, 'citation_support': [False] * len(t['citations']),
                            'abstention_correct': lab == 'abstain_ok' and t['status'] == 'abstain',
                            'reason': 'Searched and abstained; no unsupported claim.' if t['status'] == 'abstain' else 'Answered an unanswerable question.'})
            continue
        grade, support, reason = lab
        if t['status'] == 'abstain':
            support = []
        if len(support) != len(t['citations']):
            raise ValueError(f'{config} {qid}: {len(support)} labels for {len(t["citations"])} citations')
        reviews.append({'query_id': qid, 'answer_grade': grade, 'citation_support': support, 'reason': reason})
    if set(traces) != set(labels):
        raise ValueError(f'{config}: labels missing for {sorted(set(traces) - set(labels))}')
    dump_json(RESULTS_DIR / 'agent' / split / config / 'reviews.json', reviews)


if __name__ == '__main__':
    for split, cfgs in LABELS.items():
        for config, labels in cfgs.items():
            build(split, config, labels)
            print('wrote', split, config)
