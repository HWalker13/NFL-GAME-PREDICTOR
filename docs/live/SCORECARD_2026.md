# 2026 shadow-mode scorecard

_Generated 2026-11-24T20:51:00Z from 8 graded week file(s): 3, 4, 5, 6, 7, 8, 10, 11. Paper trading only -- no real money.
Rules: `docs/PHASE10_PREREG.md`. Regenerated from the graded files; the logged predictions are never edited._

## How to read this

- **Champion** is the frozen deployment model (LR-tuned, trained 2002-2025). Its paper bets are the
  official record. **Challenger** is the same frozen model with a sigmoid probability calibrator (fit
  out-of-fold on 2021-2025) that corrects its home-win bias; its bets are hypothetical and exist for the
  midseason comparison.
- **Home baseline** always picks the home team. **Vegas favorite** is the side favored by the spread in
  the snapshot the prediction was logged against; **Vegas implied** is the vig-free moneyline probability
  from that same snapshot.
- **Accuracy**: share of games picked correctly. **Log loss**: penalises confident wrong probabilities;
  lower is better; ~0.693 is a coin flip. **Brier**: mean squared error of the probability; lower is
  better; 0.25 is a coin flip.
- **Units / ROI**: profit from flat 1-unit paper bets at the logged moneyline (vig included); ROI =
  units / units staked (void ties excluded).
- **Bet CLV**: how much the vig-free implied probability of the side bet moved between the prediction
  snapshot and the **last pre-kickoff snapshot** (not the true close). Positive = the market moved
  toward the bet. **Line move -> model**: the same movement for every game, signed toward the model's view.
- **Small samples mislead.** A season is ~270 games; this rule bets only a fraction of them. With 50
  bets, a ROI 95% interval is roughly +/-25 points wide, so a hot or cold streak says almost nothing.
  CLV is the less noisy signal, and it is the only one the pre-registered verdict accepts as evidence.
  Every number below comes with its 95% interval; read the interval, not the point estimate.
- Ties are excluded from accuracy / log loss / Brier and are void (0 units) for bets. Pending games
  (no final score yet) are excluded everywhere and counted.

Games logged: 103 (decided 102, tie 1, pending 0).

## Cumulative -- prediction quality

| Predictor | n | Accuracy [95% CI] | Log loss [95% CI] | Brier [95% CI] |
|---|---:|---|---|---|
| Champion | 102 | 63/102 = 0.618 [0.521, 0.706] | 0.638 [0.571, 0.704] | 0.225 [0.195, 0.256] |
| Challenger | 102 | 63/102 = 0.618 [0.521, 0.706] | 0.635 [0.577, 0.694] | 0.224 [0.196, 0.251] |
| Home baseline | 102 | 51/102 = 0.500 [0.405, 0.595] | – | – |
| Vegas favorite (prediction snapshot) | 102 | 63/102 = 0.618 [0.521, 0.706] | – | – |
| Vegas implied (prediction snapshot) | 102 | – | 0.650 [0.578, 0.722] | 0.230 [0.198, 0.262] |

## Cumulative -- paper bets and line movement

| Model | Bets (home) | W-L-Void (pending) | Units | ROI [95% CI] | Mean bet CLV [95% CI] (n, excluded) | One-sided p (CLV>0) | Line move -> model [95% CI] (n) |
|---|---:|---|---:|---|---|---:|---|
| Champion | 66 (41) | 32-33-1 (0) | +6.98 | 0.107 [-0.233, 0.448] | -0.0011 [-0.0103, 0.0082] (n=20, excl 46) | 0.5934 | 0.0006 [-0.0054, 0.0067] (n=38) |
| Challenger | 63 (33) | 30-32-1 (0) | +11.58 | 0.187 [-0.190, 0.564] | -0.0054 [-0.0127, 0.0019] (n=21, excl 42) | 0.9314 | -0.0002 [-0.0063, 0.0058] (n=38) |
| Official record (bet_status = official) | 66 (41) | 32-33-1 (0) | +6.98 | 0.107 [-0.233, 0.448] | -0.0011 [-0.0103, 0.0082] (n=20, excl 46) | 0.5934 | – |

`Bets (home)` shows how many bets were on the home side: both models carry the known home-win bias (Phase 8 negative calibration intercepts), so a home-heavy bet mix is expected.

## Paper bets by edge size (DESCRIPTIVE ONLY)

This breakdown is descriptive only. It cannot change the season-end verdict (Section 4), the paper-bet threshold, or the checkpoint decision. Its purpose: if the model carries real information, larger edges should perform better; a flat or non-monotonic pattern indicates edges are mostly model error. Small per-bucket samples are expected and will be labelled.

Edge = model probability minus vig-free implied probability for the side bet, at the prediction snapshot. ROI = units per unit staked on settled bets.

| Model | Edge bucket | Bets | W-L (void, pending) | Units | ROI [95% CI] | Mean bet CLV [95% CI] (n) | Sample |
|---|---|---:|---|---:|---|---|---|
| champion | [0.04, 0.06) | 14 | 5-8 (1, 0) | -3.54 | -0.273 [-0.884, 0.338] | -0.0052 [-0.0629, 0.0526] (n=3) | small sample (n<30) |
| champion | [0.06, 0.08) | 16 | 10-6 (0, 0) | +5.00 | 0.312 [-0.573, 1.197] | 0.0024 [-0.0191, 0.0239] (n=8) | small sample (n<30) |
| champion | [0.08+) | 36 | 17-19 (0, 0) | +5.52 | 0.153 [-0.306, 0.613] | -0.0027 [-0.0133, 0.0078] (n=9) |  |
| challenger | [0.04, 0.06) | 17 | 9-8 (0, 0) | -1.25 | -0.074 [-0.561, 0.413] | -0.0058 [-0.0295, 0.0180] (n=6) | small sample (n<30) |
| challenger | [0.06, 0.08) | 8 | 3-4 (1, 0) | +1.55 | 0.222 [-1.500, 1.944] | -0.0181 [-0.0523, 0.0160] (n=3) | small sample (n<30) |
| challenger | [0.08+) | 38 | 18-20 (0, 0) | +11.28 | 0.297 [-0.237, 0.831] | -0.0020 [-0.0096, 0.0055] (n=12) |  |

## Value tiers, wins over expected, and toss-ups (DESCRIPTIVE ONLY)

Descriptive only (Amendments 6 and 7). Nothing here changes the paper-bet rule, the checkpoint or the season-end verdict, which rests on CLV alone. Tiers label the official bets by their own edge (Amendment 7): Lean = [0.04, 0.11), ★ = [0.11, 0.16), ★★ = 0.16 and above. A Lean is still an official value call. **WOE** (wins over expected) = wins minus **Vegas expected**, the sum of the vig-free implied probability of each call's side, over graded calls with pushes excluded. **Model expected** = the sum of the calls' edges: the WOE the model's own probabilities predicted. A toss-up is a game whose champion favorite is at 0.55 or below; accuracy follows the usual convention (ties excluded).

| Value calls | Logged | W-L (push, pending) | n (graded, no push) | Won | Vegas expected | Model expected | WOE | WOE per call | Model expected per call |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| All official value calls | 66 | 32-33 (1, 0) | 65 | 32 | 28.903 | 6.603 | +3.097 | +0.0477 | +0.1016 |
| Lean edge [0.04, 0.11) | 45 | 20-24 (1, 0) | 44 | 20 | 20.179 | 3.231 | -0.179 | -0.0041 | +0.0734 |
| ★ edge [0.11, 0.16) | 12 | 6-6 (0, 0) | 12 | 6 | 5.514 | 1.571 | +0.486 | +0.0405 | +0.1309 |
| ★★ edge [0.16+) | 9 | 6-3 (0, 0) | 9 | 6 | 3.209 | 1.801 | +2.791 | +0.3101 | +0.2001 |

| Champion picks | Logged | Accuracy [95% CI] |
|---|---:|---|
| Confident (favorite > 0.55) | 80 | 51/79 = 0.646 [0.536, 0.742] |
| Toss-up (favorite <= 0.55) | 23 | 12/23 = 0.522 [0.330, 0.708] |
| All picks | 103 | 63/102 = 0.618 [0.521, 0.706] |

## Weekly

| Week | Model | n | Acc | Log loss | Brier | Home acc | Vegas fav acc | Bets | Units | Mean bet CLV (n) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | champion | 16 | 0.625 | 0.6728 | 0.2400 | 0.688 | 0.500 | 11 | +3.22 | – (0) |
| 3 | challenger | 16 | 0.625 | 0.6732 | 0.2404 | 0.688 | 0.500 | 9 | +3.92 | – (0) |
| 4 | champion | 16 | 0.625 | 0.5955 | 0.2077 | 0.500 | 0.562 | 11 | +1.20 | – (0) |
| 4 | challenger | 16 | 0.625 | 0.5975 | 0.2070 | 0.500 | 0.562 | 10 | +3.18 | – (0) |
| 5 | champion | 14 | 0.643 | 0.5661 | 0.1902 | 0.357 | 0.643 | 10 | +8.18 | – (0) |
| 5 | challenger | 14 | 0.714 | 0.5712 | 0.1918 | 0.357 | 0.643 | 11 | +7.18 | – (0) |
| 6 | champion | 14 | 0.500 | 0.7377 | 0.2711 | 0.429 | 0.643 | 11 | +2.45 | – (0) |
| 6 | challenger | 14 | 0.500 | 0.7173 | 0.2621 | 0.429 | 0.643 | 10 | +3.45 | – (0) |
| 7 | champion | 14 | 0.571 | 0.7120 | 0.2567 | 0.357 | 0.714 | 9 | -3.06 | -0.0052 (8) |
| 7 | challenger | 14 | 0.571 | 0.6990 | 0.2513 | 0.357 | 0.714 | 9 | +0.48 | -0.0093 (8) |
| 8 | champion | 1 | 0.000 | 1.0649 | 0.4293 | 0.000 | 0.000 | 1 | -1.00 | – (0) |
| 8 | challenger | 1 | 0.000 | 0.9916 | 0.3957 | 0.000 | 0.000 | 0 | +0.00 | – (0) |
| 10 | champion | 14 | 0.643 | 0.6257 | 0.2207 | 0.286 | 0.643 | 5 | -3.77 | -0.0073 (4) |
| 10 | challenger | 14 | 0.571 | 0.6200 | 0.2177 | 0.286 | 0.643 | 5 | -3.77 | -0.0073 (4) |
| 11 | champion | 13 | 0.769 | 0.5158 | 0.1710 | 0.923 | 0.692 | 8 | -0.23 | 0.0062 (8) |
| 11 | challenger | 13 | 0.769 | 0.5375 | 0.1789 | 0.923 | 0.692 | 9 | -2.85 | -0.0011 (9) |

## Midseason checkpoint (weeks 4-9)

Rule (pre-registered): the challenger replaces the champion for the rest of the season ONLY if its mean per-game log loss over the checkpoint window (weeks 4-9) is lower by more than 0.010 (mean challenger - champion < -0.010) AND a one-sided paired t-test on per-game log loss gives p < 0.05 (Phase 8's rule). Otherwise the champion continues and the comparison is recorded. The comparison's main value is informing the 2027 model.

Evaluable (Amendment 4) once every window week is fully graded or is a documented missed week (no prediction file, every kickoff past); a missed week contributes 0 games and a partially missed week only its predicted games.

| Week | Scheduled | Predicted | Paired (decided) | Status |
|---:|---:|---:|---:|---|
| 4 | 16 | 16 | 16 | graded |
| 5 | 15 | 15 | 14 | graded |
| 6 | 14 | 14 | 14 | graded |
| 7 | 14 | 14 | 14 | graded |
| 8 | 14 | 1 | 1 | graded |
| 9 | 15 | 0 | 0 | missed (no prediction file, every kickoff past) |

Planned window: 88 games; paired games actually graded for both models: n = 59.

Mean per-game log loss: champion 0.65786, challenger 0.65046; mean difference (challenger - champion) = -0.00740; one-sided paired t p = 0.1087.

**Decision: NO SWAP -- champion continues.**

## Season-end betting verdict (PROVISIONAL -- the verdict is only taken at season end)

Rule (pre-registered): "evidence of an edge" ONLY if the official mean bet CLV > 0 with a one-sided t-test p < 0.05; the units ROI 95% CI is reported alongside. ROI alone never counts as evidence.

Official bets with measured CLV: n = 20; mean CLV = -0.0011 [-0.0103, 0.0082]; one-sided p = 0.5934. Units ROI = 0.107 [-0.233, 0.448].

**No evidence of an edge** under the rule (as of the games graded so far).
