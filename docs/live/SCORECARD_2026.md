# 2026 shadow-mode scorecard

_Generated 2026-11-02T20:18:46Z from 5 graded week file(s): 3, 4, 5, 6, 7. Paper trading only -- no real money.
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

Games logged: 75 (decided 74, tie 1, pending 0).

## Cumulative -- prediction quality

| Predictor | n | Accuracy [95% CI] | Log loss [95% CI] | Brier [95% CI] |
|---|---:|---|---|---|
| Champion | 74 | 44/74 = 0.595 [0.481, 0.699] | 0.656 [0.577, 0.734] | 0.233 [0.197, 0.268] |
| Challenger | 74 | 45/74 = 0.608 [0.494, 0.711] | 0.651 [0.582, 0.719] | 0.230 [0.198, 0.262] |
| Home baseline | 74 | 35/74 = 0.473 [0.363, 0.585] | – | – |
| Vegas favorite (prediction snapshot) | 74 | 45/74 = 0.608 [0.494, 0.711] | – | – |
| Vegas implied (prediction snapshot) | 74 | – | 0.681 [0.597, 0.765] | 0.243 [0.205, 0.280] |

## Cumulative -- paper bets and line movement

| Model | Bets (home) | W-L-Void (pending) | Units | ROI [95% CI] | Mean bet CLV [95% CI] (n, excluded) | One-sided p (CLV>0) | Line move -> model [95% CI] (n) |
|---|---:|---|---:|---|---|---:|---|
| Champion | 52 (35) | 28-23-1 (0) | +11.98 | 0.235 [-0.162, 0.632] | -0.0052 [-0.0272, 0.0168] (n=8, excl 44) | 0.7038 | -0.0054 [-0.0189, 0.0081] (n=13) |
| Challenger | 49 (29) | 27-21-1 (0) | +18.21 | 0.379 [-0.065, 0.823] | -0.0093 [-0.0281, 0.0096] (n=8, excl 41) | 0.8579 | -0.0054 [-0.0189, 0.0081] (n=13) |
| Official record (bet_status = official) | 52 (35) | 28-23-1 (0) | +11.98 | 0.235 [-0.162, 0.632] | -0.0052 [-0.0272, 0.0168] (n=8, excl 44) | 0.7038 | – |

`Bets (home)` shows how many bets were on the home side: both models carry the known home-win bias (Phase 8 negative calibration intercepts), so a home-heavy bet mix is expected.

## Paper bets by edge size (DESCRIPTIVE ONLY)

This breakdown is descriptive only. It cannot change the season-end verdict (Section 4), the paper-bet threshold, or the checkpoint decision. Its purpose: if the model carries real information, larger edges should perform better; a flat or non-monotonic pattern indicates edges are mostly model error. Small per-bucket samples are expected and will be labelled.

Edge = model probability minus vig-free implied probability for the side bet, at the prediction snapshot. ROI = units per unit staked on settled bets.

| Model | Edge bucket | Bets | W-L (void, pending) | Units | ROI [95% CI] | Mean bet CLV [95% CI] (n) | Sample |
|---|---|---:|---|---:|---|---|---|
| champion | [0.04, 0.06) | 12 | 5-6 (1, 0) | -1.54 | -0.140 [-0.845, 0.565] | -0.0090 [-0.2920, 0.2739] (n=2) | small sample (n<30) |
| champion | [0.06, 0.08) | 10 | 8-2 (0, 0) | +8.15 | 0.815 [-0.531, 2.161] | -0.0073 [-0.1064, 0.0919] (n=3) | small sample (n<30) |
| champion | [0.08+) | 30 | 15-15 (0, 0) | +5.37 | 0.179 [-0.319, 0.677] | -0.0006 [-0.0418, 0.0406] (n=3) |  |
| challenger | [0.04, 0.06) | 14 | 8-6 (0, 0) | +0.52 | 0.037 [-0.526, 0.601] | -0.0124 [-0.0957, 0.0709] (n=3) | small sample (n<30) |
| challenger | [0.06, 0.08) | 7 | 3-3 (1, 0) | +2.55 | 0.426 [-1.623, 2.475] | -0.0176 [-0.1919, 0.1567] (n=2) | small sample (n<30) |
| challenger | [0.08+) | 28 | 16-12 (0, 0) | +15.13 | 0.540 [-0.111, 1.191] | -0.0006 [-0.0418, 0.0406] (n=3) | small sample (n<30) |

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

## Midseason checkpoint (after week 9)

Rule (pre-registered): the challenger replaces the champion for the rest of the season ONLY if its mean per-game log loss over the checkpoint window (weeks 4-9) is lower by more than 0.010 (mean challenger - champion < -0.010) AND a one-sided paired t-test on per-game log loss gives p < 0.05 (Phase 8's rule). Otherwise the champion continues and the comparison is recorded. A swap at n = 88 is unlikely; the comparison's main value is informing the 2027 model.

Not yet evaluable: graded checkpoint weeks [4, 5, 6, 7], 0 pending game(s). The decision is taken once, after week 9 is fully graded.

## Season-end betting verdict (PROVISIONAL -- the verdict is only taken at season end)

Rule (pre-registered): "evidence of an edge" ONLY if the official mean bet CLV > 0 with a one-sided t-test p < 0.05; the units ROI 95% CI is reported alongside. ROI alone never counts as evidence.

Official bets with measured CLV: n = 8; mean CLV = -0.0052 [-0.0272, 0.0168]; one-sided p = 0.7038. Units ROI = 0.235 [-0.162, 0.632].

**No evidence of an edge** under the rule (as of the games graded so far).
