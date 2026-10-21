# 2026 shadow-mode scorecard

_Generated 2026-10-21T21:34:53Z from 4 graded week file(s): 3, 4, 5, 6. Paper trading only -- no real money.
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

Games logged: 61 (decided 60, tie 1, pending 0).

## Cumulative -- prediction quality

| Predictor | n | Accuracy [95% CI] | Log loss [95% CI] | Brier [95% CI] |
|---|---:|---|---|---|
| Champion | 60 | 36/60 = 0.600 [0.474, 0.714] | 0.642 [0.555, 0.729] | 0.227 [0.187, 0.267] |
| Challenger | 60 | 37/60 = 0.617 [0.490, 0.729] | 0.639 [0.563, 0.716] | 0.225 [0.190, 0.261] |
| Home baseline | 60 | 30/60 = 0.500 [0.377, 0.623] | – | – |
| Vegas favorite (prediction snapshot) | 60 | 35/60 = 0.583 [0.457, 0.699] | – | – |
| Vegas implied (prediction snapshot) | 60 | – | 0.682 [0.587, 0.777] | 0.244 [0.201, 0.286] |

## Cumulative -- paper bets and line movement

| Model | Bets (home) | W-L-Void (pending) | Units | ROI [95% CI] | Mean bet CLV [95% CI] (n, excluded) | One-sided p (CLV>0) | Line move -> model [95% CI] (n) |
|---|---:|---|---:|---|---|---:|---|
| Champion | 43 (30) | 25-17-1 (0) | +15.05 | 0.358 [-0.095, 0.812] | 0.0000 [-0.0000, 0.0000] (n=11, excl 32) | 0.4078 | 0.0000 [-0.0000, 0.0000] (n=16) |
| Challenger | 40 (25) | 24-15-1 (0) | +17.73 | 0.455 [-0.026, 0.935] | 0.0000 [-0.0000, 0.0000] (n=10, excl 30) | 0.1644 | 0.0000 [-0.0000, 0.0000] (n=16) |
| Official record (bet_status = official) | 43 (30) | 25-17-1 (0) | +15.05 | 0.358 [-0.095, 0.812] | 0.0000 [-0.0000, 0.0000] (n=11, excl 32) | 0.4078 | – |

`Bets (home)` shows how many bets were on the home side: both models carry the known home-win bias (Phase 8 negative calibration intercepts), so a home-heavy bet mix is expected.

## Paper bets by edge size (DESCRIPTIVE ONLY)

This breakdown is descriptive only. It cannot change the season-end verdict (Section 4), the paper-bet threshold, or the checkpoint decision. Its purpose: if the model carries real information, larger edges should perform better; a flat or non-monotonic pattern indicates edges are mostly model error. Small per-bucket samples are expected and will be labelled.

Edge = model probability minus vig-free implied probability for the side bet, at the prediction snapshot. ROI = units per unit staked on settled bets.

| Model | Edge bucket | Bets | W-L (void, pending) | Units | ROI [95% CI] | Mean bet CLV [95% CI] (n) | Sample |
|---|---|---:|---|---:|---|---|---|
| champion | [0.04, 0.06) | 10 | 5-4 (1, 0) | +0.46 | 0.051 [-0.774, 0.876] | -0.0000 [-0.0000, 0.0000] (n=2) | small sample (n<30) |
| champion | [0.06, 0.08) | 7 | 6-1 (0, 0) | +8.12 | 1.160 [-0.821, 3.140] | -0.0000 (n=1) | small sample (n<30) |
| champion | [0.08+) | 26 | 14-12 (0, 0) | +6.47 | 0.249 [-0.289, 0.787] | 0.0000 [-0.0000, 0.0000] (n=8) | small sample (n<30) |
| challenger | [0.04, 0.06) | 11 | 7-4 (0, 0) | +1.85 | 0.168 [-0.491, 0.827] | 0.0000 [-0.0000, 0.0000] (n=2) | small sample (n<30) |
| challenger | [0.06, 0.08) | 5 | 2-2 (1, 0) | -0.35 | -0.086 [-1.902, 1.729] | – (n=0) | small sample (n<30) |
| challenger | [0.08+) | 24 | 15-9 (0, 0) | +16.23 | 0.676 [-0.043, 1.395] | 0.0000 [-0.0000, 0.0000] (n=8) | small sample (n<30) |

## Weekly

| Week | Model | n | Acc | Log loss | Brier | Home acc | Vegas fav acc | Bets | Units | Mean bet CLV (n) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | champion | 16 | 0.625 | 0.6728 | 0.2400 | 0.688 | 0.500 | 11 | +3.22 | – (0) |
| 3 | challenger | 16 | 0.625 | 0.6732 | 0.2404 | 0.688 | 0.500 | 9 | +3.92 | – (0) |
| 4 | champion | 16 | 0.625 | 0.5955 | 0.2077 | 0.500 | 0.562 | 11 | +1.20 | 0.0000 (11) |
| 4 | challenger | 16 | 0.625 | 0.5975 | 0.2070 | 0.500 | 0.562 | 10 | +3.18 | 0.0000 (10) |
| 5 | champion | 14 | 0.643 | 0.5661 | 0.1902 | 0.357 | 0.643 | 10 | +8.18 | – (0) |
| 5 | challenger | 14 | 0.714 | 0.5712 | 0.1918 | 0.357 | 0.643 | 11 | +7.18 | – (0) |
| 6 | champion | 14 | 0.500 | 0.7377 | 0.2711 | 0.429 | 0.643 | 11 | +2.45 | – (0) |
| 6 | challenger | 14 | 0.500 | 0.7173 | 0.2621 | 0.429 | 0.643 | 10 | +3.45 | – (0) |

## Midseason checkpoint (after week 9)

Rule (pre-registered): the challenger replaces the champion for the rest of the season ONLY if its mean per-game log loss over the checkpoint window (weeks 4-9) is lower by more than 0.010 (mean challenger - champion < -0.010) AND a one-sided paired t-test on per-game log loss gives p < 0.05 (Phase 8's rule). Otherwise the champion continues and the comparison is recorded. A swap at n = 88 is unlikely; the comparison's main value is informing the 2027 model.

Not yet evaluable: graded checkpoint weeks [4, 5, 6], 0 pending game(s). The decision is taken once, after week 9 is fully graded.

## Season-end betting verdict (PROVISIONAL -- the verdict is only taken at season end)

Rule (pre-registered): "evidence of an edge" ONLY if the official mean bet CLV > 0 with a one-sided t-test p < 0.05; the units ROI 95% CI is reported alongside. ROI alone never counts as evidence.

Official bets with measured CLV: n = 11; mean CLV = 0.0000 [-0.0000, 0.0000]; one-sided p = 0.4078. Units ROI = 0.358 [-0.095, 0.812].

**No evidence of an edge** under the rule (as of the games graded so far).
