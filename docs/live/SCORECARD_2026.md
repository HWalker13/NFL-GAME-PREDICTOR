# 2026 shadow-mode scorecard

_Generated 2026-09-29T17:32:08Z from 1 graded week file(s): 3. Paper trading only -- no real money.
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

Games logged: 16 (decided 16, tie 0, pending 0).

## Cumulative -- prediction quality

| Predictor | n | Accuracy [95% CI] | Log loss [95% CI] | Brier [95% CI] |
|---|---:|---|---|---|
| Champion | 16 | 10/16 = 0.625 [0.386, 0.815] | 0.673 [0.472, 0.873] | 0.240 [0.149, 0.332] |
| Challenger | 16 | 10/16 = 0.625 [0.386, 0.815] | 0.673 [0.497, 0.849] | 0.240 [0.158, 0.322] |
| Home baseline | 16 | 11/16 = 0.688 [0.444, 0.858] | – | – |
| Vegas favorite (prediction snapshot) | 16 | 8/16 = 0.500 [0.280, 0.720] | – | – |
| Vegas implied (prediction snapshot) | 16 | – | 0.713 [0.524, 0.901] | 0.261 [0.174, 0.349] |

## Cumulative -- paper bets and line movement

| Model | Bets (home) | W-L-Void (pending) | Units | ROI [95% CI] | Mean bet CLV [95% CI] (n, excluded) | One-sided p (CLV>0) | Line move -> model [95% CI] (n) |
|---|---:|---|---:|---|---|---:|---|
| Champion | 11 (10) | 7-4-0 (0) | +3.22 | 0.292 [-0.464, 1.049] | – (n=0, excl 11) | – | – |
| Challenger | 9 (8) | 6-3-0 (0) | +3.92 | 0.436 [-0.458, 1.330] | – (n=0, excl 9) | – | – |
| Official record (bet_status = official) | 11 (10) | 7-4-0 (0) | +3.22 | 0.292 [-0.464, 1.049] | – (n=0, excl 11) | – | – |

`Bets (home)` shows how many bets were on the home side: both models carry the known home-win bias (Phase 8 negative calibration intercepts), so a home-heavy bet mix is expected.

## Paper bets by edge size (DESCRIPTIVE ONLY)

This breakdown is descriptive only. It cannot change the season-end verdict (Section 4), the paper-bet threshold, or the checkpoint decision. Its purpose: if the model carries real information, larger edges should perform better; a flat or non-monotonic pattern indicates edges are mostly model error. Small per-bucket samples are expected and will be labelled.

Edge = model probability minus vig-free implied probability for the side bet, at the prediction snapshot. ROI = units per unit staked on settled bets.

| Model | Edge bucket | Bets | W-L (void, pending) | Units | ROI [95% CI] | Mean bet CLV [95% CI] (n) | Sample |
|---|---|---:|---|---:|---|---|---|
| champion | [0.04, 0.06) | 4 | 2-2 (0, 0) | -0.41 | -0.101 [-1.877, 1.674] | – (n=0) | small sample (n<30) |
| champion | [0.06, 0.08) | 1 | 1-0 (0, 0) | +0.33 | 0.328 | – (n=0) | small sample (n<30) |
| champion | [0.08+) | 6 | 4-2 (0, 0) | +3.29 | 0.549 [-0.772, 1.870] | – (n=0) | small sample (n<30) |
| challenger | [0.04, 0.06) | 4 | 2-2 (0, 0) | -0.37 | -0.093 [-1.875, 1.689] | – (n=0) | small sample (n<30) |
| challenger | [0.06, 0.08) | 0 | 0-0 (0, 0) | +0.00 | – | – (n=0) | small sample (n<30) |
| challenger | [0.08+) | 5 | 4-1 (0, 0) | +4.29 | 0.859 [-0.535, 2.253] | – (n=0) | small sample (n<30) |

## Weekly

| Week | Model | n | Acc | Log loss | Brier | Home acc | Vegas fav acc | Bets | Units | Mean bet CLV (n) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | champion | 16 | 0.625 | 0.6728 | 0.2400 | 0.688 | 0.500 | 11 | +3.22 | – (0) |
| 3 | challenger | 16 | 0.625 | 0.6732 | 0.2404 | 0.688 | 0.500 | 9 | +3.92 | – (0) |

## Midseason checkpoint (after week 9)

Rule (pre-registered): the challenger replaces the champion for the rest of the season ONLY if its mean per-game log loss over the checkpoint window (weeks 4-9) is lower by more than 0.010 (mean challenger - champion < -0.010) AND a one-sided paired t-test on per-game log loss gives p < 0.05 (Phase 8's rule). Otherwise the champion continues and the comparison is recorded. A swap at n = 88 is unlikely; the comparison's main value is informing the 2027 model.

Not yet evaluable: graded checkpoint weeks none, 0 pending game(s). The decision is taken once, after week 9 is fully graded.

## Season-end betting verdict (PROVISIONAL -- the verdict is only taken at season end)

Rule (pre-registered): "evidence of an edge" ONLY if the official mean bet CLV > 0 with a one-sided t-test p < 0.05; the units ROI 95% CI is reported alongside. ROI alone never counts as evidence.

Official bets with measured CLV: n = 0; mean CLV = –; one-sided p = –. Units ROI = 0.292 [-0.464, 1.049].

**No evidence of an edge** under the rule (as of the games graded so far).
