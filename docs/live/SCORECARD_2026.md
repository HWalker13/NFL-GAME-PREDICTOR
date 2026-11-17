# 2026 shadow-mode scorecard

_Generated 2026-11-17T23:33:32Z from 7 graded week file(s): 3, 4, 5, 6, 7, 8, 10. Paper trading only -- no real money.
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

Games logged: 90 (decided 89, tie 1, pending 0).

## Cumulative -- prediction quality

| Predictor | n | Accuracy [95% CI] | Log loss [95% CI] | Brier [95% CI] |
|---|---:|---|---|---|
| Champion | 89 | 53/89 = 0.596 [0.492, 0.691] | 0.655 [0.585, 0.726] | 0.233 [0.200, 0.266] |
| Challenger | 89 | 53/89 = 0.596 [0.492, 0.691] | 0.650 [0.587, 0.712] | 0.230 [0.201, 0.259] |
| Home baseline | 89 | 39/89 = 0.438 [0.340, 0.542] | – | – |
| Vegas favorite (prediction snapshot) | 89 | 54/89 = 0.607 [0.503, 0.702] | – | – |
| Vegas implied (prediction snapshot) | 89 | – | 0.666 [0.592, 0.741] | 0.237 [0.203, 0.270] |

## Cumulative -- paper bets and line movement

| Model | Bets (home) | W-L-Void (pending) | Units | ROI [95% CI] | Mean bet CLV [95% CI] (n, excluded) | One-sided p (CLV>0) | Line move -> model [95% CI] (n) |
|---|---:|---|---:|---|---|---:|---|
| Champion | 58 (38) | 29-28-1 (0) | +7.21 | 0.127 [-0.239, 0.492] | -0.0059 [-0.0200, 0.0082] (n=12, excl 46) | 0.8123 | -0.0012 [-0.0091, 0.0067] (n=25) |
| Challenger | 54 (31) | 28-25-1 (0) | +14.43 | 0.272 [-0.141, 0.686] | -0.0086 [-0.0209, 0.0037] (n=12, excl 42) | 0.9237 | -0.0032 [-0.0110, 0.0046] (n=25) |
| Official record (bet_status = official) | 58 (38) | 29-28-1 (0) | +7.21 | 0.127 [-0.239, 0.492] | -0.0059 [-0.0200, 0.0082] (n=12, excl 46) | 0.8123 | – |

`Bets (home)` shows how many bets were on the home side: both models carry the known home-win bias (Phase 8 negative calibration intercepts), so a home-heavy bet mix is expected.

## Paper bets by edge size (DESCRIPTIVE ONLY)

This breakdown is descriptive only. It cannot change the season-end verdict (Section 4), the paper-bet threshold, or the checkpoint decision. Its purpose: if the model carries real information, larger edges should perform better; a flat or non-monotonic pattern indicates edges are mostly model error. Small per-bucket samples are expected and will be labelled.

Edge = model probability minus vig-free implied probability for the side bet, at the prediction snapshot. ROI = units per unit staked on settled bets.

| Model | Edge bucket | Bets | W-L (void, pending) | Units | ROI [95% CI] | Mean bet CLV [95% CI] (n) | Sample |
|---|---|---:|---|---:|---|---|---|
| champion | [0.04, 0.06) | 13 | 5-7 (1, 0) | -2.54 | -0.212 [-0.867, 0.443] | -0.0090 [-0.2920, 0.2739] (n=2) | small sample (n<30) |
| champion | [0.06, 0.08) | 12 | 9-3 (0, 0) | +7.38 | 0.615 [-0.519, 1.749] | -0.0042 [-0.0570, 0.0485] (n=4) | small sample (n<30) |
| champion | [0.08+) | 33 | 15-18 (0, 0) | +2.37 | 0.072 [-0.395, 0.539] | -0.0060 [-0.0215, 0.0095] (n=6) |  |
| challenger | [0.04, 0.06) | 15 | 9-6 (0, 0) | +0.75 | 0.050 [-0.471, 0.571] | -0.0081 [-0.0537, 0.0376] (n=4) | small sample (n<30) |
| challenger | [0.06, 0.08) | 7 | 3-3 (1, 0) | +2.55 | 0.426 [-1.623, 2.475] | -0.0176 [-0.1919, 0.1567] (n=2) | small sample (n<30) |
| challenger | [0.08+) | 32 | 16-16 (0, 0) | +11.13 | 0.348 [-0.247, 0.943] | -0.0060 [-0.0215, 0.0095] (n=6) |  |

## Value tiers, wins over expected, and toss-ups (DESCRIPTIVE ONLY)

Descriptive only (Amendments 6 and 7). Nothing here changes the paper-bet rule, the checkpoint or the season-end verdict, which rests on CLV alone. Tiers label the official bets by their own edge (Amendment 7): Lean = [0.04, 0.11), ★ = [0.11, 0.16), ★★ = 0.16 and above. A Lean is still an official value call. **WOE** (wins over expected) = wins minus **Vegas expected**, the sum of the vig-free implied probability of each call's side, over graded calls with pushes excluded. **Model expected** = the sum of the calls' edges: the WOE the model's own probabilities predicted. A toss-up is a game whose champion favorite is at 0.55 or below; accuracy follows the usual convention (ties excluded).

| Value calls | Logged | W-L (push, pending) | n (graded, no push) | Won | Vegas expected | Model expected | WOE | WOE per call | Model expected per call |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| All official value calls | 58 | 29-28 (1, 0) | 57 | 29 | 26.721 | 5.937 | +2.279 | +0.0400 | +0.1042 |
| Lean edge [0.04, 0.11) | 38 | 18-19 (1, 0) | 37 | 18 | 18.341 | 2.738 | -0.341 | -0.0092 | +0.0740 |
| ★ edge [0.11, 0.16) | 12 | 6-6 (0, 0) | 12 | 6 | 5.514 | 1.571 | +0.486 | +0.0405 | +0.1309 |
| ★★ edge [0.16+) | 8 | 5-3 (0, 0) | 8 | 5 | 2.866 | 1.628 | +2.134 | +0.2667 | +0.2035 |

| Champion picks | Logged | Accuracy [95% CI] |
|---|---:|---|
| Confident (favorite > 0.55) | 69 | 43/68 = 0.632 [0.514, 0.737] |
| Toss-up (favorite <= 0.55) | 21 | 10/21 = 0.476 [0.283, 0.676] |
| All picks | 90 | 53/89 = 0.596 [0.492, 0.691] |

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

Official bets with measured CLV: n = 12; mean CLV = -0.0059 [-0.0200, 0.0082]; one-sided p = 0.8123. Units ROI = 0.127 [-0.239, 0.492].

**No evidence of an edge** under the rule (as of the games graded so far).
