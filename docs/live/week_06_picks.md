# 2026 Week 6 -- model picks (before kickoff, no results yet)

_Predictions logged Wed 10/14 10:42 AM Pacific, against betting odds pulled Wed 10/14 10:41 AM Pacific. This page is regenerated from the official record and is not itself the record. Generated 2026-10-14T17:42:17Z._

## How to read this page

- **This is paper trading only.** No real money is bet, and nothing here is betting advice. The
  project is testing, in public and in advance, whether a statistical model knows anything the
  betting market doesn't. The honest expectation is that it doesn't.
- **What the model predicts: who wins.** Each game gets the model's probability that each team
  wins outright. It does **not** predict the score or the margin, so it says **nothing** about the
  point spread ("covering") or the over/under.
- **Model's chance**: the model's probability that the team it picks wins.
- **Vegas favorite**: who the betting market favors, and by how many points ("CIN by 3.5" means the
  market expects CIN to win by about 3.5). Shown for context only; the model does not use it.
- **Vegas's chance**: the market's probability for the **same team the model picks**, worked out
  from the betting odds with the bookmaker's built-in cut removed.
- **Gap**: model's chance minus Vegas's chance for that team, in percentage points. It is
  calculated before rounding, so it can differ by a point from subtracting the two whole numbers.
- **Paper bet**: an imaginary $1 **moneyline** bet (a bet on who wins, not on the spread or the
  total). The rule, fixed before the season: bet on a team whenever the model gives it a chance at
  least **4 points higher** than Vegas does. Otherwise, no bet. The number in brackets, e.g.
  "Bet PIT (+18.5 pts)", is that gap for **the team bet on**, so it is always +4.0 or more.
- **Why a bet can be on the team the model doesn't pick** (marked †): the model can think a team
  will probably lose but still rate it much higher than the market does. Example: the model gives an
  underdog 39% and Vegas gives it 16%. The model still picks the favorite to win, but the underdog is
  the side it thinks is underpriced, so that is where the bet goes. On those rows the Gap column
  (for the picked team) is 4 points or more *below* zero, and the bet cell shows the same gap from
  the bet team's side, with the sign flipped (e.g. Gap -23.0, "Bet MIA † (+23.0 pts)").
- **If the bet wins**: profit on a $1 bet at the odds when the prediction was logged. A losing bet
  loses the $1. A tie refunds it.
- **Champion vs challenger**: the **champion** model's bets are the official record. The
  **challenger** (the same model with a small correction for over-rating home teams) is tracked for
  comparison only; it appears below only where it disagrees with the champion.


## This week

- **14 games**, **11 official paper bets** (3 on the team the model does not pick to win, marked †).

## Official picks (champion model)

| Game | Kickoff (Pacific) | Model picks | Model's chance | Vegas favorite | Vegas's chance | Gap (pts) | Paper bet | If the bet wins |
|---|---|---|---|---|---|---|---|---|
| SEA @ DEN | Thu 10/15 5:15 PM | DEN | 54% | SEA by 2.5 | 44% | +10.4 | Bet DEN (+10.4 pts) | +$1.20 per $1 risked |
| HOU @ JAX | Sun 10/18 6:30 AM | JAX | 62% | JAX by 2.5 | 56% | +5.5 | Bet JAX (+5.5 pts) | +$0.70 per $1 risked |
| BAL @ CLE | Sun 10/18 10:00 AM | BAL | 72% | BAL by 5.5 | 70% | +2.1 | No bet | – |
| CAR @ PHI | Sun 10/18 10:00 AM | PHI | 59% | CAR by 1.5 | 50% | +8.7 | Bet PHI (+8.7 pts) | +$0.91 per $1 risked |
| CHI @ ATL | Sun 10/18 10:00 AM | CHI | 52% | CHI by 2.5 | 58% | -6.3 | Bet ATL † (+6.3 pts) | +$1.30 per $1 risked |
| NO @ NYG | Sun 10/18 10:00 AM | NYG | 58% | NO by 1.5 | 48% | +9.6 | Bet NYG (+9.6 pts) | +$0.98 per $1 risked |
| NYJ @ NE | Sun 10/18 10:00 AM | NE | 85% | NE by 7.5 | 74% | +10.8 | Bet NE (+10.8 pts) | +$0.29 per $1 risked |
| PIT @ TB | Sun 10/18 10:00 AM | TB | 53% | PIT by 2.5 | 44% | +9.7 | Bet TB (+9.7 pts) | +$1.20 per $1 risked |
| TEN @ IND | Sun 10/18 10:00 AM | IND | 74% | IND by 7 | 73% | +0.5 | No bet | – |
| ARI @ LA | Sun 10/18 1:05 PM | LA | 79% | LA by 12.5 | 86% | -7.2 | Bet ARI † (+7.2 pts) | +$5.75 per $1 risked |
| BUF @ LV | Sun 10/18 1:25 PM | BUF | 75% | BUF by 5.5 | 69% | +6.0 | Bet BUF (+6.0 pts) | +$0.39 per $1 risked |
| LAC @ KC | Sun 10/18 1:25 PM | KC | 70% | KC by 8.5 | 80% | -9.6 | Bet LAC † (+9.6 pts) | +$3.80 per $1 risked |
| DAL @ GB | Sun 10/18 5:20 PM | GB | 65% | GB by 1.5 | 53% | +12.2 | Bet GB (+12.2 pts) | +$0.82 per $1 risked |
| WAS @ SF | Mon 10/19 5:15 PM | SF | 82% | SF by 7.5 | 78% | +3.9 | No bet | – |

† Bet on the team the model does not pick to win; see "Why a bet can be on the team the model doesn't pick" above.

## Challenger -- where it differs (comparison only, not the official record)

The challenger agreed with the champion on 13 of 14 games.

| Game | Champion: pick / bet | Challenger picks | Challenger's chance | Vegas's chance | Gap (pts) | Challenger's bet (hypothetical) |
|---|---|---|---|---|---|---|
| HOU @ JAX | JAX / Bet JAX (+5.5 pts) | JAX | 60% | 56% | +3.3 | No bet |

## The official record

The official, write-once record is [`week_06_predictions.csv`](../../data/live/2026/predictions/week_06_predictions.csv). It holds the full proof columns this page leaves out: exact probabilities, model file fingerprints (sha256), the betting-odds snapshot file and its pull time, both moneylines, and the prediction timestamp. Season totals and statistics: `docs/live/SCORECARD_2026.md`. Rules: `docs/PHASE10_PREREG.md`.
