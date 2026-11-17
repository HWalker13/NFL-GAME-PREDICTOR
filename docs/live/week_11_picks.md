# 2026 Week 11 -- model picks (before kickoff, no results yet)

_Predictions logged Tue 11/17 1:24 PM Pacific, against betting odds pulled Tue 11/17 1:23 PM Pacific. This page is regenerated from the official record and is not itself the record. Generated 2026-11-17T21:24:18Z._

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

- **13 games**, **8 official paper bets** (6 on the team the model does not pick to win, marked †).

## Official picks (champion model)

| Game | Kickoff (Pacific) | Model picks | Model's chance | Vegas favorite | Vegas's chance | Gap (pts) | Paper bet | If the bet wins |
|---|---|---|---|---|---|---|---|---|
| IND @ HOU | Thu 11/19 5:15 PM | HOU | 77% | HOU by 12.5 | 84% | -7.3 | Bet IND † (+7.3 pts) | +$5.00 per $1 risked |
| ARI @ KC | Sun 11/22 10:00 AM | KC | 81% | KC by 16.5 | 90% | -9.0 | Bet ARI † (+9.0 pts) | +$8.50 per $1 risked |
| BAL @ CAR | Sun 11/22 10:00 AM | CAR | 52% | CAR by 2.5 | 56% | -4.6 | Bet BAL † (+4.6 pts) | +$1.20 per $1 risked |
| JAX @ NYG | Sun 11/22 10:00 AM | JAX | 67% | JAX by 6 | 70% | -2.3 | No bet | – |
| MIA @ BUF | Sun 11/22 10:00 AM | BUF | 83% | BUF by 16.5 | 90% | -6.7 | Bet MIA † (+6.7 pts) | +$8.50 per $1 risked |
| NO @ CHI | Sun 11/22 10:00 AM | CHI | 77% | CHI by 11.5 | 84% | -7.2 | Bet NO † (+7.2 pts) | +$5.00 per $1 risked |
| TB @ DET | Sun 11/22 10:00 AM | DET | 69% | DET by 6.5 | 70% | -0.6 | No bet | – |
| TEN @ DAL | Sun 11/22 10:00 AM | DAL | 75% | DAL by 8.5 | 76% | -1.2 | No bet | – |
| NYJ @ LAC | Sun 11/22 1:05 PM | LAC | 64% | LAC by 3.5 | 62% | +1.9 | No bet | – |
| LV @ DEN | Sun 11/22 1:25 PM | DEN | 78% | DEN by 10 | 81% | -3.6 | No bet | – |
| PIT @ PHI | Sun 11/22 1:25 PM | PHI | 52% | PIT by 4.5 | 34% | +17.2 | Bet PHI (+17.2 pts) | +$1.80 per $1 risked |
| MIN @ SF | Sun 11/22 5:20 PM | SF | 66% | SF by 3 | 59% | +6.4 | Bet SF (+6.4 pts) | +$0.62 per $1 risked |
| CIN @ WAS | Mon 11/23 5:15 PM | CIN | 63% | CIN by 6.5 | 71% | -8.0 | Bet WAS † (+8.0 pts) | +$2.35 per $1 risked |

† Bet on the team the model does not pick to win; see "Why a bet can be on the team the model doesn't pick" above.

## Challenger -- where it differs (comparison only, not the official record)

The challenger agreed with the champion on 10 of 13 games.

| Game | Champion: pick / bet | Challenger picks | Challenger's chance | Vegas's chance | Gap (pts) | Challenger's bet (hypothetical) |
|---|---|---|---|---|---|---|
| TEN @ DAL | DAL / No bet | DAL | 71% | 76% | -4.5 | Bet TEN † (+4.5 pts) |
| LV @ DEN | DEN / No bet | DEN | 74% | 81% | -6.9 | Bet LV † (+6.9 pts) |
| MIN @ SF | SF / Bet SF (+6.4 pts) | SF | 63% | 59% | +3.8 | No bet |

## The official record

The official, write-once record is [`week_11_predictions.csv`](../../data/live/2026/predictions/week_11_predictions.csv). It holds the full proof columns this page leaves out: exact probabilities, model file fingerprints (sha256), the betting-odds snapshot file and its pull time, both moneylines, and the prediction timestamp. Season totals and statistics: `docs/live/SCORECARD_2026.md`. Rules: `docs/PHASE10_PREREG.md`.
