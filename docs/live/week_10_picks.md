# 2026 Week 10 -- model picks (before kickoff, no results yet)

_Predictions logged Thu 11/12 1:09 PM Pacific, against betting odds pulled Thu 11/12 1:09 PM Pacific. This page is regenerated from the official record and is not itself the record. Generated 2026-11-12T21:10:27Z._

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

- **14 games**, **5 official paper bets** (3 on the team the model does not pick to win, marked †).

## Official picks (champion model)

| Game | Kickoff (Pacific) | Model picks | Model's chance | Vegas favorite | Vegas's chance | Gap (pts) | Paper bet | If the bet wins |
|---|---|---|---|---|---|---|---|---|
| WAS @ NYG | Thu 11/12 5:15 PM | NYG | 58% | NYG by 2.5 | 54% | +3.4 | No bet | – |
| NE @ DET | Sun 11/15 6:30 AM | DET | 50% | DET by 2.5 | 58% | -7.9 | Bet NE † (+7.9 pts) | +$1.30 per $1 risked |
| BUF @ NYJ | Sun 11/15 10:00 AM | BUF | 85% | BUF by 8.5 | 78% | +7.0 | Bet BUF (+7.0 pts) | +$0.23 per $1 risked |
| CAR @ NO | Sun 11/15 10:00 AM | NO | 54% | NO by 2.5 | 54% | -0.6 | No bet | – |
| HOU @ CLE | Sun 11/15 10:00 AM | HOU | 64% | HOU by 4.5 | 66% | -2.1 | No bet | – |
| JAX @ TEN | Sun 11/15 10:00 AM | JAX | 82% | JAX by 10.5 | 82% | +0.3 | No bet | – |
| KC @ ATL | Sun 11/15 10:00 AM | KC | 62% | KC by 3 | 62% | +0.2 | No bet | – |
| MIA @ IND | Sun 11/15 10:00 AM | IND | 53% | IND by 3.5 | 64% | -11.8 | Bet MIA † (+11.8 pts) | +$1.70 per $1 risked |
| MIN @ GB | Sun 11/15 10:00 AM | GB | 71% | GB by 3 | 58% | +12.6 | Bet GB (+12.6 pts) | +$0.65 per $1 risked |
| LA @ ARI | Sun 11/15 1:05 PM | LA | 71% | LA by 9.5 | 80% | -9.0 | Bet ARI † (+9.0 pts) | +$3.90 per $1 risked |
| SEA @ LV | Sun 11/15 1:05 PM | SEA | 69% | SEA by 6.5 | 72% | -2.6 | No bet | – |
| SF @ DAL | Sun 11/15 1:25 PM | SF | 67% | SF by 3.5 | 64% | +2.7 | No bet | – |
| PIT @ CIN | Sun 11/15 5:20 PM | CIN | 62% | CIN by 3.5 | 62% | +0.2 | No bet | – |
| LAC @ BAL | Mon 11/16 5:15 PM | BAL | 67% | BAL by 4.5 | 67% | +0.6 | No bet | – |

† Bet on the team the model does not pick to win; see "Why a bet can be on the team the model doesn't pick" above.

## Challenger -- where it differs (comparison only, not the official record)

The challenger agreed with the champion on 13 of 14 games.

| Game | Champion: pick / bet | Challenger picks | Challenger's chance | Vegas's chance | Gap (pts) | Challenger's bet (hypothetical) |
|---|---|---|---|---|---|---|
| NE @ DET | DET / Bet NE † (+7.9 pts) | NE | 51% | 42% | +9.0 | Bet NE (+9.0 pts) |

## The official record

The official, write-once record is [`week_10_predictions.csv`](../../data/live/2026/predictions/week_10_predictions.csv). It holds the full proof columns this page leaves out: exact probabilities, model file fingerprints (sha256), the betting-odds snapshot file and its pull time, both moneylines, and the prediction timestamp. Season totals and statistics: `docs/live/SCORECARD_2026.md`. Rules: `docs/PHASE10_PREREG.md`.
