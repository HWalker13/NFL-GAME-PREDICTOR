# 2026 Week 7 -- model picks (before kickoff, no results yet)

_Predictions logged Wed 10/21 2:37 PM Pacific, against betting odds pulled Wed 10/21 2:34 PM Pacific. This page is regenerated from the official record and is not itself the record. Generated 2026-10-21T21:38:11Z._

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

- **14 games**, **9 official paper bets** (2 on the team the model does not pick to win, marked †).

## Official picks (champion model)

| Game | Kickoff (Pacific) | Model picks | Model's chance | Vegas favorite | Vegas's chance | Gap (pts) | Paper bet | If the bet wins |
|---|---|---|---|---|---|---|---|---|
| NE @ CHI | Thu 10/22 5:15 PM | NE | 53% | CHI by 3 | 42% | +11.6 | Bet NE (+11.6 pts) | +$1.30 per $1 risked |
| PIT @ NO | Sun 10/25 6:30 AM | PIT | 56% | PIT by 3.5 | 62% | -5.9 | Bet NO † (+5.9 pts) | +$1.54 per $1 risked |
| CIN @ BAL | Sun 10/25 10:00 AM | BAL | 55% | CIN by 1.5 | 48% | +7.0 | Bet BAL (+7.0 pts) | +$1.00 per $1 risked |
| CLE @ TEN | Sun 10/25 10:00 AM | CLE | 55% | CLE by 1.5 | 53% | +1.3 | No bet | – |
| IND @ MIN | Sun 10/25 10:00 AM | MIN | 77% | MIN by 6.5 | 71% | +6.9 | Bet MIN (+6.9 pts) | +$0.36 per $1 risked |
| MIA @ NYJ | Sun 10/25 10:00 AM | MIA | 58% | NYJ by 4.5 | 33% | +25.3 | Bet MIA (+25.3 pts) | +$1.90 per $1 risked |
| NYG @ HOU | Sun 10/25 10:00 AM | HOU | 77% | HOU by 10.5 | 80% | -3.2 | No bet | – |
| SF @ ATL | Sun 10/25 10:00 AM | SF | 72% | SF by 3 | 58% | +13.6 | Bet SF (+13.6 pts) | +$0.65 per $1 risked |
| TB @ CAR | Sun 10/25 10:00 AM | CAR | 63% | CAR by 3 | 60% | +3.4 | No bet | – |
| DEN @ ARI | Sun 10/25 1:05 PM | DEN | 67% | DEN by 4.5 | 68% | -0.9 | No bet | – |
| GB @ DET | Sun 10/25 1:25 PM | DET | 54% | DET by 3 | 60% | -5.7 | Bet GB † (+5.7 pts) | +$1.40 per $1 risked |
| LA @ LV | Sun 10/25 1:25 PM | LA | 67% | LA by 4.5 | 66% | +1.3 | No bet | – |
| KC @ SEA | Sun 10/25 5:20 PM | SEA | 64% | SEA by 2.5 | 57% | +6.9 | Bet SEA (+6.9 pts) | +$0.68 per $1 risked |
| DAL @ PHI | Mon 10/26 5:15 PM | PHI | 56% | DAL by 4.5 | 34% | +22.3 | Bet PHI (+22.3 pts) | +$1.85 per $1 risked |

† Bet on the team the model does not pick to win; see "Why a bet can be on the team the model doesn't pick" above.

## Challenger -- where it differs (comparison only, not the official record)

The challenger agreed with the champion on 12 of 14 games.

| Game | Champion: pick / bet | Challenger picks | Challenger's chance | Vegas's chance | Gap (pts) | Challenger's bet (hypothetical) |
|---|---|---|---|---|---|---|
| IND @ MIN | MIN / Bet MIN (+6.9 pts) | MIN | 74% | 71% | +3.5 | No bet |
| NYG @ HOU | HOU / No bet | HOU | 74% | 80% | -6.5 | Bet NYG † (+6.5 pts) |

## The official record

The official, write-once record is [`week_07_predictions.csv`](../../data/live/2026/predictions/week_07_predictions.csv). It holds the full proof columns this page leaves out: exact probabilities, model file fingerprints (sha256), the betting-odds snapshot file and its pull time, both moneylines, and the prediction timestamp. Season totals and statistics: `docs/live/SCORECARD_2026.md`. Rules: `docs/PHASE10_PREREG.md`.
