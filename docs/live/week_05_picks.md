# 2026 Week 5 -- model picks (before kickoff, no results yet)

_Predictions logged Wed 10/7 11:40 AM Pacific, against betting odds pulled Wed 10/7 11:22 AM Pacific. This page is regenerated from the official record and is not itself the record. Generated 2026-10-07T18:41:11Z._

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

- **15 games**, **10 official paper bets** (6 on the team the model does not pick to win, marked †).

## Official picks (champion model)

| Game | Kickoff (Pacific) | Model picks | Model's chance | Vegas favorite | Vegas's chance | Gap (pts) | Paper bet | If the bet wins |
|---|---|---|---|---|---|---|---|---|
| TB @ DAL | Thu 10/8 5:15 PM | DAL | 62% | DAL by 8.5 | 79% | -17.4 | Bet TB † (+17.4 pts) | +$3.60 per $1 risked |
| PHI @ JAX | Sun 10/11 6:30 AM | JAX | 72% | JAX by 7 | 75% | -2.6 | No bet | – |
| CHI @ GB | Sun 10/11 10:00 AM | CHI | 52% | CHI by 2.5 | 57% | -5.5 | Bet GB † (+5.5 pts) | +$1.24 per $1 risked |
| CIN @ MIA | Sun 10/11 10:00 AM | CIN | 61% | CIN by 6.5 | 74% | -12.8 | Bet MIA † (+12.8 pts) | +$2.70 per $1 risked |
| CLE @ NYJ | Sun 10/11 10:00 AM | CLE | 62% | NYJ by 2.5 | 46% | +16.3 | Bet CLE (+16.3 pts) | +$1.10 per $1 risked |
| HOU @ TEN | Sun 10/11 10:00 AM | HOU | 74% | HOU by 7.5 | 76% | -1.8 | No bet | – |
| IND @ PIT | Sun 10/11 10:00 AM | PIT | 64% | PIT by 2.5 | 57% | +7.1 | Bet PIT (+7.1 pts) | +$0.68 per $1 risked |
| LV @ NE | Sun 10/11 10:00 AM | NE | 77% | NE by 3.5 | 64% | +13.2 | Bet NE (+13.2 pts) | +$0.51 per $1 risked |
| MIN @ NO | Sun 10/11 10:00 AM | MIN | 55% | MIN by 1.5 | 53% | +2.0 | No bet | – |
| NYG @ WAS | Sun 10/11 10:00 AM | WAS | 55% | WAS by 3.5 | 61% | -5.6 | Bet NYG † (+5.6 pts) | +$1.45 per $1 risked |
| DEN @ LAC | Sun 10/11 1:05 PM | DEN | 62% | DEN by 3.5 | 62% | -0.7 | No bet | – |
| DET @ ARI | Sun 10/11 1:25 PM | DET | 69% | DET by 5.5 | 69% | -0.2 | No bet | – |
| SF @ SEA | Sun 10/11 1:25 PM | SEA | 51% | SEA by 2.5 | 58% | -7.6 | Bet SF † (+7.6 pts) | +$1.30 per $1 risked |
| BAL @ ATL | Sun 10/11 5:20 PM | BAL | 60% | ATL by 3.5 | 38% | +21.4 | Bet BAL (+21.4 pts) | +$1.50 per $1 risked |
| BUF @ LA | Mon 10/12 5:15 PM | LA | 54% | LA by 3 | 59% | -5.4 | Bet BUF † (+5.4 pts) | +$1.36 per $1 risked |

† Bet on the team the model does not pick to win; see "Why a bet can be on the team the model doesn't pick" above.

## Challenger -- where it differs (comparison only, not the official record)

The challenger agreed with the champion on 13 of 15 games.

| Game | Champion: pick / bet | Challenger picks | Challenger's chance | Vegas's chance | Gap (pts) | Challenger's bet (hypothetical) |
|---|---|---|---|---|---|---|
| PHI @ JAX | JAX / No bet | JAX | 69% | 75% | -5.7 | Bet PHI † (+5.7 pts) |
| SF @ SEA | SEA / Bet SF † (+7.6 pts) | SF | 50% | 42% | +8.6 | Bet SF (+8.6 pts) |

## The official record

The official, write-once record is [`week_05_predictions.csv`](../../data/live/2026/predictions/week_05_predictions.csv). It holds the full proof columns this page leaves out: exact probabilities, model file fingerprints (sha256), the betting-odds snapshot file and its pull time, both moneylines, and the prediction timestamp. Season totals and statistics: `docs/live/SCORECARD_2026.md`. Rules: `docs/PHASE10_PREREG.md`.
