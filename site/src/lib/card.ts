/* Pure view model for one game card. Every card state comes from here; GameCard.astro only renders. */
import type { Game, ScheduleGame } from './schema.ts';
import { isFinal } from './season.ts';
import { showsAsFifty } from './format.ts';

export type CardStatus = 'correct' | 'wrong' | 'tie' | 'final' | 'upcoming' | 'score-pending';

export interface TeamRow {
  abbr: string;
  score: number | null;
  picked: boolean;
}

export interface Card {
  gameId: string;
  kickoffUtc: string;
  status: CardStatus;
  away: TeamRow;
  home: TeamRow;
  /** Present only when the model logged a pick. */
  pick: {
    team: string;
    modelProb: number;
    /** Vig-free market probability for the same team; null when there was no moneyline. */
    vegasProb: number | null;
  } | null;
  /** The "Model rates X higher than Vegas does" line, when the gap meets the threshold. */
  flag: { team: string; modelProb: number; vegasProb: number } | null;
  /**
   * Decimals for every percentage on the card: 1 when any of them would round to 50%, so a pick
   * never reads as a coin flip (e.g. 50.4% vs 49.6%); otherwise whole numbers.
   */
  decimals: 0 | 1;
}

/**
 * Same comparison as the exporter's rule (scripts/live/common.py edge_qualifies): edge >= threshold
 * after rounding to 10 decimals, so an edge that is exactly the threshold on paper isn't lost to
 * float noise.
 */
export const meetsThreshold = (edge: number, threshold: number): boolean =>
  Math.round(edge * 1e10) / 1e10 >= threshold;

export function buildCard(s: ScheduleGame, live: Game | undefined, threshold: number, now: Date): Card {
  const pickSide = live?.model_pick ?? null;
  const away: TeamRow = { abbr: s.away, score: s.away_score, picked: pickSide === 'away' };
  const home: TeamRow = { abbr: s.home, score: s.home_score, picked: pickSide === 'home' };

  let status: CardStatus;
  if (isFinal(s)) {
    const hs = s.home_score as number;
    const as = s.away_score as number;
    if (!pickSide) status = 'final';
    else if (hs === as) status = 'tie';
    else status = (hs > as ? 'home' : 'away') === pickSide ? 'correct' : 'wrong';
  } else {
    status = Date.parse(s.kickoff_utc) > now.getTime() ? 'upcoming' : 'score-pending';
  }

  let pick: Card['pick'] = null;
  let flag: Card['flag'] = null;
  if (live && pickSide) {
    const pHome = live.model_home_prob;
    const vHome = live.vegas_implied_home_prob;
    pick = {
      team: pickSide === 'home' ? s.home : s.away,
      modelProb: pickSide === 'home' ? pHome : 1 - pHome,
      vegasProb: vHome === null ? null : pickSide === 'home' ? vHome : 1 - vHome,
    };
    if (vHome !== null) {
      const edgeHome = pHome - vHome;
      if (meetsThreshold(edgeHome, threshold)) flag = { team: s.home, modelProb: pHome, vegasProb: vHome };
      else if (meetsThreshold(-edgeHome, threshold)) flag = { team: s.away, modelProb: 1 - pHome, vegasProb: 1 - vHome };
    }
  }

  const shown = [pick?.modelProb, pick?.vegasProb, flag?.modelProb, flag?.vegasProb].filter(
    (p): p is number => p !== undefined && p !== null,
  );
  const decimals = shown.some(showsAsFifty) ? 1 : 0;

  return { gameId: s.game_id, kickoffUtc: s.kickoff_utc, status, away, home, pick, flag, decimals };
}
