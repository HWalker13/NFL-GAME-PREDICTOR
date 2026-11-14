/* Pure view model for one game card. Every card state comes from here; GameCard.astro only renders. */
import type { Game, ScheduleGame } from './schema.ts';
import { isFinal } from './season.ts';
import { primetime, showsAsFifty, type Primetime } from './format.ts';

/** Top-line status: Final / Not played yet / Score pending. Grading lives on the chips. */
export type CardStatus = 'final' | 'upcoming' | 'score-pending';

/** A chip's grade. 'open' = not final yet; 'push' = the game ended tied. */
export type Grade = 'win' | 'loss' | 'push' | 'open';

export type Side = 'home' | 'away';

export interface TeamRow {
  abbr: string;
  score: number | null;
  picked: boolean;
}

export interface Call {
  side: Side;
  team: string;
  grade: Grade;
  modelProb: number;
  /** Vig-free market probability for the same team; null when there was no moneyline. */
  vegasProb: number | null;
}

export interface Card {
  gameId: string;
  kickoffUtc: string;
  primetime: Primetime | null;
  status: CardStatus;
  away: TeamRow;
  home: TeamRow;
  /** The team the model gives > 50%. Present only when the model logged a prediction. */
  pick: Call | null;
  /** The team whose model probability beats Vegas's by at least the threshold, if any. */
  valueCall: (Call & { vegasProb: number }) | null;
  /** Both sides' probabilities, for the details bars. Null when there's no prediction. */
  probs: { modelHome: number; vegasHome: number | null } | null;
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

/** The winning side of a final game, 'tie', or null when not final. */
export function winner(s: ScheduleGame): Side | 'tie' | null {
  if (!isFinal(s)) return null;
  const hs = s.home_score as number;
  const as = s.away_score as number;
  return hs === as ? 'tie' : hs > as ? 'home' : 'away';
}

/** ✓ if the side won, ✗ if it lost, push on a tie, open until final. */
export function gradeSide(side: Side, w: Side | 'tie' | null): Grade {
  if (w === null) return 'open';
  if (w === 'tie') return 'push';
  return w === side ? 'win' : 'loss';
}

/** Which side (if any) the model rates at least `threshold` above Vegas. */
export function valueSide(modelHome: number, vegasHome: number | null, threshold: number): Side | null {
  if (vegasHome === null) return null;
  const edgeHome = modelHome - vegasHome;
  if (meetsThreshold(edgeHome, threshold)) return 'home';
  if (meetsThreshold(-edgeHome, threshold)) return 'away';
  return null;
}

export function buildCard(s: ScheduleGame, live: Game | undefined, threshold: number, now: Date): Card {
  const pickSide = live?.model_pick ?? null;
  const away: TeamRow = { abbr: s.away, score: s.away_score, picked: pickSide === 'away' };
  const home: TeamRow = { abbr: s.home, score: s.home_score, picked: pickSide === 'home' };
  const w = winner(s);

  const status: CardStatus = isFinal(s)
    ? 'final'
    : Date.parse(s.kickoff_utc) > now.getTime()
      ? 'upcoming'
      : 'score-pending';

  const teamOf = (side: Side) => (side === 'home' ? s.home : s.away);
  const forSide = (side: Side, pHome: number) => (side === 'home' ? pHome : 1 - pHome);

  let pick: Card['pick'] = null;
  let valueCall: Card['valueCall'] = null;
  let probs: Card['probs'] = null;
  if (live && pickSide) {
    const pHome = live.model_home_prob;
    const vHome = live.vegas_implied_home_prob;
    probs = { modelHome: pHome, vegasHome: vHome };
    pick = {
      side: pickSide,
      team: teamOf(pickSide),
      grade: gradeSide(pickSide, w),
      modelProb: forSide(pickSide, pHome),
      vegasProb: vHome === null ? null : forSide(pickSide, vHome),
    };
    const vs = valueSide(pHome, vHome, threshold);
    if (vs && vHome !== null) {
      valueCall = {
        side: vs,
        team: teamOf(vs),
        grade: gradeSide(vs, w),
        modelProb: forSide(vs, pHome),
        vegasProb: forSide(vs, vHome),
      };
    }
  }

  const shown = probs ? [probs.modelHome, 1 - probs.modelHome] : [];
  if (probs?.vegasHome != null) shown.push(probs.vegasHome, 1 - probs.vegasHome);
  const decimals = shown.some(showsAsFifty) ? 1 : 0;

  return {
    gameId: s.game_id,
    kickoffUtc: s.kickoff_utc,
    primetime: primetime(s.kickoff_utc),
    status,
    away,
    home,
    pick,
    valueCall,
    probs,
    decimals,
  };
}

/** The plain-English line under the details bars. */
export function detailsSentence(c: Card, p: (x: number) => string): string | null {
  if (!c.probs || !c.pick) return null;
  const { modelHome, vegasHome } = c.probs;
  if (vegasHome === null) return `Model gives ${c.pick.team} ${p(c.pick.modelProb)}. No Vegas odds were saved for this game.`;
  if (c.valueCall) {
    const v = c.valueCall;
    return `Model gives ${v.team} ${p(v.modelProb)}, Vegas ${p(v.vegasProb)} — the model thinks ${v.team} is underrated.`;
  }
  // Under the threshold: describe the side the model likes a bit more than Vegas does.
  const side: Side = modelHome >= vegasHome ? 'home' : 'away';
  const team = side === 'home' ? c.home.abbr : c.away.abbr;
  const m = side === 'home' ? modelHome : 1 - modelHome;
  const v = side === 'home' ? vegasHome : 1 - vegasHome;
  return `Model gives ${team} ${p(m)}, Vegas ${p(v)} — close to the market.`;
}
