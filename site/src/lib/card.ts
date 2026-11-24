/* Pure view model for one game card. Every card state comes from here; GameCard.astro only renders. */
import type { Game, ScheduleGame } from './schema.ts';
import { isFinal } from './season.ts';
import { primetime, showsAsFifty, type Primetime } from './format.ts';

/** Top-line status: Final / Not played yet / Score pending. Grading lives on the chips. */
export type CardStatus = 'final' | 'upcoming' | 'score-pending';

/** A chip's grade. 'open' = not final yet; 'push' = the game ended tied. */
export type Grade = 'win' | 'loss' | 'push' | 'open';

export type Side = 'home' | 'away';

/**
 * A paper bet's tier (Amendment 7), on its own edge: 'lean' = 4 to 11 points, 'one_star' = ★, 11 to
 * 16 points, 'two_star' = ★★, 16 points or more. Only the starred tiers are value calls (Amendment
 * 8); on the cards every other game is an ordinary pick, and 'lean' is never shown as a word.
 */
export type Tier = 'lean' | 'one_star' | 'two_star';
export type StarTier = Exclude<Tier, 'lean'>;

/** Display order, lowest tier first. */
export const TIERS: readonly Tier[] = ['lean', 'one_star', 'two_star'];
export const STAR_TIERS: readonly StarTier[] = ['one_star', 'two_star'];

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
  /** The starred (★/★★) paper bet, if any: the only thing the site calls a value call. */
  valueCall: (Call & { vegasProb: number; tier: StarTier }) | null;
  /**
   * The team whose model probability beats Vegas's by at least the 4-point threshold, if any. Used
   * only to cross-check the export's paper-bet record; cards never render it. `tier` is the
   * exporter's (decided at full precision); null only when the export carries none.
   */
  paperBet: (Call & { vegasProb: number; tier: Tier | null }) | null;
  /** The model's two win chances are within 10 points (the exporter's flag). False without a pick. */
  tossUp: boolean;
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
  let paperBet: Card['paperBet'] = null;
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
      paperBet = {
        side: vs,
        team: teamOf(vs),
        grade: gradeSide(vs, w),
        modelProb: forSide(vs, pHome),
        vegasProb: forSide(vs, vHome),
        tier: live.value_tier ?? null,
      };
      const { tier } = paperBet;
      if (tier === 'one_star' || tier === 'two_star') valueCall = { ...paperBet, tier };
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
    paperBet,
    tossUp: Boolean(live && pickSide && live.toss_up),
    probs,
    decimals,
  };
}

/** Visible stars for a value call. */
export const STARS: Record<StarTier, string> = { one_star: '★', two_star: '★★' };

/** Accessible name for a tier (the stars are hidden from screen readers). */
export const tierLabel = (t: StarTier): string => ({ one_star: 'one-star value call', two_star: 'two-star value call' })[t];

/**
 * The tier for an edge, given the cutoffs; same 10-decimal comparison as the exporter. Only for the
 * build's cross-check against 4 dp probabilities. The exporter's full-precision tier is what renders.
 */
export function tierOf(edge: number, oneStar: number, twoStar: number): Tier {
  if (meetsThreshold(edge, twoStar)) return 'two_star';
  return meetsThreshold(edge, oneStar) ? 'one_star' : 'lean';
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
  // Every other game, paper bet or not, reads the same neutral way (Amendment 8): describe the side the model likes a bit more than Vegas does.
  const side: Side = modelHome >= vegasHome ? 'home' : 'away';
  const team = side === 'home' ? c.home.abbr : c.away.abbr;
  const m = side === 'home' ? modelHome : 1 - modelHome;
  const v = side === 'home' ? vegasHome : 1 - vegasHome;
  return `Model gives ${team} ${p(m)}, Vegas ${p(v)}.`;
}
