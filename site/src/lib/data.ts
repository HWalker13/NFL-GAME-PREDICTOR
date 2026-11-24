/*
  Build-time only. Reads ../exports/site_data.json (relative to site/, which is the working
  directory for npm scripts locally and on Vercel with Root Directory = site/), validates it, and
  reduces it to what the pages render. NFL_SITE_DATA overrides the path. No side effects here, so
  tests can import it; the build's singleton lives in site.ts.
*/
import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseSiteData, type Game, type ScheduleGame, type SiteData } from './schema.ts';
import { buildCard, STAR_TIERS, tierOf, TIERS, type Card, type Grade, type StarTier, type Tier } from './card.ts';
import { WEEKS, currentWeek, weekBanner, weekInfo, type WeekInfo } from './season.ts';

export function dataPath(): string {
  return process.env['NFL_SITE_DATA'] ?? resolve(process.cwd(), '../exports/site_data.json');
}

export function loadSiteData(path = dataPath()): SiteData {
  if (!existsSync(path)) {
    throw new Error(
      `site_data.json not found at ${path}. Run the build from site/ (Vercel: Root Directory = site/, ` +
        `with files outside the root directory included), or set NFL_SITE_DATA.`,
    );
  }
  return parseSiteData(JSON.parse(readFileSync(path, 'utf8')));
}

export interface WeekPage {
  info: WeekInfo;
  banner: string | null;
  cards: Card[];
  /**
   * Graded picks and value calls (★/★★) this week; null when nothing this week is graded yet.
   * valueCalls is null when the week has no graded value call.
   */
  summary: { picks: Tally; valueCalls: Tally | null } | null;
}

export interface Tally {
  w: number;
  l: number;
  push: number;
}

/** Paper bets tallied from the cards (the site's own count, used to cross-check the export). */
export interface BetStats extends Tally {
  /** Paper bets on the cards, open ones included. */
  logged: number;
  /** Sum of Vegas's no-vig probability for the bet side over graded bets, pushes excluded (4 dp inputs). */
  vegasSum: number;
}

const tally = (grades: Grade[]): Tally => ({
  w: grades.filter((g) => g === 'win').length,
  l: grades.filter((g) => g === 'loss').length,
  push: grades.filter((g) => g === 'push').length,
});

/** Record and "Vegas expected" sum from the cards' paper bets; `tiers` limits it. Starred tiers = value calls. */
export function betStats(cards: readonly Card[], tiers?: readonly Tier[]): BetStats {
  const bets = cards.flatMap((c) => (c.paperBet && (!tiers || (c.paperBet.tier !== null && tiers.includes(c.paperBet.tier))) ? [c.paperBet] : []));
  const decided = bets.filter((v) => v.grade === 'win' || v.grade === 'loss');
  return {
    ...tally(bets.map((v) => v.grade)),
    logged: bets.length,
    vegasSum: decided.reduce((a, v) => a + v.vegasProb, 0),
  };
}

/** One value tier as the header shows it (numbers from the export's value_calls). */
export interface TierLine {
  tier: StarTier;
  /** Graded calls, pushes excluded: the calls WOE is over. */
  n: number;
  woe: number;
  woePerCall: number | null;
}

/** Value calls, ★/★★ only (Amendment 8). WOE numbers are the exporter's. */
export interface ValueRecord extends Tally {
  won: number;
  vegasExpected: number;
  woe: number;
  /** ★, ★★, in that order. */
  tiers: TierLine[];
}

/**
 * Fails the build unless the export's Amendment 6-8 numbers agree with the cards: for all paper bets,
 * the non-starred ones, the value calls and each star tier, counts and W-L-push equal the export's;
 * the site's own Vegas-expected sums (from 4 dp probabilities) are within rounding; ★ + ★★ = value
 * calls, and value calls + non-starred = the official record, pushes included; the paper-bet record
 * equals official_bets; the picks-vs-Vegas counts equal a recount from the games; the toss-up /
 * confident pick records equal prediction_quality_by_confidence; and each per-game flag agrees with
 * a recomputation from the 4 dp probabilities wherever that recomputation is clear of the cutoffs.
 */
export function crossCheckAmendments(d: SiteData, cards: readonly Card[]): void {
  const errors: string[] = [];
  const L = d.live_2026;
  const vc = L.value_calls;
  const pb = L.paper_bets;
  const all = pb.wins_over_expected;
  const non = pb.non_starred;
  const groups: [string, readonly Tier[], typeof all][] = [
    ['paper_bets', TIERS, all],
    ['non_starred', ['lean'], non],
    ['value_calls', STAR_TIERS, vc],
    ['one_star', ['one_star'], vc.tiers.one_star],
    ['two_star', ['two_star'], vc.tiers.two_star],
  ];
  for (const [key, tiers, e] of groups) {
    const site = betStats(cards, tiers);
    if (site.logged !== e.logged || site.w !== e.wins || site.l !== e.losses || site.push !== e.pushes) {
      errors.push(`${key}: site ${site.logged} logged ${site.w}-${site.l}-${site.push}, export ${e.logged} logged ${e.wins}-${e.losses}-${e.pushes}`);
    }
    if (e.n !== e.wins + e.losses || e.won !== e.wins) errors.push(`${key}: export n/won inconsistent with its record`);
    if (Math.abs(site.vegasSum - e.vegas_expected) > e.n * 1e-4 + 2e-4) {
      errors.push(`${key}: site Vegas expected ${site.vegasSum.toFixed(4)} vs export ${e.vegas_expected}`);
    }
  }
  const { one_star, two_star } = vc.tiers;
  for (const f of ['logged', 'graded', 'wins', 'losses', 'pushes', 'n', 'won'] as const) {
    if (one_star[f] + two_star[f] !== vc[f]) errors.push(`star tiers' ${f} do not sum to value_calls`);
    if (vc[f] + non[f] !== all[f]) errors.push(`value calls + non-starred ${f} != all paper bets`);
  }
  for (const f of ['vegas_expected', 'woe'] as const) {
    if (Math.abs(one_star[f] + two_star[f] - vc[f]) > 2.5e-4) errors.push(`star tiers' ${f} do not sum to value_calls`);
    if (Math.abs(vc[f] + non[f] - all[f]) > 2.5e-4) errors.push(`value calls + non-starred ${f} != all paper bets`);
  }
  const o = L.official_bets;
  const recs = [[pb.wins, pb.losses, pb.pushes], [all.wins, all.losses, all.pushes], [vc.wins + non.wins, vc.losses + non.losses, vc.pushes + non.pushes]];
  if (recs.some((r) => r.join('-') !== `${o.wins}-${o.losses}-${o.voids}`)) {
    errors.push(`paper-bet records ${recs.map((r) => r.join('-')).join(', ')} != official ${o.wins}-${o.losses}-${o.voids}`);
  }
  const pv = L.picks_vs_vegas;
  const decided = d.games.filter((g) => g.evaluation === 'live_2026' && g.pick_correct !== null);
  const diff = decided.filter((g) => g.vegas_favorite !== 'unresolved' && g.vegas_favorite !== g.model_pick);
  const right = diff.filter((g) => g.pick_correct).length;
  const dw = pv.different_winner;
  if (decided.length !== pv.games || diff.length !== dw.games || right !== dw.wins || diff.length - right !== dw.losses) {
    errors.push(`picks_vs_vegas: ${right}-${diff.length - right} of ${diff.length} in the games, export ${dw.wins}-${dw.losses} of ${dw.games}`);
  }
  const q = L.prediction_quality;
  if (pv.model.k !== q.champion.accuracy.k || pv.model.n !== q.champion.accuracy.n || pv.vegas_favorite.k !== q.vegas_favorite.accuracy.k || pv.vegas_favorite.n !== q.vegas_favorite.accuracy.n) {
    errors.push('picks_vs_vegas accuracies != prediction_quality');
  }
  const conf = L.prediction_quality_by_confidence;
  const withPick = cards.filter((c) => c.pick);
  for (const [key, want] of [['confident', false], ['toss_up', true]] as const) {
    const t = tally(withPick.filter((c) => c.tossUp === want).map((c) => c.pick!.grade));
    const n = withPick.filter((c) => c.tossUp === want).length;
    const e = conf[key];
    if (t.w !== e.accuracy.k || t.w + t.l !== e.accuracy.n || n !== e.games_logged) {
      errors.push(`${key} picks: site ${t.w}/${t.w + t.l} of ${n}, export ${e.accuracy.k}/${e.accuracy.n} of ${e.games_logged}`);
    }
  }
  for (const c of withPick) {
    const fav = Math.max(c.probs!.modelHome, 1 - c.probs!.modelHome);
    if (Math.abs(fav - L.toss_up_threshold) > 1e-4 && c.tossUp !== fav <= L.toss_up_threshold) {
      errors.push(`${c.gameId}: toss_up ${c.tossUp} but the model favorite is ${fav}`);
    }
    const v = c.paperBet;
    if (v) {
      const edge = v.modelProb - v.vegasProb;
      const clear = [L.value_tier_one_star_edge, L.value_tier_two_star_edge].every((x) => Math.abs(edge - x) > 2e-4);
      if (v.tier === null) errors.push(`${c.gameId}: paper bet without a tier`);
      else if (clear && v.tier !== tierOf(edge, L.value_tier_one_star_edge, L.value_tier_two_star_edge)) {
        errors.push(`${c.gameId}: tier ${v.tier} but the edge is ${edge.toFixed(4)}`);
      }
    }
  }
  if (errors.length) throw new Error(`tiers / paper bets / toss-ups disagree with the exporter:\n  ${errors.join('\n  ')}`);
}

/**
 * Fails the build unless the site's paper bets are the exporter's official paper record: same
 * games, same side, and the same graded W-L-push as live_2026.official_bets.
 */
export function crossCheckPaperBets(d: SiteData, cards: ReadonlyMap<string, Card>, stats: Tally): void {
  const errors: string[] = [];
  for (const g of d.games) {
    if (g.evaluation !== 'live_2026') continue;
    const v = cards.get(g.game_id)?.paperBet ?? null;
    const want = g.bet ? g.bet_side : null;
    if ((v?.side ?? null) !== want) errors.push(`${g.game_id}: site paper bet ${v?.side ?? 'none'}, export ${want ?? 'none'}`);
  }
  const o = d.live_2026.official_bets;
  if (stats.w !== o.wins || stats.l !== o.losses || stats.push !== o.voids) {
    errors.push(`paper-bet record ${stats.w}-${stats.l}-${stats.push} != official ${o.wins}-${o.losses}-${o.voids}`);
  }
  if (errors.length) throw new Error(`paper bets disagree with the exporter's official record:\n  ${errors.join('\n  ')}`);
}

export interface Site {
  current: number;
  weeks: WeekInfo[];
  pages: Map<number, WeekPage>;
  record: {
    picks: {
      w: number;
      l: number;
      accuracy: number;
      vegasFavorite: number;
      /** Pick accuracy on confident games and on toss-ups; null when none is graded. */
      confident: number | null;
      tossUps: number | null;
    };
    valueCalls: ValueRecord;
  };
  /** The "vs Vegas" page's extra numbers, all from the export. */
  vsVegas: {
    games: number;
    sameWinner: number;
    differentWinner: { games: number; wins: number; losses: number };
    paper: Tally & {
      pending: number;
      graded: number;
      units: number;
      roi: number | null;
      roiCi: readonly [number | null, number | null];
      clv: { n: number; mean: number | null; ci: readonly [number | null, number | null] };
      verdict: { status: string; final: boolean; so_far: string };
    };
  };
  generatedAt: string;
  asOf: string;
  commit: string;
}

export function buildSite(d: SiteData): Site {
  const L = d.live_2026;
  const now = new Date(L.as_of_utc); // the snapshot's pull time: same clock as the scores
  const live = new Map<string, Game>(
    d.games.filter((g) => g.evaluation === 'live_2026').map((g) => [g.game_id, g]),
  );
  const byWeek = new Map<number, ScheduleGame[]>(WEEKS.map((w) => [w, []]));
  for (const g of d.schedule_2026) byWeek.get(g.week)?.push(g);

  const weeks = WEEKS.map((w) => weekInfo(w, L.first_official_week, L.weeks, d.schedule_2026));
  const pages = new Map<number, WeekPage>();
  for (const info of weeks) {
    const games = byWeek.get(info.week) ?? [];
    const cards = games.map((g) => buildCard(g, live.get(g.game_id), L.edge_threshold, now));
    const picks = tally(cards.flatMap((c) => (c.pick ? [c.pick.grade] : [])));
    const starred = tally(cards.flatMap((c) => (c.valueCall ? [c.valueCall.grade] : [])));
    const graded = picks.w + picks.l + picks.push > 0;
    pages.set(info.week, {
      info,
      banner: weekBanner(info, L.first_official_week),
      cards,
      summary: graded ? { picks, valueCalls: starred.w + starred.l + starred.push > 0 ? starred : null } : null,
    });
  }
  const allCards = [...pages.values()].flatMap((p) => p.cards);
  crossCheckPaperBets(d, new Map(allCards.map((c) => [c.gameId, c])), betStats(allCards));
  crossCheckAmendments(d, allCards);
  const vc = L.value_calls;
  const pb = L.paper_bets;
  const pv = L.picks_vs_vegas;
  const conf = L.prediction_quality_by_confidence;

  const champ = L.prediction_quality.champion.accuracy;
  const fav = L.prediction_quality.vegas_favorite.accuracy;
  if (champ.value === null || fav.value === null) throw new Error('live_2026 accuracy has no value yet');

  return {
    current: currentWeek(d.schedule_2026),
    weeks,
    pages,
    record: {
      picks: {
        w: champ.k,
        l: champ.n - champ.k,
        accuracy: champ.value,
        vegasFavorite: fav.value,
        confident: conf.confident.accuracy.value,
        tossUps: conf.toss_up.accuracy.value,
      },
      valueCalls: {
        w: vc.wins,
        l: vc.losses,
        push: vc.pushes,
        won: vc.won,
        vegasExpected: vc.vegas_expected,
        woe: vc.woe,
        tiers: STAR_TIERS.map((t) => {
          const e = vc.tiers[t];
          return { tier: t, n: e.n, woe: e.woe, woePerCall: e.woe_per_call };
        }),
      },
    },
    vsVegas: {
      games: pv.games,
      sameWinner: pv.same_winner,
      differentWinner: pv.different_winner,
      paper: {
        w: pb.wins,
        l: pb.losses,
        push: pb.pushes,
        pending: pb.pending,
        graded: pb.graded,
        units: pb.units,
        roi: pb.roi,
        roiCi: pb.roi_ci95,
        clv: { n: pb.clv.n, mean: pb.clv.mean, ci: pb.clv.ci95 },
        verdict: pb.verdict,
      },
    },
    generatedAt: d.generated_at,
    asOf: L.as_of_utc,
    commit: d.source_commit.hash,
  };
}
