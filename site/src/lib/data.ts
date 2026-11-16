/*
  Build-time only. Reads ../exports/site_data.json (relative to site/, which is the working
  directory for npm scripts locally and on Vercel with Root Directory = site/), validates it, and
  reduces it to what the pages render. NFL_SITE_DATA overrides the path. No side effects here, so
  tests can import it; the build's singleton lives in site.ts.
*/
import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseSiteData, type Game, type ScheduleGame, type SiteData } from './schema.ts';
import { buildCard, type Card, type Grade, type Tier } from './card.ts';
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
  /** Graded picks and value calls this week; null when nothing this week is graded yet. */
  summary: { picks: Tally; valueCalls: Tally } | null;
}

export interface Tally {
  w: number;
  l: number;
  push: number;
}

/** Value calls tallied from the cards (the site's own count, used to cross-check the export). */
export interface ValueCallStats extends Tally {
  /** Value calls on the cards, open ones included. */
  calls: number;
  /** Sum of Vegas's no-vig probability for the call side over graded calls, pushes excluded (4 dp inputs). */
  vegasSum: number;
}

const tally = (grades: Grade[]): Tally => ({
  w: grades.filter((g) => g === 'win').length,
  l: grades.filter((g) => g === 'loss').length,
  push: grades.filter((g) => g === 'push').length,
});

/** Value-call record and "Vegas expected" sum from the cards; `tier` limits it to one tier. */
export function valueCallStats(cards: readonly Card[], tier?: Tier): ValueCallStats {
  const calls = cards.flatMap((c) => (c.valueCall && (tier === undefined || c.valueCall.tier === tier) ? [c.valueCall] : []));
  const decided = calls.filter((v) => v.grade === 'win' || v.grade === 'loss');
  return {
    ...tally(calls.map((v) => v.grade)),
    calls: calls.length,
    vegasSum: decided.reduce((a, v) => a + v.vegasProb, 0),
  };
}

/** One value tier as the header shows it (numbers from the export's value_woe). */
export interface TierLine {
  tier: Tier;
  /** Graded calls, pushes excluded: the calls WOE is over. */
  n: number;
  woe: number;
  woePerCall: number | null;
}

/** The value tile: counts only (Amendment 6). WOE numbers are the exporter's. */
export interface ValueRecord extends Tally {
  won: number;
  vegasExpected: number;
  woe: number;
  tiers: TierLine[];
}

const TIER_KEY = { 1: 'one_star', 2: 'two_star' } as const;

/**
 * Fails the build unless the export's Amendment 6 numbers agree with the cards: per-tier and overall
 * call counts and W-L-push equal value_woe; the site's own Vegas-expected sums (from 4 dp
 * probabilities) are within rounding of the exporter's; the toss-up / confident pick records equal
 * prediction_quality_by_confidence; and each per-game flag agrees with a recomputation from the 4 dp
 * probabilities wherever that recomputation is clear of the cutoff.
 */
export function crossCheckAmendment6(d: SiteData, cards: readonly Card[]): void {
  const errors: string[] = [];
  const L = d.live_2026;
  const vw = L.value_woe;
  const blocks = [['all', undefined], ['one_star', 1], ['two_star', 2]] as const;
  for (const [key, tier] of blocks) {
    const site = valueCallStats(cards, tier);
    const e = vw[key];
    if (site.calls !== e.calls_logged || site.w !== e.wins || site.l !== e.losses || site.push !== e.pushes) {
      errors.push(`${key}: site ${site.calls} calls ${site.w}-${site.l}-${site.push}, export ${e.calls_logged} calls ${e.wins}-${e.losses}-${e.pushes}`);
    }
    if (e.n !== e.wins + e.losses || e.won !== e.wins) errors.push(`${key}: export n/won inconsistent with its record`);
    if (Math.abs(site.vegasSum - e.vegas_expected) > e.n * 1e-4 + 2e-4) {
      errors.push(`${key}: site Vegas expected ${site.vegasSum.toFixed(4)} vs export ${e.vegas_expected}`);
    }
  }
  const o = L.official_bets;
  if (vw.all.wins !== o.wins || vw.all.losses !== o.losses || vw.all.pushes !== o.voids) {
    errors.push(`value_woe.all ${vw.all.wins}-${vw.all.losses}-${vw.all.pushes} != official ${o.wins}-${o.losses}-${o.voids}`);
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
    const v = c.valueCall;
    if (v) {
      const edge = v.modelProb - v.vegasProb;
      if (v.tier === null) errors.push(`${c.gameId}: value call without a tier`);
      else if (Math.abs(edge - L.value_tier_two_star_edge) > 2e-4 && v.tier !== (edge >= L.value_tier_two_star_edge ? 2 : 1)) {
        errors.push(`${c.gameId}: tier ${v.tier} but the edge is ${edge.toFixed(4)}`);
      }
    }
  }
  if (errors.length) throw new Error(`tiers / toss-ups disagree with the exporter:\n  ${errors.join('\n  ')}`);
}

/**
 * Fails the build unless the site's value calls are the exporter's official paper record: same
 * games, same side, and the same graded W-L-push as live_2026.official_bets.
 */
export function crossCheckValueCalls(d: SiteData, cards: ReadonlyMap<string, Card>, stats: Tally): void {
  const errors: string[] = [];
  for (const g of d.games) {
    if (g.evaluation !== 'live_2026') continue;
    const v = cards.get(g.game_id)?.valueCall ?? null;
    const want = g.bet ? g.bet_side : null;
    if ((v?.side ?? null) !== want) errors.push(`${g.game_id}: site value call ${v?.side ?? 'none'}, export ${want ?? 'none'}`);
  }
  const o = d.live_2026.official_bets;
  if (stats.w !== o.wins || stats.l !== o.losses || stats.push !== o.voids) {
    errors.push(`value-call record ${stats.w}-${stats.l}-${stats.push} != official ${o.wins}-${o.losses}-${o.voids}`);
  }
  if (errors.length) throw new Error(`value calls disagree with the exporter's official record:\n  ${errors.join('\n  ')}`);
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
    const valueCalls = tally(cards.flatMap((c) => (c.valueCall ? [c.valueCall.grade] : [])));
    const graded = picks.w + picks.l + picks.push > 0;
    pages.set(info.week, {
      info,
      banner: weekBanner(info, L.first_official_week),
      cards,
      summary: graded ? { picks, valueCalls } : null,
    });
  }
  const allCards = [...pages.values()].flatMap((p) => p.cards);
  const vc = valueCallStats(allCards);
  crossCheckValueCalls(d, new Map(allCards.map((c) => [c.gameId, c])), vc);
  crossCheckAmendment6(d, allCards);
  const vw = L.value_woe;
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
        w: vw.all.wins,
        l: vw.all.losses,
        push: vw.all.pushes,
        won: vw.all.won,
        vegasExpected: vw.all.vegas_expected,
        woe: vw.all.woe,
        tiers: ([1, 2] as const).map((t) => {
          const e = vw[TIER_KEY[t]];
          return { tier: t, n: e.n, woe: e.woe, woePerCall: e.woe_per_call };
        }),
      },
    },
    generatedAt: d.generated_at,
    asOf: L.as_of_utc,
    commit: d.source_commit.hash,
  };
}
