/*
  Build-time only. Reads ../exports/site_data.json (relative to site/, which is the working
  directory for npm scripts locally and on Vercel with Root Directory = site/), validates it, and
  reduces it to what the pages render. NFL_SITE_DATA overrides the path. No side effects here, so
  tests can import it; the build's singleton lives in site.ts.
*/
import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseSiteData, type Game, type ScheduleGame, type SiteData } from './schema.ts';
import { buildCard, type Card, type Grade } from './card.ts';
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

export interface ValueCallStats extends Tally {
  /** w / (w + l); null before any value call is graded. */
  hitRate: number | null;
  /** Mean of Vegas's no-vig probability for the value-call side over graded calls, pushes excluded. */
  vegasExpected: number | null;
}

const tally = (grades: Grade[]): Tally => ({
  w: grades.filter((g) => g === 'win').length,
  l: grades.filter((g) => g === 'loss').length,
  push: grades.filter((g) => g === 'push').length,
});

/** Season value-call record and the "Vegas expected" rate, from the cards. */
export function valueCallStats(cards: readonly Card[]): ValueCallStats {
  const calls = cards.flatMap((c) => (c.valueCall ? [c.valueCall] : []));
  const t = tally(calls.map((v) => v.grade));
  const decided = calls.filter((v) => v.grade === 'win' || v.grade === 'loss');
  return {
    ...t,
    hitRate: decided.length ? t.w / decided.length : null,
    vegasExpected: decided.length ? decided.reduce((a, v) => a + v.vegasProb, 0) / decided.length : null,
  };
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
    picks: { w: number; l: number; accuracy: number; vegasFavorite: number };
    valueCalls: ValueCallStats;
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

  const champ = L.prediction_quality.champion.accuracy;
  const fav = L.prediction_quality.vegas_favorite.accuracy;
  if (champ.value === null || fav.value === null) throw new Error('live_2026 accuracy has no value yet');

  return {
    current: currentWeek(d.schedule_2026),
    weeks,
    pages,
    record: {
      picks: { w: champ.k, l: champ.n - champ.k, accuracy: champ.value, vegasFavorite: fav.value },
      valueCalls: vc,
    },
    generatedAt: d.generated_at,
    asOf: L.as_of_utc,
    commit: d.source_commit.hash,
  };
}
