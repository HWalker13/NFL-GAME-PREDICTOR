/*
  Build-time only. Reads ../exports/site_data.json (relative to site/, which is the working
  directory for npm scripts locally and on Vercel with Root Directory = site/), validates it, and
  reduces it to what the pages render. NFL_SITE_DATA overrides the path. No side effects here, so
  tests can import it; the build's singleton lives in site.ts.
*/
import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseSiteData, type Game, type ScheduleGame, type SiteData } from './schema.ts';
import { buildCard, type Card } from './card.ts';
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
}

export interface Site {
  current: number;
  weeks: WeekInfo[];
  pages: Map<number, WeekPage>;
  record: { k: number; n: number; value: number; vegasValue: number };
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
    pages.set(info.week, {
      info,
      banner: weekBanner(info, L.first_official_week),
      cards: games.map((g) => buildCard(g, live.get(g.game_id), L.edge_threshold, now)),
    });
  }

  const champ = L.prediction_quality.champion.accuracy;
  const fav = L.prediction_quality.vegas_favorite.accuracy;
  if (champ.value === null || fav.value === null) throw new Error('live_2026 accuracy has no value yet');

  return {
    current: currentWeek(d.schedule_2026),
    weeks,
    pages,
    record: { k: champ.k, n: champ.n, value: champ.value, vegasValue: fav.value },
    generatedAt: d.generated_at,
    asOf: L.as_of_utc,
    commit: d.source_commit.hash,
  };
}
