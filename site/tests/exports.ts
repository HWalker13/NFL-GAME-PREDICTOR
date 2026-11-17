/*
  The two exports the tests read.

  FIXTURE: a frozen copy of the 2026-11-12 export (tests/fixtures/site_data_2026-11-12.json; how it was
  made is in tests/fixtures/README.md). Every exact-value assertion (records, week 10 as the current
  week, banners, NE at DET's 49.6%, the snapshot time ...) reads this file, so those assertions stay
  true after every weekly export.

  REAL: the export the build actually reads (dataPath(): ../exports/site_data.json, or NFL_SITE_DATA).
  Tests on it assert only invariants that hold every week.
*/
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readdirSync, readFileSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { dataPath, loadSiteData } from '../src/lib/data.ts';
import type { SiteData } from '../src/lib/schema.ts';

export const FIXTURE_PATH = resolve(dirname(fileURLToPath(import.meta.url)), 'fixtures/site_data_2026-11-12.json');
export const REAL_PATH = dataPath();

export const loadFixture = (): SiteData => loadSiteData(FIXTURE_PATH);
export const loadReal = (): SiteData => loadSiteData(REAL_PATH);

/**
 * Env for an `astro build` of one export. The fixture is honestly marked dirty (it was regenerated
 * with two later snapshots removed; see the fixtures README), so its build needs NFL_ALLOW_DIRTY=1.
 * The real build gets the caller's environment unchanged, exactly like `npm run build`.
 */
export function buildEnv(which: 'fixture' | 'real'): NodeJS.ProcessEnv {
  return which === 'fixture'
    ? { ...process.env, NFL_SITE_DATA: FIXTURE_PATH, NFL_ALLOW_DIRTY: '1' }
    : { ...process.env };
}

export interface BuiltSite {
  dir: string;
  files: string[];
  html: (p: string) => string;
  /** Visible text plus attribute values (aria-labels count), without CSS. */
  text: (p: string) => string;
  cleanup: () => void;
}

/** Builds the real site (astro build into a temp dir) from one export. */
export function buildOnce(which: 'fixture' | 'real'): BuiltSite {
  const dir = mkdtempSync(join(tmpdir(), `nfl-picks-dist-${which}-`));
  execFileSync('npx', ['astro', 'build', '--outDir', dir], { stdio: 'pipe', env: buildEnv(which) });
  const files: string[] = [];
  const walk = (d: string): void => {
    for (const f of readdirSync(d)) {
      const p = join(d, f);
      if (statSync(p).isDirectory()) walk(p);
      else files.push(p.slice(dir.length + 1));
    }
  };
  walk(dir);
  const html = (p: string): string => readFileSync(join(dir, p), 'utf8');
  const text = (p: string): string =>
    html(p).replace(/<style[\s\S]*?<\/style>/g, '').replace(/<[^>]*?(?:aria-label|content|title)="([^"]*)"[^>]*>/g, ' $1 ').replace(/<[^>]+>/g, ' ')
      .replace(/&#39;/g, "'").replace(/&quot;/g, '"').replace(/&amp;/g, '&');
  return { dir, files, html, text, cleanup: () => rmSync(dir, { recursive: true, force: true }) };
}
