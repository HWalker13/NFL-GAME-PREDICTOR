/*
  Builds the real site once (astro build into a temp dir) and checks the output: every week page,
  no client JS or JSON shipped, and none of the words the site must never show.
*/
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readdirSync, readFileSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';

let out = '';
const files: string[] = [];
const html = (p: string): string => readFileSync(join(out, p), 'utf8');
/** Visible text plus attribute values (aria-labels count), without CSS. */
const text = (p: string): string =>
  html(p).replace(/<style[\s\S]*?<\/style>/g, '').replace(/<[^>]*?(?:aria-label|content|title)="([^"]*)"[^>]*>/g, ' $1 ').replace(/<[^>]+>/g, ' ')
    .replace(/&#39;/g, "'").replace(/&quot;/g, '"').replace(/&amp;/g, '&');

function walk(dir: string): void {
  for (const f of readdirSync(dir)) {
    const p = join(dir, f);
    if (statSync(p).isDirectory()) walk(p);
    else files.push(p.slice(out.length + 1));
  }
}

beforeAll(() => {
  out = mkdtempSync(join(tmpdir(), 'nfl-picks-dist-'));
  execFileSync('npx', ['astro', 'build', '--outDir', out], { stdio: 'pipe' });
  walk(out);
}, 120_000);

afterAll(() => {
  if (out) rmSync(out, { recursive: true, force: true });
});

const weekPages = Array.from({ length: 18 }, (_, i) => `week/${i + 1}/index.html`);

describe('build output', () => {
  it('builds all 18 week pages and the home page, nothing else', () => {
    expect(files.sort()).toEqual(['index.html', ...weekPages].sort());
  });

  it('ships no client JS and never the JSON', () => {
    for (const f of files) {
      const h = html(f);
      expect(h, f).not.toMatch(/<script/i);
      expect(h, f).not.toMatch(/schema_version|model_home_prob|edge_threshold/);
    }
    expect(files.some((f) => /\.(js|mjs|json)$/.test(f))).toBe(false);
  });

  it('never shows betting vocabulary (the footer\'s "Not betting advice." aside)', () => {
    const banned = [/\bbets?\b/i, /\bunits?\b/i, /\bROI\b/i, /\bCLV\b/i, /closing.line/i, /interval/i, /\bci95\b/i, /\bwager/i];
    for (const f of files) {
      const t = text(f);
      for (const re of banned) expect(t, `${f} ${re}`).not.toMatch(re);
    }
  });

  it('has the header, tabs and footer on every page', () => {
    for (const f of files) {
      const t = text(f);
      expect(t, f).toMatch(/NFL picks/);
      expect(t, f).toMatch(/Week \d+ · 2026/);
      const ts = t.replace(/\s+/g, ' ');
      expect(ts, f).toMatch(/Picks \d+–\d+ \d+% accuracy Vegas favorite \d+%/);
      expect(ts, f).toMatch(/Value calls \d+–\d+(?: \d+ push(?:es)?)? \d+% hit rate Vegas expected \d+%/);
      expect(ts, f).toContain('Small sample — read this as a trend, not proof.');
      expect(html(f), f).toMatch(/<a href="https:\/\/holden-anderson\.vercel\.app\/projects\/nfl-game-predictor\/results\/"[^>]*>Small sample/);
      expect(t, f).toContain('Paper picks for a machine-learning project. Not betting advice.');
      expect(t, f).toContain("Vegas % is the market's implied win chance at the time the pick was logged.");
      expect(html(f), f).toMatch(/href="https:\/\/github.com\/HWalker13\/NFL-GAME-PREDICTOR\/commit\/[0-9a-f]{40}"/);
      expect(html(f), f).toContain('href="https://holden-anderson.vercel.app/projects/nfl-game-predictor/results/"');
      expect(html(f).match(/class="tab(?: [^"]*)?"/g)?.length, f).toBe(18);
    }
  });

  it('marks the viewed tab and the current week', () => {
    const w5 = html('week/5/index.html');
    expect(w5).toMatch(/href="\/week\/5\/"[^>]*aria-current="page"/);
    expect(w5.match(/aria-current="page"/g)?.length).toBe(1);
    expect(w5).toMatch(/class="tab now"[^>]*href="\/week\/10\/"|href="\/week\/10\/"[^>]*class="tab now"/);
    expect(w5).toMatch(/class="tab dim"[^>]*href="\/week\/9\/"|href="\/week\/9\/"[^>]*class="tab dim"/);
  });

  it('home page is the current week', () => {
    const strip = (s: string) => s.replace(/aria-current="page"/g, '');
    expect(text('index.html')).toContain('Week 10 · 2026');
    expect(strip(html('index.html'))).toBe(strip(html('week/10/index.html')));
  });

  it('renders the week states', () => {
    expect(text('week/1/index.html')).toContain('Tracking started in week 3.');
    expect(text('week/8/index.html')).toContain('Picks were logged for 1 of 14 games this week.');
    expect(text('week/9/index.html')).toContain('No picks were logged this week.');
    expect(text('week/12/index.html')).toContain("Picks will appear here once they're logged.");
    expect(text('week/10/index.html')).toContain('Not played yet');
    for (const f of files) expect(text(f), f).not.toMatch(/\b50%/);
  });

  it('season tiles show the real numbers', () => {
    const t = text('week/3/index.html').replace(/\s+/g, ' ');
    expect(t).toContain('Picks 44–31 59% accuracy Vegas favorite 60%');
    expect(t).toMatch(/Value calls 28–24 1 push 54% hit rate Vegas expected \d+%/);
  });

  it('week summary line on graded weeks only', () => {
    const t = (p: string) => text(p).replace(/\s+/g, ' ');
    expect(t('week/7/index.html')).toMatch(/Picks \d+–\d+ · Value calls \d+–\d+/);
    expect(t('week/5/index.html')).toMatch(/Picks \d+–\d+ · Value calls \d+–\d+ \(1 push\)/);
    expect(t('week/10/index.html')).not.toMatch(/Picks \d+–\d+ ·/);
    expect(t('week/12/index.html')).not.toMatch(/Picks \d+–\d+ ·/);
  });

  it('chips: pick and value call, with icon and text', () => {
    const w7 = html('week/7/index.html');
    expect(w7).toMatch(/class="chip win[^"]*"[\s\S]*?<svg[\s\S]*?Pick: <strong[^>]*>[A-Z]{2,3}<\/strong>[\s\S]*?\(correct\)/);
    expect(w7).toMatch(/class="chip loss[^"]*"[\s\S]*?<svg[\s\S]*?\(wrong\)/);
    expect(text('week/7/index.html').replace(/\s+/g, ' ')).toMatch(/Value call: [A-Z]{2,3}/);
    const w5 = text('week/5/index.html').replace(/\s+/g, ' ');
    expect(w5).toMatch(/Pick: WAS · Tie/);
    expect(w5).toMatch(/Value call: NYG · Push/);
    expect(html('week/10/index.html')).toMatch(/class="chip open[^"]*"/);
  });

  it('week 10 NE at DET: one decimal everywhere, details bars and sentence', () => {
    const w10 = text('week/10/index.html').replace(/\s+/g, ' ');
    expect(w10).toContain('Pick: DET');
    expect(w10).toContain('Value call: NE');
    expect(w10).toContain('Model NE 49.6% DET 50.4%');
    expect(w10).toContain('Vegas NE 41.7% DET 58.3%');
    expect(w10).toContain('Model gives NE 49.6%, Vegas 41.7% — the model thinks NE is underrated.');
    expect(w10).toMatch(/Model WAS \d+% NYG \d+%/); // other cards stay whole numbers
  });

  it('details element on every card with a pick, collapsed, and nowhere else', () => {
    let withPick = 0;
    for (const f of files) {
      for (const card of html(f).match(/<article[\s\S]*?<\/article>/g) ?? []) {
        const hasPick = /Pick: <strong\b/.test(card);
        expect(/<details\b/.test(card), `${f} ${card.slice(0, 120)}`).toBe(hasPick);
        if (hasPick) {
          withPick++;
          expect(card).toMatch(/<details class="details[^"]*"[^>]*>\s*<summary[^>]*>Details<\/summary>/);
          expect(card).not.toMatch(/<details[^>]*\bopen\b/);
        }
      }
    }
    expect(withPick).toBeGreaterThan(90);
  });

  it('primetime badges on the right games', () => {
    const badges = (p: string) =>
      [...html(p).matchAll(/aria-label="([A-Z]+ at [A-Z]+)"[\s\S]*?<\/article>/g)]
        .map((m) => [m[1], m[0].match(/class="badge[^"]*"[^>]*>(\w+)</)?.[1] ?? null])
        .filter(([, b]) => b);
    expect(badges('week/3/index.html')).toEqual([['ATL at GB', 'TNF'], ['LA at DEN', 'SNF'], ['PHI at CHI', 'MNF']]);
    expect(badges('week/10/index.html')).toEqual([['WAS at NYG', 'TNF'], ['PIT at CIN', 'SNF'], ['LAC at BAL', 'MNF']]);
  });

  it('card grid: one column, two from 768px, three from 1200px; container 1150px', () => {
    const css = html('week/3/index.html').replace(/\s+/g, '');
    expect(css).toMatch(/\.games\[[^\]]*\]\{[^}]*grid-template-columns:1fr/);
    // The minifier may write (min-width:768px) as (width>=768px).
    expect(css).toMatch(/@media\((?:min-width:|width>=)768px\)\{\.games\[[^\]]*\]\{grid-template-columns:repeat\(2,minmax\(0,1fr\)\)/);
    expect(css).toMatch(/@media\((?:min-width:|width>=)1200px\)\{\.games\[[^\]]*\]\{grid-template-columns:repeat\(3,minmax\(0,1fr\)\)/);
    expect(css).toMatch(/\.wrap\{max-width:1150px;margin:0auto/);
    expect(html('week/3/index.html')).toMatch(/<ol class="games[^"]*"/);
  });
});
