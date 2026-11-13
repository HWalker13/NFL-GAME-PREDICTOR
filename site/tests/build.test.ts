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
      expect(t, f).toMatch(/Season: model \d+ of \d+ correct \(\d+%\) · Vegas favorite \d+%/);
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
    expect(text('week/7/index.html')).toMatch(/Correct/);
    expect(text('week/7/index.html')).toMatch(/Model rates [A-Z]{2,3} higher than Vegas does \(\d+% vs \d+%\)/);
    expect(text('week/10/index.html')).toContain('Not played yet');
    const w10 = text('week/10/index.html').replace(/\s+/g, ' ');
    expect(w10).toContain('DET pick · 50.4%');
    expect(w10).toContain('Vegas: DET 58.3%');
    expect(w10).toContain('Model rates NE higher than Vegas does (49.6% vs 41.7%)');
    expect(w10).toContain('NYG pick · 58%'); // other cards stay whole numbers
    for (const f of files) expect(text(f), f).not.toMatch(/\b50%/);
  });
});
