/*
  Builds the real site (astro build into a temp dir) twice and checks the output:
  - from the real export (../exports/site_data.json): only invariants that hold every week
    (every page, no client JS or JSON, no betting vocabulary, a card for every scheduled game,
    header numbers equal the export, toss-up pills only on flagged games, no "50%");
  - from the frozen 2026-11-12 fixture (tests/fixtures/): the exact rendered values.
*/
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { buildSite } from '../src/lib/data.ts';
import type { SiteData } from '../src/lib/schema.ts';
import { buildOnce, loadFixture, loadReal, type BuiltSite } from './exports.ts';

const weekPages = Array.from({ length: 18 }, (_, i) => `week/${i + 1}/index.html`);
const flat = (s: string): string => s.replace(/\s+/g, ' ');
const articles = (h: string): string[] => h.match(/<article[\s\S]*?<\/article>/g) ?? [];

/** Games the export flags as toss-ups in week w, as "AWAY at HOME", sorted. */
const tossUpsIn = (d: SiteData, w: number): string[] =>
  d.games.filter((g) => g.evaluation === 'live_2026' && g.week === w && g.toss_up).map((g) => `${g.away} at ${g.home}`).sort();

/** Cards on a page that carry the TOSS UP pill; asserts the pill sits in the top row next to a kept pick chip. */
function tossPills(html: string): string[] {
  return [...html.matchAll(/<article[^>]*aria-label="([A-Z]+ at [A-Z]+)"[\s\S]*?<\/article>/g)]
    .filter((m) => /class="toss[^"]*"/.test(m[0]))
    .map((m) => {
      const top = m[0].match(/<div class="top[^"]*"[^>]*>([\s\S]*?)<\/div>/)![1]!;
      expect(top, m[1]).toMatch(/class="toss[^"]*"[^>]*>Toss up<span class="sr[^"]*"[^>]*>: the model's two win chances are within 10 points<\/span>/);
      expect(m[0], m[1]).toMatch(/Pick: <strong/);
      return m[1]!;
    })
    .sort();
}

describe('build output from the real export (invariants, every week)', () => {
  let b: BuiltSite;
  let d: SiteData;
  beforeAll(() => {
    d = loadReal();
    b = buildOnce('real');
  }, 120_000);
  afterAll(() => b?.cleanup());

  it('builds all 18 week pages and the home page, nothing else', () => {
    expect([...b.files].sort()).toEqual(['index.html', ...weekPages].sort());
  });

  it('ships no client JS and never the JSON', () => {
    for (const f of b.files) {
      const h = b.html(f);
      expect(h, f).not.toMatch(/<script/i);
      expect(h, f).not.toMatch(/schema_version|model_home_prob|edge_threshold/);
    }
    expect(b.files.some((f) => /\.(js|mjs|json)$/.test(f))).toBe(false);
  });

  it('never shows betting vocabulary (the footer\'s "Not betting advice." aside)', () => {
    const banned = [/\bbets?\b/i, /\bunits?\b/i, /\bROI\b/i, /\bCLV\b/i, /closing.line/i, /interval/i, /\bci95\b/i, /\bwager/i];
    for (const f of b.files) {
      const t = b.text(f);
      for (const re of banned) expect(t, `${f} ${re}`).not.toMatch(re);
    }
  });

  it('never rounds a percentage to 50%', () => {
    for (const f of b.files) expect(b.text(f), f).not.toMatch(/\b50%/);
  });

  it('has the header, tabs and footer on every page', () => {
    for (const f of b.files) {
      const t = b.text(f);
      expect(t, f).toMatch(/NFL picks/);
      expect(t, f).toMatch(/Week \d+ · 2026/);
      const ts = flat(t);
      expect(ts, f).toMatch(/Picks \d+–\d+ \d+% accuracy Vegas favorite \d+% Confident picks \d+% · Toss-ups \d+%/);
      expect(ts, f).toMatch(
        /Value calls (?:\+\d+\.\d ahead of Vegas|\d+\.\d behind Vegas|Even with Vegas) Won \d+ · Vegas expected \d+\.\d ★ one-star value calls: \d+ calls? · .+ ★★ two-star value calls: \d+ calls? · .+ Record \d+–\d+/,
      );
      expect(ts, f).not.toMatch(/hit rate/i);
      expect(ts, f).toContain('Small sample — read this as a trend, not proof.');
      expect(b.html(f), f).toMatch(/<a href="https:\/\/holden-anderson\.vercel\.app\/projects\/nfl-game-predictor\/results\/"[^>]*>Small sample/);
      expect(t, f).toContain('Paper picks for a machine-learning project. Not betting advice.');
      expect(t, f).toContain("Vegas % is the market's implied win chance at the time the pick was logged.");
      expect(b.html(f), f).toMatch(/href="https:\/\/github.com\/HWalker13\/NFL-GAME-PREDICTOR\/commit\/[0-9a-f]{40}"/);
      expect(b.html(f), f).toContain('href="https://holden-anderson.vercel.app/projects/nfl-game-predictor/results/"');
      expect(b.html(f).match(/class="tab(?: [^"]*)?"/g)?.length, f).toBe(18);
    }
  });

  it('header numbers equal the export: picks record, value-call record (official_bets) and tier counts', () => {
    const q = d.live_2026.prediction_quality.champion.accuracy;
    const o = d.live_2026.official_bets;
    const vw = d.live_2026.value_woe;
    const record = `Record ${o.wins}–${o.losses}${o.voids > 0 ? ` (${o.voids} ${o.voids === 1 ? 'push' : 'pushes'})` : ''}`;
    const calls = (n: number) => `${n} ${n === 1 ? 'call' : 'calls'}`;
    for (const f of b.files) {
      const t = flat(b.text(f));
      expect(t, f).toContain(`Picks ${q.k}–${q.n - q.k} `);
      expect(t, f).toContain(`Won ${vw.all.won} · Vegas expected ${vw.all.vegas_expected.toFixed(1)}`);
      expect(t, f).toMatch(new RegExp(`${record.replace(/[()]/g, '\\$&')}(?![\\d(])`));
      expect(t, f).toContain(`★ one-star value calls: ${calls(vw.one_star.n)} · `);
      expect(t, f).toContain(`★★ two-star value calls: ${calls(vw.two_star.n)} · `);
    }
  });

  it('a card for every scheduled game, on its own week', () => {
    for (let w = 1; w <= 18; w++) {
      const want = d.schedule_2026.filter((g) => g.week === w).map((g) => `${g.away} at ${g.home}`).sort();
      const got = [...b.html(`week/${w}/index.html`).matchAll(/<article[^>]*aria-label="([A-Z]+ at [A-Z]+)"/g)].map((m) => m[1]).sort();
      expect(got, `week ${w}`).toEqual(want);
    }
  });

  it('home page is the current week, and every page marks itself and the current week', () => {
    const current = buildSite(d).current;
    const strip = (s: string) => s.replace(/aria-current="page"/g, '');
    expect(b.text('index.html')).toContain(`Week ${current} · 2026`);
    expect(strip(b.html('index.html'))).toBe(strip(b.html(`week/${current}/index.html`)));
    for (let w = 1; w <= 18; w++) {
      const h = b.html(`week/${w}/index.html`);
      expect(h.match(/aria-current="page"/g)?.length, `week ${w}`).toBe(1);
      expect(h, `week ${w}`).toMatch(new RegExp(`href="/week/${w}/"[^>]*aria-current="page"`));
      expect(h.match(/class="tab now"/g)?.length, `week ${w}`).toBe(1);
      expect(h, `week ${w}`).toMatch(new RegExp(`class="tab now"[^>]*href="/week/${current}/"|href="/week/${current}/"[^>]*class="tab now"`));
    }
  });

  it('details element on every card with a pick, collapsed, and nowhere else', () => {
    let withPick = 0;
    for (const f of b.files) {
      for (const card of articles(b.html(f))) {
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

  it('every value chip carries a tier; no pick chip does', () => {
    for (const f of b.files) {
      for (const card of articles(b.html(f))) {
        expect(/Value call: /.test(card), `${f} untiered value chip`).toBe(false);
        expect(/Pick <span class="stars/.test(card), f).toBe(false);
      }
    }
  });

  it('TOSS UP pill on exactly the toss-up games, in the top row, pick chip kept', () => {
    for (let w = 1; w <= 18; w++) expect(tossPills(b.html(`week/${w}/index.html`)), `week ${w}`).toEqual(tossUpsIn(d, w));
    const css = b.html('week/3/index.html').replace(/\s+/g, '');
    expect(css).toMatch(/\.toss\[[^\]]*\]\{[^}]*background:var\(--amber-bg\);color:var\(--on-amber\)/);
    expect(css).toMatch(/text-transform:uppercase/);
  });

  it('card grid: one column, two from 768px, three from 1200px; container 1150px', () => {
    const css = b.html('week/3/index.html').replace(/\s+/g, '');
    expect(css).toMatch(/\.games\[[^\]]*\]\{[^}]*grid-template-columns:1fr/);
    // The minifier may write (min-width:768px) as (width>=768px).
    expect(css).toMatch(/@media\((?:min-width:|width>=)768px\)\{\.games\[[^\]]*\]\{grid-template-columns:repeat\(2,minmax\(0,1fr\)\)/);
    expect(css).toMatch(/@media\((?:min-width:|width>=)1200px\)\{\.games\[[^\]]*\]\{grid-template-columns:repeat\(3,minmax\(0,1fr\)\)/);
    expect(css).toMatch(/\.wrap\{max-width:1150px;margin:0auto/);
    expect(b.html('week/3/index.html')).toMatch(/<ol class="games[^"]*"/);
  });
});

describe('build output from the 2026-11-12 fixture (exact values)', () => {
  let b: BuiltSite;
  let d: SiteData;
  beforeAll(() => {
    d = loadFixture();
    b = buildOnce('fixture');
  }, 120_000);
  afterAll(() => b?.cleanup());
  const html = (p: string) => b.html(p);
  const text = (p: string) => b.text(p);

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
  });

  it('season tiles show the real numbers', () => {
    const t = flat(text('week/3/index.html'));
    expect(t).toContain('Picks 44–31 59% accuracy Vegas favorite 60% Confident picks 63% · Toss-ups 44%');
    expect(t).toContain(
      'Value calls +3.6 ahead of Vegas Won 28 · Vegas expected 24.4 ' +
        '★ one-star value calls: 22 calls · +1.2 ahead (+0.6 per 10 calls) ' +
        '★★ two-star value calls: 30 calls · +2.4 ahead (+0.8 per 10 calls) ' +
        'Record 28–24 (1 push)',
    );
  });

  it('week summary line on graded weeks only', () => {
    const t = (p: string) => flat(text(p));
    expect(t('week/7/index.html')).toMatch(/Picks \d+–\d+ · Value calls \d+–\d+/);
    expect(t('week/5/index.html')).toMatch(/Picks \d+–\d+ · Value calls \d+–\d+ \(1 push\)/);
    expect(t('week/10/index.html')).not.toMatch(/Picks \d+–\d+ ·/);
    expect(t('week/12/index.html')).not.toMatch(/Picks \d+–\d+ ·/);
  });

  it('chips: pick and value call, with icon and text', () => {
    const w7 = html('week/7/index.html');
    expect(w7).toMatch(/class="chip win[^"]*"[\s\S]*?<svg[\s\S]*?Pick: <strong[^>]*>[A-Z]{2,3}<\/strong>[\s\S]*?\(correct\)/);
    expect(w7).toMatch(/class="chip loss[^"]*"[\s\S]*?<svg[\s\S]*?\(wrong\)/);
    expect(flat(text('week/7/index.html'))).toMatch(/Value call ★{1,2} (?:one|two)-star value call : [A-Z]{2,3}/);
    const w5 = flat(text('week/5/index.html'));
    expect(w5).toMatch(/Pick: WAS · Tie/);
    expect(w5).toMatch(/Value call ★{1,2} (?:one|two)-star value call : NYG · Push/);
    expect(html('week/10/index.html')).toMatch(/class="chip open[^"]*"/);
  });

  it('week 10 NE at DET: one decimal everywhere, details bars and sentence', () => {
    const w10 = flat(text('week/10/index.html'));
    expect(w10).toContain('Pick: DET');
    expect(w10).toMatch(/Value call ★{1,2} (?:one|two)-star value call : NE/);
    expect(w10).toContain('Model NE 49.6% DET 50.4%');
    expect(w10).toContain('Vegas NE 41.7% DET 58.3%');
    expect(w10).toContain('Model gives NE 49.6%, Vegas 41.7% — the model thinks NE is underrated.');
    expect(w10).toMatch(/Model WAS \d+% NYG \d+%/); // other cards stay whole numbers
  });

  it('primetime badges on the right games', () => {
    const badges = (p: string) =>
      [...html(p).matchAll(/aria-label="([A-Z]+ at [A-Z]+)"[\s\S]*?<\/article>/g)]
        .map((m) => [m[1], m[0].match(/class="badge[^"]*"[^>]*>(\w+)</)?.[1] ?? null])
        .filter(([, x]) => x);
    expect(badges('week/3/index.html')).toEqual([['ATL at GB', 'TNF'], ['LA at DEN', 'SNF'], ['PHI at CHI', 'MNF']]);
    expect(badges('week/10/index.html')).toEqual([['WAS at NYG', 'TNF'], ['PIT at CIN', 'SNF'], ['LAC at BAL', 'MNF']]);
  });

  it('tier chip: visible stars hidden from screen readers, which get the words instead', () => {
    expect(html('week/7/index.html')).toMatch(
      /<span aria-hidden="true"[^>]*>Value call <span class="stars[^"]*"[^>]*>★{1,2}<\/span><\/span><span class="sr[^"]*"[^>]*>(?:one|two)-star value call<\/span>: <strong/,
    );
  });

  it('TOSS UP pills on the 2026-11-12 weeks, next to a primetime badge where both apply', () => {
    for (const w of [3, 5, 7, 10]) expect(tossPills(html(`week/${w}/index.html`)), `week ${w}`).toEqual(tossUpsIn(d, w));
    expect(tossUpsIn(d, 10).length).toBeGreaterThan(0);
    expect(tossPills(html('week/12/index.html'))).toEqual([]); // no picks yet, no pills
    // week 3 PHI at CHI is MNF and a toss-up
    const phiChi = html('week/3/index.html').match(/<article[^>]*aria-label="PHI at CHI"[\s\S]*?<\/article>/)![0];
    expect(phiChi).toMatch(/class="badge[^"]*"[^>]*>MNF<\/span>\s*<span class="toss/);
  });
});
