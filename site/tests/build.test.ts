/*
  Builds the real site (astro build into a temp dir) twice and checks the output:
  - from the real export (../exports/site_data.json): only invariants that hold every week
    (every page, no client JS or JSON, no betting vocabulary, a card for every scheduled game,
    header and vs Vegas numbers equal the export, value chips = the export's ★/★★ calls, no value
    wording on any other game, toss-up pills only on flagged games, no "50%");
  - from the frozen 2026-11-12 fixture (tests/fixtures/): the exact rendered values.
*/
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { buildSite } from '../src/lib/data.ts';
import type { SiteData } from '../src/lib/schema.ts';
import { buildOnce, loadFixture, loadReal, type BuiltSite } from './exports.ts';

const weekPages = Array.from({ length: 18 }, (_, i) => `week/${i + 1}/index.html`);
const VS = 'vs-vegas/index.html';
const flat = (s: string): string => s.replace(/\s+/g, ' ');
const articles = (h: string): string[] => h.match(/<article[\s\S]*?<\/article>/g) ?? [];

/** Games the export flags as toss-ups in week w, as "AWAY at HOME", sorted. */
const tossUpsIn = (d: SiteData, w: number): string[] =>
  d.games.filter((g) => g.evaluation === 'live_2026' && g.week === w && g.toss_up).map((g) => `${g.away} at ${g.home}`).sort();

/** Every card on a page as [label "AWAY at HOME", article html]. */
const cardsOf = (html: string): [string, string][] =>
  [...html.matchAll(/<article[^>]*aria-label="([A-Z]+ at [A-Z]+)"[\s\S]*?<\/article>/g)].map((m) => [m[1]!, m[0]]);

/** Value chips on a page by tier, as "AWAY at HOME TEAM", sorted. */
function valueChips(html: string): Record<'one_star' | 'two_star', string[]> {
  const out: Record<'one_star' | 'two_star', string[]> = { one_star: [], two_star: [] };
  for (const [label, card] of cardsOf(html)) {
    const star = card.match(/Value call <span class="stars[^"]*"[^>]*>(★{1,2})<\/span><\/span><span class="sr[^"]*"[^>]*>(one|two)-star value call<\/span>: <strong[^>]*>([A-Z]+)</);
    if (star) {
      expect(star[1]!.length === 1 ? 'one' : 'two', label).toBe(star[2]);
      out[star[1]!.length === 1 ? 'one_star' : 'two_star'].push(`${label} ${star[3]}`);
    }
  }
  out.one_star.sort();
  out.two_star.sort();
  return out;
}

/** The export's paper bets in week w by tier (all three), as "AWAY at HOME TEAM", sorted. */
function exportTiers(d: SiteData, w: number): Record<'lean' | 'one_star' | 'two_star', string[]> {
  const out: Record<'lean' | 'one_star' | 'two_star', string[]> = { lean: [], one_star: [], two_star: [] };
  for (const g of d.games) {
    if (g.evaluation !== 'live_2026' || g.week !== w || !g.bet) continue;
    out[g.value_tier!].push(`${g.away} at ${g.home} ${g.bet_side === 'home' ? g.home : g.away}`);
  }
  for (const k of Object.keys(out) as (keyof typeof out)[]) out[k].sort();
  return out;
}

/** The export's starred (★/★★) calls in week w, in valueChips' format. */
const exportStarred = (d: SiteData, w: number) => {
  const { one_star, two_star } = exportTiers(d, w);
  return { one_star, two_star };
};

/** Grades of the value chips on a page, tallied. */
function starredChipTally(html: string): { w: number; l: number; push: number } {
  const grades = [...html.matchAll(/<li class="chip (win|loss|push|open)\b[^"]*"[^>]*>((?:(?!<\/li>)[\s\S])*)<\/li>/g)]
    .filter((m) => /Value call <span class="stars/.test(m[2]!))
    .map((m) => m[1]!);
  return { w: grades.filter((g) => g === 'win').length, l: grades.filter((g) => g === 'loss').length, push: grades.filter((g) => g === 'push').length };
}

/** The week summary line's inner html, or null when the page has none. */
const summaryHtml = (html: string): string | null => html.match(/<p class="summary[^"]*"[^>]*>([\s\S]*?)<\/p>/)?.[1] ?? null;
const textOf = (h: string): string => h.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
/** An aria-hidden span, one level of nested spans allowed. */
const ARIA_HIDDEN = /<span aria-hidden="true"[^>]*>(?:[^<]|<span[^>]*>[^<]*<\/span>)*<\/span>/g;

/** The week summary line as seen (screen-reader-only text removed), or null when the page has none. */
const summaryLine = (html: string): string | null => {
  const h = summaryHtml(html);
  return h === null ? null : textOf(h.replace(/<span class="sr[^"]*"[^>]*>[^<]*<\/span>/g, ''));
};

/** The week summary line as a screen reader reads it (aria-hidden text removed), or null when none. */
const summarySpoken = (html: string): string | null => {
  const h = summaryHtml(html);
  return h === null ? null : textOf(h.replace(ARIA_HIDDEN, ''));
};

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

/** Owner's text (2026-11-24), verbatim. */
const STORY = [
  'How we got here',
  'I started with one rule: a value call was any game where my model and Vegas disagreed by 4+ percentage points. By midseason that was 64% of all games, and if almost everything is a value call, nothing is.',
  `My first fix split them into ★ and ★★, but the "rare" ★★ tier ended up bigger than ★. So I flipped the approach: I decided how rare stars should be before looking at how many games each cutoff would star (about 1 in 4 games starred, 1 in 10 double-starred), then let about 800 games from past seasons set the cutoffs at 11 and 16 points, using only the size of the disagreements, not who won. Value calls on this site are now ★/★★ only, and every past week was regraded that way. I made that switch after the tier results were already visible, so I'm saying so here.`,
  'The original 4-point rule still runs in the background as a pretend $1 bet on every disagreement of 4+ points. It was locked in before the first official pick to test whether the model can beat the betting market, so I left it alone: changing a test midseason makes the test meaningless. Its results are above.',
  "I missed part of week 8 and all of week 9 during midterms. Those games were never predicted after kickoff; they're marked as missed.",
].join(' ');

describe('build output from the real export (invariants, every week)', () => {
  let b: BuiltSite;
  let d: SiteData;
  beforeAll(() => {
    d = loadReal();
    b = buildOnce('real');
  }, 120_000);
  afterAll(() => b?.cleanup());

  it('builds all 18 week pages, the home page and vs Vegas, nothing else', () => {
    expect([...b.files].sort()).toEqual(['index.html', VS, ...weekPages].sort());
  });

  it('ships no client JS and never the JSON', () => {
    for (const f of b.files) {
      const h = b.html(f);
      expect(h, f).not.toMatch(/<script/i);
      expect(h, f).not.toMatch(/schema_version|model_home_prob|edge_threshold/);
    }
    expect(b.files.some((f) => /\.(js|mjs|json)$/.test(f))).toBe(false);
  });

  it('picks pages never show betting vocabulary (the footer\'s "Not betting advice." aside)', () => {
    const banned = [/\bbets?\b/i, /\bunits?\b/i, /\bROI\b/i, /\bCLV\b/i, /closing.line/i, /interval/i, /\bci95\b/i, /\bwager/i];
    for (const f of b.files.filter((x) => x !== VS)) {
      const t = b.text(f);
      for (const re of banned) expect(t, `${f} ${re}`).not.toMatch(re);
    }
  });

  it('never rounds a percentage to 50%', () => {
    for (const f of b.files) expect(b.text(f), f).not.toMatch(/\b50%/);
  });

  it('has the header, tabs and footer on every picks page', () => {
    for (const f of b.files.filter((x) => x !== VS)) {
      const t = b.text(f);
      expect(t, f).toMatch(/NFL picks/);
      expect(t, f).toMatch(/Week \d+ · 2026/);
      const ts = flat(t);
      expect(ts, f).toMatch(/Picks \d+–\d+ \d+% accuracy Vegas favorite \d+% Confident picks \d+% · Toss-ups \d+%/);
      expect(ts, f).toMatch(
        /Value calls (?:\+\d+\.\d ahead of Vegas|\d+\.\d behind Vegas|Even with Vegas) Record \d+–\d+(?: \(\d+ push(?:es)?\))? Won \d+ · Vegas expected \d+\.\d ★ one-star value calls: \d+ calls? · .+ ★★ two-star value calls: \d+ calls? · [^★]+ Small sample/,
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

  it('header numbers equal the export: picks record, value-call record = ★ + ★★ rows', () => {
    const q = d.live_2026.prediction_quality.champion.accuracy;
    const vc = d.live_2026.value_calls;
    const { one_star, two_star } = vc.tiers;
    const record = `Record ${vc.wins}–${vc.losses}${vc.pushes > 0 ? ` (${vc.pushes} ${vc.pushes === 1 ? 'push' : 'pushes'})` : ''} Won`;
    const calls = (n: number) => `${n} ${n === 1 ? 'call' : 'calls'}`;
    expect([one_star.wins + two_star.wins, one_star.losses + two_star.losses, one_star.pushes + two_star.pushes]).toEqual([vc.wins, vc.losses, vc.pushes]);
    for (const f of b.files.filter((x) => x !== VS)) {
      const t = flat(b.text(f));
      expect(t, f).toContain(`Picks ${q.k}–${q.n - q.k} `);
      expect(t, f).toContain(record);
      expect(t, f).toContain(`Won ${vc.won} · Vegas expected ${vc.vegas_expected.toFixed(1)}`);
      expect(t, f).toContain(`★ one-star value calls: ${calls(one_star.n)} · `);
      expect(t, f).toContain(`★★ two-star value calls: ${calls(two_star.n)} · `);
    }
  });

  it('vs Vegas tab: numbers match the export', () => {
    const L = d.live_2026;
    const { value_calls: vc, paper_bets: pb, picks_vs_vegas: pv } = L;
    const t = flat(b.text(VS));
    const section = (id: string) => flat(textOf(b.html(VS).match(new RegExp(`<section[^>]*id="${id}"[^>]*>([\\s\\S]*?)</section>`))![1]!));
    const pushes = (n: number) => (n > 0 ? ` (${n} ${n === 1 ? 'push' : 'pushes'})` : '');
    const sign = (x: number) => (x > 0 ? '+' : x < 0 ? '−' : '');
    const picks = section('picks');
    expect(picks).toContain('Does the model pick winners as well as Vegas?');
    expect(picks).toContain(`Model ${Math.round(pv.model.value! * 100)}% Vegas favorite ${Math.round(pv.vegas_favorite.value! * 100)}%`);
    expect(picks).toContain(`Over ${pv.games} games with a winner. They picked the same winner in ${pv.same_winner}. In the other ${pv.different_winner.games},`);
    expect(picks).toContain(`the model went ${pv.different_winner.wins}–${pv.different_winner.losses}`);
    const value = section('value-calls');
    expect(value).toContain('When the model strongly disagrees, is it right?');
    expect(value).toContain(`${vc.wins}–${vc.losses}${pushes(vc.pushes)} Won ${vc.won} · Vegas expected ${vc.vegas_expected.toFixed(1)} · `);
    expect(value).toContain(`${vc.tiers.one_star.n} calls`);
    expect(value).toContain(`${vc.tiers.two_star.n} calls`);
    const paper = section('paper-bets');
    expect(paper).toContain('Does the model beat the betting market?');
    expect(paper).toContain(`Paper bets ${pb.wins}–${pb.losses}${pushes(pb.pushes)} Pretend profit ${sign(pb.units)}$${Math.abs(pb.units).toFixed(2)} on $1 bets`);
    expect(paper).toContain(`CLV ${sign(pb.clv.mean!)}${Math.abs(pb.clv.mean! * 100).toFixed(2)} points`);
    expect(paper).toContain(`${pb.clv.n} of ${pb.graded} bets measured`);
    expect(paper).toContain(pb.verdict.so_far);
    expect(paper).toContain(pb.verdict.final ? 'Final verdict: Decided' : 'Final verdict: decided after week 18');
    expect(paper).not.toMatch(/value call/i); // paper bets are never called value calls
    if (pb.pending > 0) expect(paper).toContain(`Not settled yet ${pb.pending}`);
    for (const h of ["What's an edge?", 'Why small samples mislead', 'How we got here']) expect(t).toContain(h);
    expect(t).toContain('vs Vegas · 2026');
    expect(b.html(VS)).not.toMatch(/class="tiles/);
  });

  it('vs Vegas tab: the "How we got here" text, verbatim', () => {
    const story = flat(textOf(b.html(VS).match(/<section[^>]*id="how-we-got-here"[^>]*>([\s\S]*?)<\/section>/)![1]!)).replace(/&quot;|&#34;/g, '"').replace(/&#39;/g, "'");
    expect(story).toBe(STORY);
  });

  it('section tabs on every page; exactly one current page', () => {
    for (const f of b.files) {
      const nav = b.html(f).match(/<nav class="sections[\s\S]*?<\/nav>/)![0];
      expect(nav, f).toMatch(f === VS ? /href="\/vs-vegas\/"[^>]*aria-current="page"/ : /href="\/"[^>]*aria-current="true"/);
      expect(b.html(f).match(/aria-current="page"/g)?.length, f).toBe(1);
    }
    expect(b.html(VS)).toContain('Not betting advice.');
  });

  it('retired words never render anywhere: Lean, slight value, smaller edges, close to the market', () => {
    for (const f of b.files) {
      expect(b.text(f), f).not.toMatch(/\blean\b|slight value|smaller.?edge|close to the market/i);
      expect(b.html(f), f).not.toMatch(/\blean\b|slight value|smaller.?edge/i);
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

  it('every value chip carries stars; no pick chip does', () => {
    for (const f of b.files) {
      for (const card of articles(b.html(f))) {
        expect(/Value call: /.test(card), `${f} untiered value chip`).toBe(false);
        expect(/Pick <span class="stars/.test(card), f).toBe(false);
      }
    }
  });

  it('"value call" never labels a non-starred game (text, screen-reader text or attributes)', () => {
    let n = 0;
    for (let w = 1; w <= 18; w++) {
      const starred = new Set(d.games.filter((g) => g.evaluation === 'live_2026' && g.week === w && (g.value_tier === 'one_star' || g.value_tier === 'two_star')).map((g) => `${g.away} at ${g.home}`));
      for (const [label, card] of cardsOf(b.html(`week/${w}/index.html`))) {
        if (starred.has(label)) continue;
        expect(card, `week ${w} ${label}`).not.toMatch(/value|underrated/i);
        if (/Pick: <strong/.test(card)) {
          expect(card.match(/<li class="chip /g)?.length, `week ${w} ${label}`).toBe(1);
          n++;
        }
      }
    }
    const logged = d.live_2026.prediction_quality_by_confidence.all.games_logged;
    expect(n).toBe(logged - d.live_2026.value_calls.logged);
  });

  it("starred value chips are exactly the export's ★/★★ calls, and their count is one_star + two_star, week by week", () => {
    let total = 0;
    for (let w = 1; w <= 18; w++) {
      const h = b.html(`week/${w}/index.html`);
      const got = valueChips(h);
      expect(got, `week ${w}`).toEqual(exportStarred(d, w));
      const want = d.games.filter((g) => g.evaluation === 'live_2026' && g.week === w && g.bet && (g.value_tier === 'one_star' || g.value_tier === 'two_star')).length;
      const chipCount = (h.match(/>Value call <span class="stars/g) ?? []).length;
      expect(chipCount, `week ${w}`).toBe(want);
      total += chipCount;
    }
    expect(total).toBe(d.live_2026.value_calls.logged);
  });

  it("details sentence: value calls 'underrated', every other game the neutral sentence", () => {
    const L = new Map(d.games.filter((g) => g.evaluation === 'live_2026').map((g) => [`${g.week} ${g.away} at ${g.home}`, g]));
    for (let w = 1; w <= 18; w++) {
      for (const [label, card] of cardsOf(b.html(`week/${w}/index.html`))) {
        const g = L.get(`${w} ${label}`);
        const why = flat(card.match(/<p class="why[^"]*"[^>]*>([\s\S]*?)<\/p>/)?.[1] ?? '');
        if (!g || !g.model_pick) continue;
        const team = g.bet ? (g.bet_side === 'home' ? g.home : g.away) : null;
        if (g.value_tier === 'one_star' || g.value_tier === 'two_star') {
          expect(why, g.game_id).toMatch(new RegExp(`— the model thinks ${team} is underrated\\.$`));
        } else if (g.bet) {
          expect(why, g.game_id).toMatch(new RegExp(`^Model gives ${team} [\\d.]+%, Vegas [\\d.]+%\\.$`));
        } else if (g.vegas_implied_home_prob !== null) {
          expect(why, g.game_id).toMatch(/^Model gives [A-Z]+ [\d.]+%, Vegas [\d.]+%\.$/);
        }
      }
    }
  });

  it('week summary: value part counts only the graded starred chips (★/★★), every week', () => {
    let lineWeeks = 0;
    for (let w = 1; w <= 18; w++) {
      const h = b.html(`week/${w}/index.html`);
      const line = summaryLine(h);
      if (line === null) continue;
      lineWeeks++;
      const m = line.match(/^Picks (\d+)–(\d+)(?: · Value calls ★ (\d+)–(\d+)(?: \((\d+) push(?:es)?\))?)?$/);
      expect(m, `week ${w}: ${line}`).not.toBeNull();
      const chips = starredChipTally(h);
      const graded = chips.w + chips.l + chips.push;
      if (graded === 0) {
        expect(m![3], `week ${w}: ${line}`).toBeUndefined();
      } else {
        expect({ w: Number(m![3]), l: Number(m![4]), push: Number(m![5] ?? 0) }, `week ${w}: ${line}`).toEqual(chips);
      }
      // and the data layer agrees with the rendered chips
      const site = buildSite(d).pages.get(w)!.summary!;
      expect(site.valueCalls ?? { w: 0, l: 0, push: 0 }, `week ${w}`).toEqual(chips);
      // screen readers: the ★ is hidden and words replace it, as on the chips; same numbers
      const spoken = summarySpoken(h)!;
      expect(spoken, `week ${w}`).not.toContain('★');
      expect(spoken, `week ${w}`).toBe(line.replace('Value calls ★', 'Starred value calls'));
      if (graded > 0) {
        expect(summaryHtml(h), `week ${w}`).toMatch(
          /<span aria-hidden="true"[^>]*>Value calls <span class="stars[^"]*"[^>]*>★<\/span><\/span><span class="sr[^"]*"[^>]*>Starred value calls<\/span>/,
        );
      }
    }
    expect(lineWeeks).toBeGreaterThan(0);
  });

  it('a week whose only graded paper bets are non-starred shows no value part', () => {
    const w8 = b.html('week/8/index.html');
    const leanOnly = d.games.filter((g) => g.evaluation === 'live_2026' && g.week === 8 && g.bet);
    expect(leanOnly.length).toBeGreaterThan(0);
    expect(leanOnly.every((g) => g.value_tier === 'lean')).toBe(true);
    expect(summaryLine(w8)).toMatch(/^Picks \d+–\d+$/);
    expect(summarySpoken(w8)).toBe(summaryLine(w8));
  });

  it('TOSS UP pill on exactly the toss-up games, in the top row, pick chip kept', () => {
    for (let w = 1; w <= 18; w++) expect(tossPills(b.html(`week/${w}/index.html`)), `week ${w}`).toEqual(tossUpsIn(d, w));
    const css = b.html('week/3/index.html').replace(/\s+/g, '');
    expect(css).toMatch(/\.toss\[[^\]]*\]\{[^}]*background:var\(--amber-bg\);color:var\(--on-amber\)/);
    expect(css).toMatch(/text-transform:uppercase/);
  });

  it('wide screens: the week strip never scrolls the page past the header', () => {
    const css = b.html('week/3/index.html').replace(/\s+/g, '');
    expect(css).toMatch(/@media\((?:min-width:|width>=)960px\)\{[^@]*scroll-initial-target:none/);
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
      'Value calls +3.6 ahead of Vegas Record 11–7 Won 11 · Vegas expected 7.4 ' +
        '★ one-star value calls: 10 calls · +1.4 ahead (+1.4 per 10 calls) ' +
        '★★ two-star value calls: 8 calls · +2.1 ahead (+2.7 per 10 calls) ' +
        'Small sample',
    );
  });

  it('vs Vegas tab shows the real numbers', () => {
    const t = flat(text(VS));
    expect(t).toContain('Model 59% Vegas favorite 60% Over 75 games with a winner. They picked the same winner in 60. In the other 15,');
    expect(t).toContain('the model went 7–8');
    expect(t).toContain('11–7 Won 11 · Vegas expected 7.4 · +3.6 ahead of Vegas');
    expect(t).toContain('No evidence of an edge so far Final verdict: decided after week 18 Paper bets 28–24 (1 push) Pretend profit +$10.98 on $1 bets');
    expect(t).toContain('8 of 53 bets measured');
    expect(t).toContain('Not settled yet 5');
  });

  it('week summary line on graded weeks only', () => {
    // Value part = graded value calls (★/★★) only.
    const lines = Object.fromEntries([3, 4, 5, 6, 7, 8, 9, 10, 12].map((w) => [w, summaryLine(html(`week/${w}/index.html`))]));
    expect(lines).toEqual({
      3: 'Picks 10–6 · Value calls ★ 4–1',
      4: 'Picks 10–6 · Value calls ★ 2–1',
      5: 'Picks 9–5 · Value calls ★ 3–2', // the week 5 push was a non-starred paper bet: no "(1 push)"
      6: 'Picks 7–7 · Value calls ★ 1–0',
      7: 'Picks 8–6 · Value calls ★ 1–3',
      8: 'Picks 0–1', // its only paper bet (a loss) is non-starred: no value part
      9: null,
      10: null,
      12: null,
    });
    expect(summarySpoken(html('week/7/index.html'))).toBe('Picks 8–6 · Starred value calls 1–3');
    expect(summarySpoken(html('week/8/index.html'))).toBe('Picks 0–1');
  });

  it('chips: pick, ★ and ★★ value calls, with icon and text; other paper bets get no chip', () => {
    const w7 = html('week/7/index.html');
    expect(w7).toMatch(/class="chip win[^"]*"[\s\S]*?<svg[\s\S]*?Pick: <strong[^>]*>[A-Z]{2,3}<\/strong>[\s\S]*?\(correct\)/);
    expect(w7).toMatch(/class="chip loss[^"]*"[\s\S]*?<svg[\s\S]*?\(wrong\)/);
    const t7 = flat(text('week/7/index.html'));
    expect(t7).toContain('Value call ★ one-star value call : NE (wrong)');
    expect(t7).toContain('Value call ★★ two-star value call : MIA (correct)');
    // week 7's non-starred paper bets (MIN ✓, NO ✗): the neutral sentence only, no chip
    expect(t7).toMatch(/Model gives MIN \d+%, Vegas \d+%\. /);
    expect(t7).toMatch(/Model gives NO \d+%, Vegas \d+%\. /);
    expect(t7).not.toMatch(/Value call [^:]*: (?:MIN|NO)\b/);
    expect(w7).not.toMatch(/class="chip[^"]*\blean\b/);
    const w5 = flat(text('week/5/index.html'));
    expect(w5).toMatch(/Pick: WAS · Tie/);
    expect(w5).not.toMatch(/NYG · Push/); // the week 5 push was a non-starred paper bet: no chip
    expect(w5).toMatch(/Model gives NYG \d+%, Vegas \d+%\. /);
    expect(html('week/10/index.html')).toMatch(/class="chip open[^"]*"/);
  });

  it('week 3: starred chips match the fixture; its six non-starred paper bets have no chip', () => {
    expect(valueChips(html('week/3/index.html'))).toEqual({
      one_star: ['LA at DEN DEN', 'PHI at CHI CHI', 'TEN at NYG NYG'],
      two_star: ['CIN at PIT PIT', 'KC at MIA MIA'],
    });
    expect(exportTiers(d, 3).lean).toEqual(['ATL at GB GB', 'CAR at CLE CLE', 'HOU at IND HOU', 'LAC at BUF BUF', 'LV at NO NO', 'NYJ at DET DET']);
    const w3 = new Map(cardsOf(html('week/3/index.html')));
    for (const g of ['ATL at GB', 'CAR at CLE', 'HOU at IND', 'LAC at BUF', 'LV at NO', 'NYJ at DET']) {
      expect(w3.get(g)!.match(/<li class="chip /g)?.length, g).toBe(1);
      expect(w3.get(g), g).toMatch(/Model gives [A-Z]+ \d+%, Vegas \d+%\.</);
    }
    for (const w of [3, 5, 7, 10]) expect(valueChips(html(`week/${w}/index.html`)), `week ${w}`).toEqual(exportStarred(d, w));
  });

  it('week 10 NE at DET: one decimal everywhere, details bars and sentence', () => {
    const w10 = flat(text('week/10/index.html'));
    expect(w10).toContain('Pick: DET');
    expect(w10).not.toMatch(/Value call [^:]*: NE\b/); // NE is a non-starred paper bet: no chip
    expect(w10).toContain('Model NE 49.6% DET 50.4%');
    expect(w10).toContain('Vegas NE 41.7% DET 58.3%');
    expect(w10).toContain('Model gives NE 49.6%, Vegas 41.7%.');
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
