import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { REAL_PATH, loadReal } from './exports.ts';
import { parseSiteData } from '../src/lib/schema.ts';

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- deliberately mutated into invalid shapes
const raw = (): any => JSON.parse(readFileSync(REAL_PATH, 'utf8'));

describe('schema', () => {
  it('parses the real export', () => {
    const d = loadReal();
    expect(d.schema_version).toMatch(/^1\.\d+\.\d+$/);
    expect(d.schedule_2026.length).toBeGreaterThan(250);
    expect(d.live_2026.edge_threshold).toBe(0.04);
    expect(d.games.some((g) => g.evaluation === 'live_2026')).toBe(true);
  });

  it('rejects a schema 2.x export', () => {
    const d = raw();
    d['schema_version'] = '2.0.0';
    expect(() => parseSiteData(d)).toThrow(/schema 1\.x/);
  });

  it('rejects a 1.1 export without schedule_2026', () => {
    const d = raw();
    delete d['schedule_2026'];
    expect(() => parseSiteData(d)).toThrow(/schedule_2026/);
  });

  it('rejects a live game that is not on the schedule', () => {
    const d = raw();
    const g = d['games'].find((x: { evaluation: string }) => x.evaluation === 'live_2026');
    g.game_id = '2026_10_XXX_YYY';
    expect(() => parseSiteData(d)).toThrow(/does not match schedule_2026/);
  });

  it('rejects an export without the Amendment 8 blocks (1.4.x and older)', () => {
    for (const k of ['value_calls', 'paper_bets', 'picks_vs_vegas']) {
      const d = raw();
      delete d['live_2026'][k];
      expect(() => parseSiteData(d)).toThrow(new RegExp(k));
    }
  });

  it('rejects a live game missing its toss-up flag, or a tier out of step with the paper bet', () => {
    const d = raw();
    const g = d['games'].find((x: { evaluation: string }) => x.evaluation === 'live_2026');
    delete g.toss_up;
    expect(() => parseSiteData(d)).toThrow(/toss_up/);
    const e = raw();
    const b = e['games'].find((x: { evaluation: string; bet: boolean }) => x.evaluation === 'live_2026' && x.bet);
    b.value_tier = null;
    expect(() => parseSiteData(e)).toThrow(/value_tier/);
  });

  it('rejects a 1.3 export: numeric tiers, no non-starred block, no one-star cutoff', () => {
    const d = raw();
    const b = d['games'].find((x: { evaluation: string; bet: boolean }) => x.evaluation === 'live_2026' && x.bet);
    b.value_tier = 2;
    expect(() => parseSiteData(d)).toThrow(/value_tier/);
    const e = raw();
    delete e['live_2026'].paper_bets.non_starred;
    expect(() => parseSiteData(e)).toThrow(/paper_bets\.non_starred/);
    const f = raw();
    delete f['live_2026'].value_tier_one_star_edge;
    expect(() => parseSiteData(f)).toThrow(/value_tier_one_star_edge/);
  });

  it('accepts exactly the three Amendment 7 tiers, with cutoffs in order', () => {
    const d = loadReal();
    const tiers = new Set(d.games.filter((g) => g.evaluation === 'live_2026' && g.bet).map((g) => g.value_tier));
    expect([...tiers].sort()).toEqual(['lean', 'one_star', 'two_star']);
    const e = raw();
    e['live_2026'].value_tier_one_star_edge = 0.2;
    expect(() => parseSiteData(e)).toThrow(/cutoffs out of order/);
  });

  it('rejects a missing edge threshold', () => {
    const d = raw();
    delete d['live_2026'].edge_threshold;
    expect(() => parseSiteData(d)).toThrow(/edge_threshold/);
  });
});
