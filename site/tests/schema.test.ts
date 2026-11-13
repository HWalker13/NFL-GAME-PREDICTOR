import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { dataPath, loadSiteData } from '../src/lib/data.ts';
import { parseSiteData } from '../src/lib/schema.ts';

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- deliberately mutated into invalid shapes
const raw = (): any => JSON.parse(readFileSync(dataPath(), 'utf8'));

describe('schema', () => {
  it('parses the real export', () => {
    const d = loadSiteData();
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

  it('rejects a missing edge threshold', () => {
    const d = raw();
    delete d['live_2026'].edge_threshold;
    expect(() => parseSiteData(d)).toThrow(/edge_threshold/);
  });
});
