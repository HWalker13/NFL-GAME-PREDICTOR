import { describe, expect, it } from 'vitest';
import { buildCard, meetsThreshold } from '../src/lib/card.ts';
import { loadSiteData } from '../src/lib/data.ts';
import { kickoffEt, pct } from '../src/lib/format.ts';
import { live, sched } from './fixtures.ts';

const NOW = new Date('2026-11-20T00:00:00Z'); // after every fixture kickoff
const T = 0.04;

describe('status matches the pick and the score', () => {
  const final = (away_score: number, home_score: number) => sched({ away_score, home_score });

  it('home pick, home wins: correct', () => {
    expect(buildCard(final(17, 24), live({ model_pick: 'home' }), T, NOW).status).toBe('correct');
  });
  it('home pick, away wins: wrong', () => {
    expect(buildCard(final(27, 20), live({ model_pick: 'home' }), T, NOW).status).toBe('wrong');
  });
  it('away pick, away wins: correct', () => {
    expect(buildCard(final(27, 20), live({ model_pick: 'away', model_home_prob: 0.4 }), T, NOW).status).toBe('correct');
  });
  it('away pick, home wins: wrong', () => {
    expect(buildCard(final(17, 24), live({ model_pick: 'away', model_home_prob: 0.4 }), T, NOW).status).toBe('wrong');
  });
  it('tie: neither', () => {
    expect(buildCard(final(20, 20), live(), T, NOW).status).toBe('tie');
  });
  it('no pick: final', () => {
    expect(buildCard(final(17, 24), undefined, T, NOW).status).toBe('final');
  });

  it('agrees with the exporter on every graded live game', () => {
    const d = loadSiteData();
    const sch = new Map(d.schedule_2026.map((g) => [g.game_id, g]));
    const graded = d.games.filter((g) => g.evaluation === 'live_2026' && g.pick_correct !== null);
    expect(graded.length).toBeGreaterThan(50);
    for (const g of graded) {
      const c = buildCard(sch.get(g.game_id)!, g, d.live_2026.edge_threshold, new Date(d.live_2026.as_of_utc));
      expect(c.status, g.game_id).toBe(g.pick_correct ? 'correct' : 'wrong');
    }
  });
});

describe('pick and Vegas lines', () => {
  it('shows the picked team, model % and Vegas % for the same team', () => {
    const c = buildCard(sched(), live({ model_pick: 'away', model_home_prob: 0.36, vegas_implied_home_prob: 0.26 }), T, NOW);
    expect(c.away.picked).toBe(true);
    expect(c.home.picked).toBe(false);
    expect(c.pick?.team).toBe('DEN');
    expect(pct(c.pick!.modelProb)).toBe('64%');
    expect(pct(c.pick!.vegasProb!)).toBe('74%');
  });
  it('no moneyline: Vegas prob is null and no flag', () => {
    const c = buildCard(sched(), live({ vegas_implied_home_prob: null }), T, NOW);
    expect(c.pick?.vegasProb).toBeNull();
    expect(c.flag).toBeNull();
  });
});

describe('"rates higher than Vegas" line', () => {
  const flagFor = (p: number, v: number) =>
    buildCard(sched(), live({ model_home_prob: p, vegas_implied_home_prob: v, model_pick: p > 0.5 ? 'home' : 'away' }), T, NOW).flag;

  it('appears for the away team when the model is 4+ points above Vegas on it', () => {
    // model KC 64% vs Vegas 74% -> model DEN 36% vs Vegas 26%
    expect(flagFor(0.64, 0.74)).toEqual({ team: 'DEN', modelProb: expect.closeTo(0.36, 10), vegasProb: expect.closeTo(0.26, 10) });
  });
  it('appears for the home team', () => {
    expect(flagFor(0.7, 0.6)?.team).toBe('KC');
  });
  it('appears at exactly the threshold', () => {
    expect(flagFor(0.64, 0.6)?.team).toBe('KC');
    expect(flagFor(0.56, 0.6)?.team).toBe('DEN');
  });
  it('does not appear just below it', () => {
    expect(flagFor(0.6399, 0.6)).toBeNull();
    expect(flagFor(0.5601, 0.6)).toBeNull();
  });
  it('compares like the exporter (10-decimal rounding)', () => {
    expect(meetsThreshold(0.64 - 0.6, 0.04)).toBe(true); // 0.040000000000000036 in floats
    expect(meetsThreshold(0.0399999999, 0.04)).toBe(false);
  });

  it("appears on exactly the real live games the exporter's rule flagged", () => {
    const d = loadSiteData();
    const sch = new Map(d.schedule_2026.map((g) => [g.game_id, g]));
    const games = d.games.filter((g) => g.evaluation === 'live_2026');
    for (const g of games) {
      const c = buildCard(sch.get(g.game_id)!, g, d.live_2026.edge_threshold, new Date(d.live_2026.as_of_utc));
      expect(c.flag !== null, g.game_id).toBe(g.bet);
    }
    expect(games.filter((g) => g.bet).length).toBeGreaterThan(0);
  });
});

describe('one decimal when a percentage would read as 50%', () => {
  it('week 10 NE at DET: every percentage on the card gets one decimal', () => {
    const d = loadSiteData();
    const g = d.games.find((x) => x.game_id === '2026_10_NE_DET')!;
    const s = d.schedule_2026.find((x) => x.game_id === '2026_10_NE_DET')!;
    expect([g.model_home_prob, g.vegas_implied_home_prob]).toEqual([0.5037, 0.583]);
    const c = buildCard(s, g, d.live_2026.edge_threshold, new Date(d.live_2026.as_of_utc));
    expect(c.decimals).toBe(1);
    expect(c.pick?.team).toBe('DET');
    expect(pct(c.pick!.modelProb, c.decimals)).toBe('50.4%');
    expect(pct(c.pick!.vegasProb!, c.decimals)).toBe('58.3%');
    expect(c.flag?.team).toBe('NE');
    expect(pct(c.flag!.modelProb, c.decimals)).toBe('49.6%');
    expect(pct(c.flag!.vegasProb, c.decimals)).toBe('41.7%');
  });

  it('a 50% Vegas number alone also switches the card', () => {
    const c = buildCard(sched(), live({ model_home_prob: 0.6, vegas_implied_home_prob: 0.503 }), T, NOW);
    expect(c.decimals).toBe(1);
  });

  it('other cards stay whole numbers', () => {
    const c = buildCard(sched(), live({ model_home_prob: 0.64, vegas_implied_home_prob: 0.74 }), T, NOW);
    expect(c.decimals).toBe(0);
    expect(pct(0.494)).toBe('49%');
    expect(pct(0.506)).toBe('51%');
  });
});

describe('format', () => {
  it('kickoff in US Eastern, standard and daylight time', () => {
    expect(kickoffEt('2026-11-15T18:00:00Z')).toBe('Sun 1:00 PM ET');
    expect(kickoffEt('2026-09-13T17:00:00Z')).toBe('Sun 1:00 PM ET');
    expect(kickoffEt('2026-11-17T01:15:00Z')).toBe('Mon 8:15 PM ET');
  });
});
