import { describe, expect, it } from 'vitest';
import { buildCard, detailsSentence, gradeSide, meetsThreshold, type Card } from '../src/lib/card.ts';
import { buildSite, crossCheckValueCalls, loadSiteData, valueCallStats } from '../src/lib/data.ts';
import { kickoffEt, pct, primetime } from '../src/lib/format.ts';
import { live, sched } from './fixtures.ts';

const NOW = new Date('2026-11-20T00:00:00Z'); // after every fixture kickoff
const T = 0.04;
const final = (away_score: number, home_score: number) => sched({ away_score, home_score });

describe('pick grade matches the pick and the score', () => {
  it('home pick, home wins: win', () => {
    expect(buildCard(final(17, 24), live({ model_pick: 'home' }), T, NOW).pick?.grade).toBe('win');
  });
  it('home pick, away wins: loss', () => {
    expect(buildCard(final(27, 20), live({ model_pick: 'home' }), T, NOW).pick?.grade).toBe('loss');
  });
  it('away pick, away wins: win', () => {
    expect(buildCard(final(27, 20), live({ model_pick: 'away', model_home_prob: 0.4 }), T, NOW).pick?.grade).toBe('win');
  });
  it('away pick, home wins: loss', () => {
    expect(buildCard(final(17, 24), live({ model_pick: 'away', model_home_prob: 0.4 }), T, NOW).pick?.grade).toBe('loss');
  });
  it('tie: push grade (rendered as "Tie" on the pick chip)', () => {
    expect(buildCard(final(20, 20), live(), T, NOW).pick?.grade).toBe('push');
  });
  it('not final: open', () => {
    expect(buildCard(sched(), live(), T, NOW).pick?.grade).toBe('open');
  });
  it('no pick: final status, no chips, no details data', () => {
    const c = buildCard(final(17, 24), undefined, T, NOW);
    expect([c.status, c.pick, c.valueCall, c.probs]).toEqual(['final', null, null, null]);
  });

  it('agrees with the exporter on every graded live game', () => {
    const d = loadSiteData();
    const sch = new Map(d.schedule_2026.map((g) => [g.game_id, g]));
    const graded = d.games.filter((g) => g.evaluation === 'live_2026' && g.pick_correct !== null);
    expect(graded.length).toBeGreaterThan(50);
    for (const g of graded) {
      const c = buildCard(sch.get(g.game_id)!, g, d.live_2026.edge_threshold, new Date(d.live_2026.as_of_utc));
      expect(c.pick?.grade, g.game_id).toBe(g.pick_correct ? 'win' : 'loss');
    }
  });
});

describe('value call', () => {
  const callFor = (p: number, v: number) =>
    buildCard(sched(), live({ model_home_prob: p, vegas_implied_home_prob: v, model_pick: p > 0.5 ? 'home' : 'away' }), T, NOW).valueCall;

  it('goes to the underdog the model picked to lose when it beats Vegas on it', () => {
    // model KC 64% vs Vegas 74% -> model DEN 36% vs Vegas 26%
    expect(callFor(0.64, 0.74)).toMatchObject({ side: 'away', team: 'DEN', modelProb: expect.closeTo(0.36, 10), vegasProb: expect.closeTo(0.26, 10) });
  });
  it('can be the favorite', () => {
    expect(callFor(0.7, 0.6)?.team).toBe('KC');
  });
  it('appears at exactly the threshold', () => {
    expect(callFor(0.64, 0.6)?.team).toBe('KC');
    expect(callFor(0.56, 0.6)?.team).toBe('DEN');
  });
  it('does not appear just below it', () => {
    expect(callFor(0.6399, 0.6)).toBeNull();
    expect(callFor(0.5601, 0.6)).toBeNull();
  });
  it('no moneyline: no value call', () => {
    expect(buildCard(sched(), live({ vegas_implied_home_prob: null }), T, NOW).valueCall).toBeNull();
  });
  it('compares like the exporter (10-decimal rounding)', () => {
    expect(meetsThreshold(0.64 - 0.6, 0.04)).toBe(true); // 0.040000000000000036 in floats
    expect(meetsThreshold(0.0399999999, 0.04)).toBe(false);
  });

  describe('grading', () => {
    const lv = live({ model_home_prob: 0.64, vegas_implied_home_prob: 0.74 }); // value call: DEN (away)
    it('✓ when the value-call team wins', () => {
      expect(buildCard(final(27, 20), lv, T, NOW).valueCall?.grade).toBe('win');
    });
    it('✗ when it loses, even though the pick was right', () => {
      const c = buildCard(final(17, 24), lv, T, NOW);
      expect([c.pick?.grade, c.valueCall?.grade]).toEqual(['win', 'loss']);
    });
    it('push on a tie', () => {
      expect(buildCard(final(20, 20), lv, T, NOW).valueCall?.grade).toBe('push');
    });
    it('open until final', () => {
      expect(buildCard(sched(), lv, T, NOW).valueCall?.grade).toBe('open');
    });
    it('gradeSide covers every case', () => {
      expect(['home', 'away', 'tie', null].map((w) => gradeSide('home', w as never))).toEqual(['win', 'loss', 'push', 'open']);
    });
  });

  it("matches the exporter's flag and side on every real live game", () => {
    const d = loadSiteData();
    const sch = new Map(d.schedule_2026.map((g) => [g.game_id, g]));
    const games = d.games.filter((g) => g.evaluation === 'live_2026');
    for (const g of games) {
      const c = buildCard(sch.get(g.game_id)!, g, d.live_2026.edge_threshold, new Date(d.live_2026.as_of_utc));
      expect(c.valueCall?.side ?? null, g.game_id).toBe(g.bet ? g.bet_side : null);
    }
    expect(games.filter((g) => g.bet).length).toBeGreaterThan(0);
  });

  it('season value-call record (W-L-push) equals the official record', () => {
    const d = loadSiteData();
    const vc = buildSite(d).record.valueCalls;
    const o = d.live_2026.official_bets;
    expect([vc.w, vc.l, vc.push]).toEqual([o.wins, o.losses, o.voids]);
    expect(vc.push).toBeGreaterThan(0); // the real data has a tie (week 5 NYG at WAS)
  });

  it('the cross-check fails the build on a side mismatch or a record mismatch', () => {
    const d = loadSiteData();
    const site = buildSite(d);
    const cards = new Map([...site.pages.values()].flatMap((p) => p.cards).map((c) => [c.gameId, c]));
    const vc = site.record.valueCalls;
    expect(() => crossCheckValueCalls(d, cards, vc)).not.toThrow();
    expect(() => crossCheckValueCalls(d, cards, { ...vc, w: vc.w + 1 })).toThrow(/official/);
    const g = d.games.find((x) => x.evaluation === 'live_2026' && x.bet)!;
    const flipped = { ...d, games: d.games.map((x) => (x === g ? { ...x, bet_side: x.bet_side === 'home' ? 'away' as const : 'home' as const } : x)) };
    expect(() => crossCheckValueCalls(flipped, cards, vc)).toThrow(g.game_id);
    const dropped = { ...d, games: d.games.map((x) => (x === g ? { ...x, bet: false, bet_side: null } : x)) };
    expect(() => buildSite(dropped)).toThrow(/value calls disagree/);
  });
});

describe('Vegas expected', () => {
  // Four value calls on a fixture: two wins, a loss, a push and one still open.
  const mk = (id: string, p: number, v: number, score: [number, number] | null): Card =>
    buildCard(
      sched({ game_id: id, ...(score ? { away_score: score[0], home_score: score[1] } : {}) }),
      live({ game_id: id, model_home_prob: p, vegas_implied_home_prob: v, model_pick: p > 0.5 ? 'home' : 'away' }),
      T,
      NOW,
    );
  const cards = [
    mk('a', 0.7, 0.6, [10, 20]), // KC call, Vegas 0.60, KC wins
    mk('b', 0.64, 0.74, [27, 20]), // DEN call, Vegas 0.26, DEN wins
    mk('c', 0.55, 0.45, [21, 14]), // KC call, Vegas 0.45, KC loses
    mk('d', 0.6, 0.5, [17, 17]), // KC call, tie: push, excluded
    mk('e', 0.6, 0.5, null), // not final: excluded
    mk('f', 0.6, 0.59, [3, 0]), // under the threshold: no call
  ];

  it('averages Vegas no-vig probability for the call side over graded calls, pushes excluded', () => {
    const s = valueCallStats(cards);
    expect([s.w, s.l, s.push]).toEqual([2, 1, 1]);
    expect(s.hitRate).toBeCloseTo(2 / 3, 10);
    expect(s.vegasExpected).toBeCloseTo((0.6 + 0.26 + 0.45) / 3, 10);
  });
  it('is null before anything is graded', () => {
    expect(valueCallStats([cards[4]!])).toMatchObject({ w: 0, l: 0, push: 0, hitRate: null, vegasExpected: null });
  });
});

describe('details sentence', () => {
  const p = (x: number) => pct(x);
  it('names the value-call team as underrated', () => {
    const c = buildCard(sched({ away: 'CHI' }), live({ away: 'CHI', model_home_prob: 0.51, vegas_implied_home_prob: 0.66 }), T, NOW);
    expect(detailsSentence(c, p)).toBe('Model gives CHI 49%, Vegas 34% — the model thinks CHI is underrated.');
  });
  it('under the threshold: close to the market', () => {
    const c = buildCard(sched(), live({ model_home_prob: 0.62, vegas_implied_home_prob: 0.6 }), T, NOW);
    expect(detailsSentence(c, p)).toBe('Model gives KC 62%, Vegas 60% — close to the market.');
  });
  it('no moneyline', () => {
    const c = buildCard(sched(), live({ vegas_implied_home_prob: null }), T, NOW);
    expect(detailsSentence(c, p)).toBe('Model gives KC 64%. No Vegas odds were saved for this game.');
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
    expect(c.valueCall?.team).toBe('NE');
    expect(pct(c.valueCall!.modelProb, c.decimals)).toBe('49.6%');
    expect(pct(c.valueCall!.vegasProb, c.decimals)).toBe('41.7%');
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

describe('primetime badge', () => {
  it('Thursday night: TNF', () => {
    expect(primetime('2026-11-13T01:15:00Z')).toBe('TNF'); // Thu 8:15 PM ET (EST)
    expect(primetime('2026-09-25T00:15:00Z')).toBe('TNF'); // Thu 8:15 PM ET (EDT)
  });
  it('Thanksgiving afternoon: none', () => {
    expect(primetime('2026-11-26T17:30:00Z')).toBeNull(); // Thu 12:30 PM ET
  });
  it('Sunday at or after 7 PM ET: SNF', () => {
    expect(primetime('2026-11-16T01:20:00Z')).toBe('SNF'); // Sun 8:20 PM ET
    expect(primetime('2026-11-16T00:00:00Z')).toBe('SNF'); // Sun 7:00 PM ET exactly
  });
  it('Sunday 4:25 PM: none', () => {
    expect(primetime('2026-11-15T21:25:00Z')).toBeNull();
  });
  it('Sunday early and international games: none', () => {
    expect(primetime('2026-11-15T18:00:00Z')).toBeNull();
    expect(primetime('2026-11-15T14:30:00Z')).toBeNull(); // 9:30 AM ET
  });
  it('Monday: MNF', () => {
    expect(primetime('2026-11-17T01:15:00Z')).toBe('MNF'); // Mon 8:15 PM ET
    expect(primetime('2026-09-29T00:15:00Z')).toBe('MNF');
  });
  it('lands on the card', () => {
    expect(buildCard(sched({ kickoff_utc: '2026-11-17T01:15:00Z' }), undefined, T, NOW).primetime).toBe('MNF');
  });
});
