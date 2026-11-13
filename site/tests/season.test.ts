import { describe, expect, it } from 'vitest';
import { buildSite, loadSiteData } from '../src/lib/data.ts';
import { currentWeek, weekBanner, weekInfo } from '../src/lib/season.ts';
import { buildCard } from '../src/lib/card.ts';
import { sched } from './fixtures.ts';

const played = { away_score: 17, home_score: 24 };

describe('currentWeek', () => {
  it('is the earliest week with a game not yet final', () => {
    const s = [
      sched({ game_id: 'a', week: 1, ...played }),
      sched({ game_id: 'b', week: 2, ...played }),
      sched({ game_id: 'c', week: 3, ...played }),
      sched({ game_id: 'd', week: 3 }), // Monday night still to play
      sched({ game_id: 'e', week: 4 }),
    ];
    expect(currentWeek(s)).toBe(3);
  });

  it('moves on once the last game of a week is final', () => {
    const s = [sched({ game_id: 'a', week: 3, ...played }), sched({ game_id: 'b', week: 4 })];
    expect(currentWeek(s)).toBe(4);
  });

  it('stays on week 18 after the season', () => {
    expect(currentWeek([sched({ game_id: 'a', week: 18, ...played })])).toBe(18);
  });

  it('picks week 10 from the real export (scores through week 9)', () => {
    const site = buildSite(loadSiteData());
    expect(site.current).toBe(10);
  });
});

describe('a fixed "now" decides only upcoming vs score pending', () => {
  const g = sched({ kickoff_utc: '2026-11-15T18:00:00Z' });
  it('before kickoff: not played yet', () => {
    expect(buildCard(g, undefined, 0.04, new Date('2026-11-15T17:59:59Z')).status).toBe('upcoming');
  });
  it('after kickoff with no score in the data: score pending', () => {
    expect(buildCard(g, undefined, 0.04, new Date('2026-11-15T18:00:00Z')).status).toBe('score-pending');
  });
  it('the build uses the snapshot time, not the wall clock', () => {
    const d = loadSiteData();
    const site = buildSite(d);
    const first = site.pages.get(10)?.cards.find((c) => c.gameId === '2026_10_WAS_NYG');
    // kicked off 2026-11-13T01:15Z, after the 2026-11-12T21:09Z snapshot
    expect(d.live_2026.as_of_utc).toBe('2026-11-12T21:09:40Z');
    expect(first?.status).toBe('upcoming');
  });
});

describe('week kinds and banners', () => {
  const d = loadSiteData();
  const site = buildSite(d);
  const info = (w: number) => weekInfo(w, d.live_2026.first_official_week, d.live_2026.weeks, d.schedule_2026);

  it('classifies the real season', () => {
    expect([1, 2].map((w) => info(w).kind)).toEqual(['before-tracking', 'before-tracking']);
    expect([3, 4, 5, 6, 7].map((w) => info(w).kind)).toEqual(Array(5).fill('tracked'));
    expect(info(8).kind).toBe('partial');
    expect(info(9).kind).toBe('missed');
    expect(info(10).kind).toBe('tracked');
    expect(info(11).kind).toBe('future');
  });

  it('writes the approved banners, reading numbers from the data', () => {
    expect(site.pages.get(1)?.banner).toBe(`Tracking started in week ${d.live_2026.first_official_week}.`);
    expect(site.pages.get(8)?.banner).toBe('Picks were logged for 1 of 14 games this week.');
    expect(site.pages.get(9)?.banner).toBe('No picks were logged this week.');
    expect(site.pages.get(10)?.banner).toBeNull();
    expect(site.pages.get(11)?.banner).toBe("Picks will appear here once they're logged.");
    expect(weekBanner({ week: 3, kind: 'before-tracking', scheduled: 16, predicted: 0 }, 4)).toBe(
      'Tracking started in week 4.',
    );
  });

  it('has a card for every scheduled game and a pick for every logged one', () => {
    for (const w of site.weeks) {
      const page = site.pages.get(w.week);
      expect(page?.cards.length).toBe(w.scheduled);
      expect(page?.cards.filter((c) => c.pick).length).toBe(w.kind === 'before-tracking' ? 0 : w.predicted);
    }
  });
});
