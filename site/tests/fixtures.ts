import type { Game, ScheduleGame } from '../src/lib/schema.ts';

export function sched(over: Partial<ScheduleGame> = {}): ScheduleGame {
  return {
    game_id: '2026_10_DEN_KC',
    week: 10,
    kickoff_utc: '2026-11-15T18:00:00Z',
    away: 'DEN',
    home: 'KC',
    away_score: null,
    home_score: null,
    ...over,
  };
}

export function live(over: Partial<Game> = {}): Game {
  return {
    evaluation: 'live_2026',
    season: 2026,
    week: 10,
    game_id: '2026_10_DEN_KC',
    away: 'DEN',
    home: 'KC',
    kickoff_utc: '2026-11-15T18:00:00Z',
    model_home_prob: 0.64,
    vegas_implied_home_prob: 0.74,
    model_pick: 'home',
    result: 'pending',
    away_score: null,
    home_score: null,
    pick_correct: null,
    bet: false,
    bet_side: null,
    ...over,
  };
}
