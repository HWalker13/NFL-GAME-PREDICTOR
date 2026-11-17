/*
  Zod schema for the parts of exports/site_data.json (schema 1.x, field docs in exports/SCHEMA.md)
  that this site reads. Unknown keys are dropped, so the export can grow without breaking the
  build; a field the site uses that disappears or changes type fails the build instead of
  rendering blanks. `schedule_2026` and `live_2026.edge_threshold` arrived in 1.2.0; the
  Amendment 6 fields (`value_tier`, `toss_up`, `value_woe`, `prediction_quality_by_confidence`,
  `toss_up_threshold`, `value_tier_two_star_edge`) in 1.3.0; Amendment 7's three tiers (string
  `value_tier`, `value_woe.lean`, `value_tier_one_star_edge`) in 1.4.0. The official record
  (`live_2026.official_bets`) and `games[].bet_side` are read only to cross-check the value calls;
  they're never rendered.
*/
import { z } from 'zod';

const isoUtc = z.string().regex(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/, 'expected UTC ISO-8601 with a Z suffix');
const sha = z.string().regex(/^[0-9a-f]{40}$/, 'expected a full 40-character git hash');
const team = z.string().regex(/^[A-Z]{2,3}$/, 'expected a team abbreviation');
const week = z.number().int().min(1).max(18);
const count = z.number().int().min(0);
const prob = z.number().min(0).max(1);
const score = count.nullable();
const side = z.enum(['home', 'away']);

const rate = z.object({ k: count, n: count, value: prob.nullable() });
/** Amendment 7 display tier of an official value call. */
const tier = z.enum(['lean', 'one_star', 'two_star']);

export const scheduleGameSchema = z.object({
  game_id: z.string(),
  week,
  kickoff_utc: isoUtc,
  away: team,
  home: team,
  away_score: score,
  home_score: score,
});

export const gameSchema = z.object({
  evaluation: z.string(),
  season: z.number().int(),
  week: z.number().int(),
  game_id: z.string(),
  away: team,
  home: team,
  kickoff_utc: isoUtc,
  model_home_prob: prob,
  vegas_implied_home_prob: prob.nullable(),
  model_pick: side,
  result: z.enum(['home', 'away', 'tie', 'pending']),
  away_score: score,
  home_score: score,
  pick_correct: z.boolean().nullable(),
  bet: z.boolean(),
  bet_side: side.nullable(),
  /** Live games only (1.4.0): the value call's tier, decided by the exporter at full precision. */
  value_tier: tier.nullable().optional(),
  /** Live games only (1.3.0): the champion's favorite is at or below toss_up_threshold. */
  toss_up: z.boolean().optional(),
});

/** The official paper record's outcome counts; the site's graded value calls must equal these. */
const officialRecord = z.object({ wins: count, losses: count, voids: count, pending: count });

/** Wins over expected for one tier (or all calls). Sums are over graded calls, pushes excluded. */
const woeBlock = z.object({
  calls_logged: count,
  wins: count,
  losses: count,
  pushes: count,
  n: count,
  won: count,
  vegas_expected: z.number().min(0),
  woe: z.number(),
  woe_per_call: z.number().nullable(),
});

const byConfidence = z.object({ games_logged: count, accuracy: rate });

const weekSchema = z.object({
  week,
  status: z.enum(['graded', 'partial', 'missed', 'pending']),
  scheduled: count,
  predicted: count,
});

export const siteDataSchema = z
  .object({
    schema_version: z.string().regex(/^1\.\d+\.\d+$/, 'expected schema 1.x'),
    generated_at: isoUtc,
    source_commit: z.object({ hash: sha, dirty: z.boolean() }),
    live_2026: z.object({
      as_of_utc: isoUtc,
      first_official_week: week,
      edge_threshold: z.number().gt(0).lt(1),
      value_tier_one_star_edge: z.number().gt(0).lt(1),
      value_tier_two_star_edge: z.number().gt(0).lt(1),
      toss_up_threshold: z.number().gt(0.5).lt(1),
      value_woe: z.object({ all: woeBlock, lean: woeBlock, one_star: woeBlock, two_star: woeBlock }),
      prediction_quality_by_confidence: z.object({ confident: byConfidence, toss_up: byConfidence, all: byConfidence }),
      prediction_quality: z.object({
        champion: z.object({ accuracy: rate }),
        vegas_favorite: z.object({ accuracy: rate }),
      }),
      official_bets: officialRecord,
      weeks: z.array(weekSchema),
    }),
    schedule_2026: z.array(scheduleGameSchema).min(1),
    games: z.array(gameSchema),
  })
  .superRefine((d, ctx) => {
    const L = d.live_2026;
    if (!(L.edge_threshold < L.value_tier_one_star_edge && L.value_tier_one_star_edge < L.value_tier_two_star_edge)) {
      ctx.addIssue({ code: 'custom', path: ['live_2026'], message: 'tier cutoffs out of order (edge_threshold < one-star < two-star)' });
    }
    const weeks = new Set(d.schedule_2026.map((g) => g.week));
    for (let w = 1; w <= 18; w++) {
      if (!weeks.has(w)) ctx.addIssue({ code: 'custom', path: ['schedule_2026'], message: `no games in week ${w}` });
    }
    const byId = new Map(d.schedule_2026.map((g) => [g.game_id, g]));
    if (byId.size !== d.schedule_2026.length) {
      ctx.addIssue({ code: 'custom', path: ['schedule_2026'], message: 'duplicate game_id' });
    }
    for (const g of d.games) {
      if (g.evaluation !== 'live_2026') continue;
      const s = byId.get(g.game_id);
      if (!s || s.week !== g.week || s.away !== g.away || s.home !== g.home) {
        ctx.addIssue({ code: 'custom', path: ['games'], message: `live game ${g.game_id} does not match schedule_2026` });
      }
      if (g.bet !== (g.bet_side !== null)) {
        ctx.addIssue({ code: 'custom', path: ['games'], message: `live game ${g.game_id} has bet/bet_side out of step` });
      }
      if (g.toss_up === undefined || g.value_tier === undefined || g.bet !== (g.value_tier !== null)) {
        ctx.addIssue({ code: 'custom', path: ['games'], message: `live game ${g.game_id} is missing toss_up/value_tier or has value_tier out of step with bet` });
      }
    }
    for (const s of d.schedule_2026) {
      if ((s.away_score === null) !== (s.home_score === null)) {
        ctx.addIssue({ code: 'custom', path: ['schedule_2026'], message: `${s.game_id} has only one score` });
      }
    }
  });

export type SiteData = z.infer<typeof siteDataSchema>;
export type ScheduleGame = z.infer<typeof scheduleGameSchema>;
export type Game = z.infer<typeof gameSchema>;
export type WeekStatus = z.infer<typeof weekSchema>;

export function parseSiteData(raw: unknown): SiteData {
  const r = siteDataSchema.safeParse(raw);
  if (!r.success) {
    const lines = r.error.issues.slice(0, 20).map((i) => `  ${i.path.join('.') || '(root)'}: ${i.message}`);
    throw new Error(`site_data.json failed validation (needs schema 1.x, >= 1.4.0):\n${lines.join('\n')}`);
  }
  return r.data;
}
