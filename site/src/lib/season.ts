/* Pure season logic: which week is current, and what kind of week each one is. */
import type { ScheduleGame, WeekStatus } from './schema.ts';

export const WEEKS = Array.from({ length: 18 }, (_, i) => i + 1);

export const isFinal = (g: ScheduleGame): boolean => g.home_score !== null && g.away_score !== null;

/**
 * The earliest week with any game that has no final score yet. Scores come from the export's
 * schedule snapshot, so the answer depends on the data, never on the build machine's clock.
 * Once every game is final, the season is over and week 18 stays current.
 */
export function currentWeek(schedule: readonly ScheduleGame[]): number {
  const open = schedule.filter((g) => !isFinal(g)).map((g) => g.week);
  return open.length ? Math.min(...open) : 18;
}

/**
 * before-tracking: earlier than the first official week (no record by design).
 * missed / partial: straight from the export's week status.
 * tracked: picks exist for the week (graded, or pending with predictions).
 * future: nothing logged yet.
 */
export type WeekKind = 'before-tracking' | 'missed' | 'partial' | 'tracked' | 'future';

export interface WeekInfo {
  week: number;
  kind: WeekKind;
  scheduled: number;
  predicted: number;
}

export function weekInfo(
  week: number,
  firstOfficialWeek: number,
  statuses: readonly WeekStatus[],
  schedule: readonly ScheduleGame[],
): WeekInfo {
  const scheduled = schedule.filter((g) => g.week === week).length;
  const s = statuses.find((x) => x.week === week);
  const predicted = s?.predicted ?? 0;
  let kind: WeekKind;
  if (week < firstOfficialWeek) kind = 'before-tracking';
  else if (s?.status === 'missed') kind = 'missed';
  else if (s?.status === 'partial') kind = 'partial';
  else if (s && s.predicted > 0) kind = 'tracked';
  else kind = 'future';
  return { week, kind, scheduled, predicted };
}

/** The note above a week's game list, or null when every game has a pick. */
export function weekBanner(info: WeekInfo, firstOfficialWeek: number): string | null {
  switch (info.kind) {
    case 'before-tracking':
      return `Tracking started in week ${firstOfficialWeek}.`;
    case 'missed':
      return 'No picks were logged this week.';
    case 'partial':
      return `Picks were logged for ${info.predicted} of ${info.scheduled} games this week.`;
    case 'future':
      return "Picks will appear here once they're logged.";
    case 'tracked':
      return null;
  }
}

/** Dimmed in the week tabs: weeks with no record, by design or by a miss. */
export const isDimmed = (kind: WeekKind): boolean => kind === 'before-tracking' || kind === 'missed';
