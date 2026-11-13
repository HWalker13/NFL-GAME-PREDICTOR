/* Display formatting. Built from formatToParts so ICU's narrow no-break spaces never leak in. */

const ET = 'America/New_York';

function parts(iso: string, opts: Intl.DateTimeFormatOptions): Record<string, string> {
  const out: Record<string, string> = {};
  for (const p of new Intl.DateTimeFormat('en-US', { timeZone: ET, ...opts }).formatToParts(new Date(iso))) {
    out[p.type] = p.value;
  }
  return out;
}

/** "Sun 1:00 PM ET" */
export function kickoffEt(iso: string): string {
  const p = parts(iso, { weekday: 'short', hour: 'numeric', minute: '2-digit', hour12: true });
  return `${p['weekday']} ${p['hour']}:${p['minute']} ${p['dayPeriod']} ET`;
}

/** "Nov 13, 2026, 3:12 PM ET" */
export function dateTimeEt(iso: string): string {
  const p = parts(iso, { month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit', hour12: true });
  return `${p['month']} ${p['day']}, ${p['year']}, ${p['hour']}:${p['minute']} ${p['dayPeriod']} ET`;
}

/** 0.6431 -> "64%"; with decimals = 1, 0.5037 -> "50.4%" */
export const pct = (p: number, decimals: 0 | 1 = 0): string =>
  decimals === 1 ? `${(Math.round(p * 1000) / 10).toFixed(1)}%` : `${Math.round(p * 100)}%`;

/** True when a probability would read as a coin flip at whole-number precision. */
export const showsAsFifty = (p: number): boolean => Math.round(p * 100) === 50;
