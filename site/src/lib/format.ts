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

export type Primetime = 'TNF' | 'SNF' | 'MNF';

/**
 * Primetime badge from the kickoff in US Eastern time: TNF = Thursday night (7 PM ET or later, so
 * Thanksgiving afternoon games get none), SNF = Sunday at or after 7 PM ET, MNF = any Monday game.
 */
export function primetime(iso: string): Primetime | null {
  const p = parts(iso, { weekday: 'short', hour: 'numeric', hourCycle: 'h23' });
  const hour = Number(p['hour']);
  if (p['weekday'] === 'Mon') return 'MNF';
  if (p['weekday'] === 'Thu' && hour >= 19) return 'TNF';
  if (p['weekday'] === 'Sun' && hour >= 19) return 'SNF';
  return null;
}

/* ---- Wins over expected (Amendment 6), counts only: no rates, no betting words ---- */

/** One decimal with a true minus sign: +3.6, −1.2, 0.0. */
export function signed1(x: number): string {
  const r = Math.round(x * 10) / 10;
  if (r === 0) return '0.0';
  return `${r > 0 ? '+' : '−'}${Math.abs(r).toFixed(1)}`;
}

/** The value tile's headline, split into the number and the words: "+3.6" "ahead of Vegas". */
export function woeHeadline(woe: number): [string, string] {
  const r = Math.round(woe * 10) / 10;
  if (r > 0) return [`+${r.toFixed(1)}`, 'ahead of Vegas'];
  if (r < 0) return [Math.abs(r).toFixed(1), 'behind Vegas'];
  return ['Even', 'with Vegas'];
}

/** "+1.2 ahead" / "1.2 behind" / "even". */
export function woeShort(woe: number): string {
  const r = Math.round(woe * 10) / 10;
  if (r > 0) return `+${r.toFixed(1)} ahead`;
  if (r < 0) return `${Math.abs(r).toFixed(1)} behind`;
  return 'even';
}

/** A tier line after its stars: "22 calls · +1.2 ahead (+0.6 per 10 calls)". */
export function tierText(n: number, woe: number, woePerCall: number | null): string {
  const calls = `${n} ${n === 1 ? 'call' : 'calls'}`;
  if (n === 0 || woePerCall === null) return `${calls} · none graded yet`;
  return `${calls} · ${woeShort(woe)} (${signed1(woePerCall * 10)} per 10 calls)`;
}
