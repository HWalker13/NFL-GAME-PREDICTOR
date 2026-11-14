/* Every text/background pair in global.css meets WCAG AA (4.5:1), and every bar fill 3:1, light and dark. */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const css = readFileSync(new URL('../src/styles/global.css', import.meta.url), 'utf8');

function tokens(block: string): Record<string, string> {
  return Object.fromEntries([...block.matchAll(/--([\w-]+):\s*(#[0-9a-f]{6})/gi)].map((m) => [m[1]!, m[2]!]));
}
const light = tokens(css.split('@media')[0]!);
const dark = { ...light, ...tokens(css.slice(css.indexOf('@media'))) };

function lum(hex: string): number {
  const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  const [r, g, b] = c.map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4)) as [number, number, number];
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
const ratio = (a: string, b: string): number => {
  const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p) as [number, number];
  return (x + 0.05) / (y + 0.05);
};

const PAIRS: [string, string][] = [
  ['text', 'bg'], ['text', 'surface'], ['text', 'note'],
  ['muted', 'bg'], ['muted', 'surface'],
  ['accent', 'bg'], ['accent', 'surface'],
  ['on-accent', 'accent'],
  // header band
  ['on-brand', 'brand'], ['on-brand-muted', 'brand'],
  // TNF/SNF/MNF pills
  ['on-red', 'red'],
  // chips: neutral, ✓, ✗, and the open chip on the card surface
  ['text', 'chip'], ['good', 'good-bg'], ['bad', 'bad-bg'],
  // details summary and tabs on the card surface
  ['accent', 'note'],
];

/** Non-text graphics (WCAG 1.4.11): the details bars against the card and each other. */
const GRAPHICS: [string, string][] = [
  ['bar-away', 'surface'], ['bar-home', 'surface'],
];

describe.each([['light', light], ['dark', dark]] as const)('%s theme', (_, t) => {
  it.each(PAIRS)('%s on %s >= 4.5:1', (fg, bg) => {
    expect(t[fg], fg).toBeDefined();
    expect(t[bg], bg).toBeDefined();
    expect(ratio(t[fg]!, t[bg]!)).toBeGreaterThanOrEqual(4.5);
  });
  it.each(GRAPHICS)('%s on %s >= 3:1', (fg, bg) => {
    expect(ratio(t[fg]!, t[bg]!)).toBeGreaterThanOrEqual(3);
  });
  it('the red accent separates from the navy band (3:1 against the page)', () => {
    expect(ratio(t['red']!, t['bg']!)).toBeGreaterThanOrEqual(3);
  });
});

it('the dark theme redefines every light token', () => {
  const darkOnly = tokens(css.slice(css.indexOf('@media')));
  expect(Object.keys(darkOnly).sort()).toEqual(Object.keys(light).sort());
});
