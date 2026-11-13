// @ts-check
import { defineConfig } from 'astro/config';

export default defineConfig({
  output: 'static',
  trailingSlash: 'always',
  // One request per page: the whole stylesheet is a few KB, so inlining beats a second round trip.
  build: { inlineStylesheets: 'always' },
});
