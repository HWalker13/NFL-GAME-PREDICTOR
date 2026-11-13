/*
  The build's one copy of the site model. Pages import `site` in frontmatter only, so the JSON is
  never bundled or shipped; the HTML carries only the rendered text.

  An export built from uncommitted code (source_commit.dirty) fails the build, because its commit
  link would point at code that doesn't match the numbers. NFL_ALLOW_DIRTY=1 overrides that for
  local previews.
*/
import { buildSite, loadSiteData, type Site } from './data.ts';

function load(): Site {
  const d = loadSiteData();
  if (d.source_commit.dirty && process.env['NFL_ALLOW_DIRTY'] !== '1') {
    throw new Error(
      'site_data.json was exported from uncommitted code (source_commit.dirty). Commit, re-run ' +
        '`python -m scripts.export_site_data`, and commit the export. NFL_ALLOW_DIRTY=1 overrides for local previews.',
    );
  }
  return buildSite(d);
}

export const site: Site = load();
