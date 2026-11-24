# Documentation images

## Selected hero

`lastknown-hero.png` is the selected original bright editorial concept: ivory paper,
cobalt source links, black typography and an orange changed passage. Generated with the
built-in imagegen tool in text-to-image mode, without a reference image. It is conceptual
artwork, not an application screenshot or a quantitative chart. Its illustration connects
a retrieved excerpt to its source; actual freshness behavior is documented in
[Freshness](../FRESHNESS.md) and the [portable demo](../TRY-IT.md).

Prompt brief: wide 2:1 developer-tool README hero; oversized lowercase “lastknown” wordmark;
exact tagline “Local memory that checks its sources.”; restrained ivory/cobalt/black/orange
palette; clean editorial typography; abstract Markdown source sheets and a linked excerpt,
with changed lines highlighted; generous spacing; no robot, brain, invented UI or statistics.

## Reproducible figures

`retrieval.svg` is a standard matplotlib plot generated from
[the aggregate snapshot](../../bench/evidence/public-results.json). It preserves all three
channels on a common zero-to-one axis. No image model supplies benchmark values.
`freshness.svg` is the earlier vector illustration of the tested portable demo, retained
as documentation source; the main README now uses the selected generated hero.

To regenerate the vector figures with Python and matplotlib installed, from the root:

```sh
python docs/images/render.py --preview-dir /path/to/private-previews
```

This renderer does not regenerate the hero. Its PNG previews are optional review outputs.
It is documentation tooling, unused by the application, installer or native CI.
See [evaluation](../EVALUATION.md) for dates, protocols and limitations.
