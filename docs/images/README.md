# README figures

These are original vector documentation assets. `freshness.svg` illustrates the tested
portable demo; it is not an application screenshot. `retrieval.svg` is a standard
matplotlib plot generated from [the aggregate snapshot](../../bench/evidence/public-results.json).
It preserves all three channels on a common zero-to-one axis. No image model invented
benchmark values, interface features or product screenshots.

To regenerate with Python and matplotlib installed, from the repository root:

```sh
python docs/images/render.py --preview-dir /path/to/private-previews
```

The PNGs are optional review previews; the README consumes the SVGs. The renderer is
documentation tooling and is not used by the application, installer or native CI.
See [evaluation](../EVALUATION.md) for dates, protocols and limitations.
