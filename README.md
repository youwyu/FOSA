# FOSA project page

Website for *Generative Flow on Symbolic Abstractions for Embodiment Capability Discovery* (CoRL 2026), served from the `gh-pages` branch.

## Updating the results

The frontier explorer and the result tables are generated from the paper's frozen discovery logs:

```bash
python design/build_data.py \
    --histories logs/figures/gemma/source-rendered/vlm \
    --paper-final logs/figures/paper-final
```

This writes `static/data/frontier-<robot>.js` and `static/data/summary.js`, and checks the recomputed Frontier-Prox of every method against `SUMMARY.md`.

# Website License

The page template is borrowed from [Nerfies](https://nerfies.github.io).
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
