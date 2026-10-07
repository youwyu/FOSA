# FOSA project page

Website for *Generative Flow on Symbolic Abstractions for Embodiment Capability Discovery* (CoRL 2026), served from the `gh-pages` branch.

## Construction figure

`design/policy_flow.py` replays the trained GFlowNet of FOSA's final vehicle run (read-only, no LLM calls) and beam-searches its forward policy; the output is `design/flow-jackal.json`. `python design/build_flow.py` turns it into the figure inside `index.html`.

## Capability frontier landscape

1. Render clips of FOSA's trained task policies on Big Red 200 (`design/render_batch.py`, submitted with `design/render_landscape.sh`; 60 tasks per robot sampled uniformly at random, listed in `design/landscape_tasks.jsonl`). The script only reads the discovery runs and calls no LLM.
2. Copy the clips without re-encoding, moving the index to the front for instant playback:
   `ffmpeg -i in.mp4 -c copy -movflags +faststart static/videos/landscape/<key>.mp4`
3. Build the landscape data (needs numpy and scikit-learn):
   ```bash
   python design/build_landscape.py \
       --histories logs/figures/gemma/source-rendered/vlm \
       --videos static/videos/landscape
   ```
4. After editing `static/js/landscape.js`, rebuild the classic-script bundle the page loads (so it also works from `file://`):
   `npx esbuild static/js/landscape.js --bundle --format=iife --minify --legal-comments=eof --outfile=static/js/landscape.bundle.js`

# Website License

The page template is borrowed from [Nerfies](https://nerfies.github.io).
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
