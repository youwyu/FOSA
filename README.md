# FOSA project page

Website for *Generative Flow on Symbolic Abstractions for Embodiment Capability Discovery* (CoRL 2026), served from the `gh-pages` branch.

## Construction figure

The flow under "Which tasks can a robot actually learn?" shows the semantics of all 256 tasks FOSA sampled on the vehicle, from the discovery history:

```bash
python design/build_flow.py --history logs/figures/gemma/source-rendered/vlm/jackal/fosa_learn_history.jsonl
```

It writes the figure between the `<!-- flow:start -->` / `<!-- flow:end -->` markers in `index.html` and the counts to `design/flow-jackal.json`. `design/policy_flow.py` is a read-only probe of the trained GFlowNet's P_F (not used by the page).

## Capability frontier landscape

1. Sample 60 tasks per robot uniformly at random (`design/landscape_tasks.jsonl`).
2. Give each a render scene in one house style, written into a copy of its spec (no LLM, no decorative props): `python design/house_style.py --tasks <tasks with spec/render_spec paths>`.
3. Render 960x540, 8 s clips of the trained policies with `design/render_batch.py` (one process per clip; specs and checkpoints are only read). The vehicle and hand were rendered on a local RTX 4090, the quadruped and humanoid on Big Red 200 (`design/render_house.sh`). Two render-only patches apply to the hand (render notes below): the camera follows the hand's root body rather than the first free body in the scene (the object), and large scene objects rising above the hand are drawn see-through so task walls and overhead slabs do not hide it.
4. Copy the clips without re-encoding, moving the index to the front for instant playback:
   `ffmpeg -i in.mp4 -map 0:v -c copy -movflags +faststart static/videos/landscape/<key>.mp4`
5. Build the landscape data (needs numpy and scikit-learn):
   ```bash
   python design/build_landscape.py \
       --histories logs/figures/gemma/source-rendered/vlm \
       --videos static/videos/landscape
   ```
6. After editing `static/js/landscape.js`, rebuild the classic-script bundle the page loads (so it also works from `file://`):
   `npx esbuild static/js/landscape.js --bundle --format=iife --minify --legal-comments=eof --outfile=static/js/landscape.bundle.js`

Render notes, applied to `_src/mjp/rendering.py` in the rendering copy of HandVLA only (`_fade_hand_occluders(model, robot)` runs on each render model in `render_policy_video`, making non-hand geoms with `rbound >= 0.12` m whose top rises 5 cm above the hand 22% opaque):

```python
def _preferred_track_body(render_model):
    root = ROBOT_SPECS[render_model.robot].freebase_root_body
    if root:
        body = mujoco.mj_name2id(render_model.model, mujoco.mjtObj.mjOBJ_BODY, root)
        if body >= 0:
            return int(body)
    return _first_freejoint_body(render_model.model)
```

# Website License

The page template is borrowed from [Nerfies](https://nerfies.github.io).
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
