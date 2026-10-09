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
2. Write a render scene into a copy of each task spec (training specs are never modified): `design/house_style.py` sets the
   camera and studio look (kept for the hand), then `design/website_scenes.py` gives the vehicle, quadruped and humanoid a
   rich setting built with the realistic held-out scene kit (`lib.py`, `env.py`, generated `tex/`): street, park, winter
   lot, parking lot or indoor hall, chosen from the skeleton's environment atom. Props are static, non-colliding and kept
   off the robot's path; a base ground plane covers the area beyond each task's terrain.
3. Render 960x540, 8 s clips of the trained policies with `design/render_batch.py` (one process per clip; specs and
   checkpoints are only read; no LLM). The rich-scene clips were rendered on Big Red 200 (`design/render_rich2.sh`), the hand
   on a local RTX 4090.
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

## Sim-to-real held-out tasks

The real-world clips (4K HDR iPhone footage) are tone-mapped to SDR (zscale + hable), downscaled to 960x540 at 30 fps,
sped up to about 30 s where longer (the badge shows the factor; `design/real_clips.json`), and capped near 1.8 Mbps.
`python design/build_real.py` writes the section between the `<!-- real:start -->` / `<!-- real:end -->` markers.

Render notes, applied to `_src/mjp/rendering.py` in the rendering copy of HandVLA only: `_attach_website_decor` attaches `render_scene.website_decor` (scenery XML) with prefix `decor_` and textures the `floor` plane or heightfield; `_fade_hand_occluders(model, robot)` makes non-hand geoms with `rbound >= 0.12` m whose top rises 5 cm above the hand 22% opaque; and the camera follows a hand's root body:

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
