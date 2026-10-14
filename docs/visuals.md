# Visual reproduction

The result figures and replay captions use a shared
STIX General serif family, supplied by Matplotlib. The README keeps the
original project banner. Figures have larger type, external
legends, separate title bands and both PNG and scalable SVG exports. The
README architecture is a native Mermaid diagram rendered by GitHub. Its
three result figures cover payload/friction adaptation, posture/delay limits
and the planar motor-fault comparison. All eight result figures are indexed
in [results](results.md). Plot
generators check canvas bounds and separation of neighbouring panels before
writing. Replay captions have reserved text bands whose bounds are checked
in every frame.

The two GIFs replay the first retained cases. The nominal motion uses seed
`9804000` at one quarter speed, with the saved tip path and target. The paired
control comparison uses seed `9804300` at one eighth speed, with joint-reference
error, target markers, recorded tip trails and a Cartesian close-up. The
close-up uses millimetres and equal X/Z scales; saved joint motions are not
amplified. Full-resolution MP4 files accompany both. Neither renderer advances
the physical plant or opens new seeds. Collision meshes are those packaged
with the model; the payload changes inertia without adding a visible object.

## Commands

From a repository checkout, install `.[demo]`, or use the Python 3.12
[reference environment](demo.md). On headless Linux, set `MUJOCO_GL=egl`.
Existing outputs are not overwritten, so use a fresh destination:

```bash
python -m tools.render_panda_demo \
  --output visual-output/control.mp4 --poster visual-output/control.png \
  --gif visual-output/control.gif
python -m tools.render_panda_motion \
  --output visual-output/motion.mp4 --poster visual-output/motion.png \
  --gif visual-output/motion.gif
python -m tools.make_visual_overview --output visual-output/overview
python -m tools.make_panda_figure --output visual-output/nominal.png
python -m tools.make_panda_adaptation_figure --output visual-output/adaptation.png
python -m tools.make_panda_robustness_figure --output visual-output/robustness.png
python -m tools.make_panda_delay_figure --output visual-output/delay.png
python -m tools.make_panda_posture_delay_figure --output visual-output/posture.png
python -m tools.make_panda_variable_delay_figure --output visual-output/variable.png
python -m tools.make_panda_score_memory_figure --output visual-output/memory.png
python -m tools.make_v20_figure --output visual-output/planar.png
```

The Panda plots verify completion hashes and use saved data. The planar v2.0
figure replays the already-opened motor-fault seed selected by the original
documented outcome rule and reads the full retained identification table.
This replay is illustrative; it is not a new campaign or a randomly selected
representative episode. Original evidence and prior release images remain
preserved. The plot scripts emit SVG alongside PNG, using the same data.
