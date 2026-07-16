# DRAMA-X Viewer

Streamlit viewer for the [DRAMA-X](https://github.com/taco-group/DRAMA-X) benchmark
(fine-grained VRU intent prediction + risk reasoning, [arXiv:2506.17590](https://arxiv.org/abs/2506.17590)).

- `data/drama_x_annotated.jsonl` — the 5,686 public annotations, downloaded from
  [HuggingFace: mgod96/DRAMA-X](https://huggingface.co/datasets/mgod96/DRAMA-X) (CC-BY-4.0).
- Images are **not** public: request the original DRAMA dataset at
  <https://usa.honda-ri.com/drama>, then set "DRAMA images root" in the app sidebar
  to draw boxes on real frames. Without it the app shows an annotation-only scene layout.

## Run

```bash
streamlit run app.py
```

## What you get

- **Scene viewer** — browse/filter scenes (risk, intents, positions, suggested action,
  cyclists-only); bounding boxes at true pixel coordinates, intent arrows
  (lateral ↔ / vertical ↕ relative to ego), per-agent motion descriptions,
  scene risk badge and suggested ego action.
- **Dataset statistics** — scenes / pedestrians / cyclists / % risky, intent and
  action distributions, agents-per-scene, and a raw table view.
