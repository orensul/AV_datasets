"""DRAMA-X dataset viewer.

Browse the 5,686 annotated scenes of DRAMA-X (fine-grained VRU intent + risk
benchmark, arXiv:2506.17590): per-agent bounding boxes, dual intent labels
(lateral / vertical), positions, motion descriptions, scene-level risk and
suggested ego actions.

Images are NOT part of the public annotations — they come from the original
DRAMA dataset (request access at https://usa.honda-ri.com/drama). Until then
the viewer renders an annotation-only scene layout; point "DRAMA images root"
at your local DRAMA copy to see boxes drawn on real frames.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_PATH = Path(__file__).parent / "data" / "drama_x_annotated.jsonl"

# Validated categorical palette (dataviz reference palette, light mode).
PED_COLOR = "#2a78d6"   # slot 1 — blue
CYC_COLOR = "#008300"   # slot 2 — green
SEQ_BLUE = "#2a78d6"    # single-hue magnitude bars
RISK_YES = "#d03b3b"    # status: critical
RISK_NO = "#0ca30c"     # status: good
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

LATERAL = ["goes to the left", "goes to the right", "stationary"]
VERTICAL = ["moves towards ego vehicle", "moves away from ego vehicle", "stationary"]
POSITIONS = ["Left of ego vehicle", "Front of ego vehicle", "Right of ego vehicle"]


@st.cache_data(show_spinner="Loading DRAMA-X annotations…")
def load_records() -> list[dict]:
    records = []
    with open(DATA_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def iter_agents(rec: dict):
    """Yield (agent_key, kind, agent_dict) for every VRU in a record."""
    for kind, prefix in (("Pedestrians", "P"), ("Cyclists", "C")):
        for aid, agent in rec.get(kind, {}).items():
            yield f"{prefix}{aid}", kind, agent


@st.cache_data(show_spinner=False)
def build_index(_records_len: int) -> pd.DataFrame:
    """One row per scene with the fields the filters need."""
    rows = []
    for i, rec in enumerate(load_records()):
        laterals, verticals, positions = set(), set(), set()
        n_ped = len(rec.get("Pedestrians", {}))
        n_cyc = len(rec.get("Cyclists", {}))
        for _, _, agent in iter_agents(rec):
            intent = agent.get("Intent", [])
            if len(intent) > 0:
                laterals.add(intent[0])
            if len(intent) > 1:
                verticals.add(intent[1])
            positions.add(agent.get("Position", ""))
        rows.append(
            {
                "idx": i,
                "id": rec["id"],
                "risk": rec.get("Risk", "N/A"),
                "action": rec.get("suggested_action", ""),
                "n_ped": n_ped,
                "n_cyc": n_cyc,
                "laterals": laterals,
                "verticals": verticals,
                "positions": positions,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def find_image(scene_id: str, images_root: str) -> str | None:
    """Best-effort lookup of the DRAMA frame for a scene id like
    clip_305_000786_frame_000786 under a local DRAMA images root."""
    root = Path(images_root).expanduser()
    if not root.is_dir():
        return None
    parts = scene_id.split("_frame_")
    stem = parts[0]  # e.g. clip_305_000786
    frame = parts[1] if len(parts) > 1 else ""
    for pattern in (f"**/{scene_id}*", f"**/{stem}*{frame}*", f"**/*{stem}*"):
        for hit in root.glob(pattern):
            if hit.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                return str(hit)
    return None


def intent_vector(intent: list[str]) -> tuple[float, float]:
    """Arrow direction (image coords, y down) for a [lateral, vertical] intent."""
    dx = {"goes to the left": -1.0, "goes to the right": 1.0}.get(
        intent[0] if intent else "", 0.0
    )
    dy = {"moves towards ego vehicle": 1.0, "moves away from ego vehicle": -1.0}.get(
        intent[1] if len(intent) > 1 else "", 0.0
    )
    return dx, dy


def scene_figure(rec: dict, image_path: str | None) -> go.Figure:
    """Bounding boxes + intent arrows, over the real frame when available."""
    boxes = [a.get("Box", [0, 0, 0, 0]) for _, _, a in iter_agents(rec)]
    width = max(1920, max((b[2] for b in boxes), default=0) + 40)
    height = max(1080, max((b[3] for b in boxes), default=0) + 40)

    fig = go.Figure()
    if image_path:
        from PIL import Image

        img = Image.open(image_path)
        width, height = img.size
        fig.add_layout_image(
            source=img, xref="x", yref="y", x=0, y=0,
            sizex=width, sizey=height, sizing="stretch", layer="below",
        )

    seen_kinds = set()
    for label, kind, agent in iter_agents(rec):
        x1, y1, x2, y2 = agent.get("Box", [0, 0, 0, 0])
        color = PED_COLOR if kind == "Pedestrians" else CYC_COLOR
        intent = agent.get("Intent", [])
        hover = (
            f"<b>{label} — {kind[:-1]}</b><br>"
            f"Lateral: {intent[0] if intent else '—'}<br>"
            f"Vertical: {intent[1] if len(intent) > 1 else '—'}<br>"
            f"Position: {agent.get('Position', '—')}<br>"
            + "<br>".join(
                agent.get("Description", "")[i : i + 70]
                for i in range(0, min(len(agent.get("Description", "")), 280), 70)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[x1, x2, x2, x1, x1],
                y=[y1, y1, y2, y2, y1],
                mode="lines",
                line=dict(color=color, width=2),
                name=kind[:-1],
                legendgroup=kind,
                showlegend=kind not in seen_kinds,
                hoverinfo="skip",
            )
        )
        seen_kinds.add(kind)
        # invisible fill to give the whole box a hover target
        fig.add_trace(
            go.Scatter(
                x=[(x1 + x2) / 2], y=[(y1 + y2) / 2],
                mode="markers",
                marker=dict(size=max((x2 - x1) / 4, 12), opacity=0),
                hovertemplate=hover + "<extra></extra>",
                showlegend=False,
            )
        )
        fig.add_annotation(
            x=x1, y=y1 - 8, text=f"<b>{label}</b>", showarrow=False,
            font=dict(color="#ffffff", size=12),
            bgcolor=color, borderpad=2, xanchor="left", yanchor="bottom",
        )
        dx, dy = intent_vector(intent)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if dx or dy:
            fig.add_annotation(
                x=cx + dx * 90, y=cy + dy * 60, ax=cx, ay=cy,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowwidth=3, arrowcolor=color,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[cx], y=[cy], mode="markers",
                    marker=dict(size=10, color=color, symbol="circle-open",
                                line=dict(width=3)),
                    showlegend=False, hoverinfo="skip",
                )
            )

    fig.update_xaxes(range=[0, width], visible=False, constrain="domain")
    fig.update_yaxes(
        range=[height, 0], visible=False, scaleanchor="x", scaleratio=1,
    )
    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE if not image_path else "#000000",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    font=dict(color=INK)),
        hoverlabel=dict(bgcolor="#ffffff", font=dict(color=INK)),
    )
    return fig


def bar_chart(labels: list[str], values: list[int], title: str,
              horizontal: bool = False, color: str = SEQ_BLUE) -> go.Figure:
    text = [f"{v:,}" for v in values]
    if horizontal:
        trace = go.Bar(y=labels, x=values, orientation="h", text=text,
                       textposition="outside", cliponaxis=False)
    else:
        trace = go.Bar(x=labels, y=values, text=text, textposition="outside",
                       cliponaxis=False)
    trace.update(marker=dict(color=color, line=dict(color=SURFACE, width=2)))
    fig = go.Figure(trace)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=INK)),
        height=320,
        margin=dict(l=10, r=30, t=40, b=10),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        bargap=0.45,
        font=dict(color=INK_MUTED, size=12),
        xaxis=dict(showgrid=not horizontal, gridcolor=GRID, zeroline=False,
                   linecolor=BASELINE, tickfont=dict(color=INK_MUTED)),
        yaxis=dict(showgrid=horizontal is False, gridcolor=GRID, zeroline=False,
                   linecolor=BASELINE, tickfont=dict(color=INK)),
        showlegend=False,
    )
    if horizontal:
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def risk_badge(risk: str) -> str:
    if risk == "Yes":
        return (f'<span style="background:{RISK_YES};color:#fff;padding:4px 12px;'
                'border-radius:6px;font-weight:600">&#9888; Risk: Yes</span>')
    if risk == "No":
        return (f'<span style="background:{RISK_NO};color:#fff;padding:4px 12px;'
                'border-radius:6px;font-weight:600">&#10003; Risk: No</span>')
    return ('<span style="background:#898781;color:#fff;padding:4px 12px;'
            'border-radius:6px;font-weight:600">Risk: N/A</span>')


def main() -> None:
    st.set_page_config(page_title="DRAMA-X Viewer", page_icon="🚸", layout="wide")
    records = load_records()
    index = build_index(len(records))

    st.title("DRAMA-X dataset viewer")
    st.caption(
        "Fine-grained VRU intent prediction & risk reasoning benchmark "
        "([paper](https://arxiv.org/abs/2506.17590) · "
        "[data](https://huggingface.co/datasets/mgod96/DRAMA-X)) — "
        "5,686 accident-prone scenes from DRAMA (Tokyo), annotated with "
        "per-agent boxes, directional intents, risk and suggested ego actions."
    )

    # ---------------- sidebar: filters ----------------
    with st.sidebar:
        st.header("Filters")
        risk_filter = st.multiselect("Scene risk", ["Yes", "No", "N/A"],
                                     default=["Yes", "No", "N/A"])
        only_cyclists = st.checkbox("Only scenes with cyclists")
        lat_filter = st.multiselect("Lateral intent (any agent)", LATERAL)
        vert_filter = st.multiselect("Vertical intent (any agent)", VERTICAL)
        pos_filter = st.multiselect("Agent position (any agent)", POSITIONS)
        action_filter = st.multiselect(
            "Suggested ego action", sorted(index["action"].unique()))
        st.divider()
        images_root = st.text_input(
            "DRAMA images root (optional)",
            help="Local path to the original DRAMA dataset images "
                 "(https://usa.honda-ri.com/drama). When set, boxes are drawn "
                 "on the real frames instead of the layout canvas.",
        )

    mask = index["risk"].isin(risk_filter)
    if only_cyclists:
        mask &= index["n_cyc"] > 0
    if lat_filter:
        mask &= index["laterals"].apply(lambda s: bool(s & set(lat_filter)))
    if vert_filter:
        mask &= index["verticals"].apply(lambda s: bool(s & set(vert_filter)))
    if pos_filter:
        mask &= index["positions"].apply(lambda s: bool(s & set(pos_filter)))
    if action_filter:
        mask &= index["action"].isin(action_filter)
    filtered = index[mask].reset_index(drop=True)

    tab_scene, tab_stats = st.tabs(["Scene viewer", "Dataset statistics"])

    # ---------------- scene viewer ----------------
    with tab_scene:
        if filtered.empty:
            st.warning("No scenes match the current filters.")
            st.stop()

        if "pos" not in st.session_state:
            st.session_state.pos = 0
        st.session_state.pos = min(st.session_state.pos, len(filtered) - 1)

        nav_prev, nav_pick, nav_next = st.columns([1, 5, 1])
        with nav_prev:
            if st.button("◀ Prev", width="stretch"):
                st.session_state.pos = max(0, st.session_state.pos - 1)
        with nav_next:
            if st.button("Next ▶", width="stretch"):
                st.session_state.pos = min(len(filtered) - 1,
                                           st.session_state.pos + 1)
        with nav_pick:
            st.session_state.pos = st.selectbox(
                f"Scene ({len(filtered):,} match filters)",
                options=range(len(filtered)),
                index=st.session_state.pos,
                format_func=lambda i: filtered.loc[i, "id"],
                label_visibility="collapsed",
            )

        row = filtered.loc[st.session_state.pos]
        rec = records[row["idx"]]

        image_path = find_image(rec["id"], images_root) if images_root else None
        left, right = st.columns([3, 2])
        with left:
            if images_root and not image_path:
                st.info("Frame not found under the given images root — "
                        "showing annotation layout only.")
            elif not images_root:
                st.info("Annotation-only layout (no DRAMA images configured). "
                        "Boxes are drawn at their true pixel coordinates; "
                        "arrows show the annotated intent direction.")
            st.plotly_chart(scene_figure(rec, image_path),
                            width="stretch")

        with right:
            st.markdown(risk_badge(rec.get("Risk", "N/A")),
                        unsafe_allow_html=True)
            st.markdown(f"**Suggested ego action:** "
                        f"{rec.get('suggested_action', '—')}")
            st.markdown(f"`{rec['id']}`")
            st.divider()
            for label, kind, agent in iter_agents(rec):
                intent = agent.get("Intent", ["—", "—"])
                color = PED_COLOR if kind == "Pedestrians" else CYC_COLOR
                st.markdown(
                    f'<span style="color:{color};font-weight:700">■</span> '
                    f"**{label} · {kind[:-1]}** — {agent.get('Position', '—')}",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"↔ {intent[0] if intent else '—'} &nbsp;·&nbsp; "
                    f"↕ {intent[1] if len(intent) > 1 else '—'}"
                )
                st.caption(agent.get("Description", ""))

    # ---------------- dataset statistics ----------------
    with tab_stats:
        n_scenes = len(index)
        n_ped, n_cyc = int(index["n_ped"].sum()), int(index["n_cyc"].sum())
        pct_risk = 100 * (index["risk"] == "Yes").mean()
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Scenes", f"{n_scenes:,}")
        t2.metric("Pedestrians", f"{n_ped:,}")
        t3.metric("Cyclists", f"{n_cyc:,}")
        t4.metric("Risk-positive scenes", f"{pct_risk:.1f}%")
        st.caption("Statistics below reflect the full dataset, not the "
                   "current filters.")

        lat_counts: dict[str, int] = {k: 0 for k in LATERAL}
        vert_counts: dict[str, int] = {k: 0 for k in VERTICAL}
        for rec in records:
            for _, _, agent in iter_agents(rec):
                intent = agent.get("Intent", [])
                if intent and intent[0] in lat_counts:
                    lat_counts[intent[0]] += 1
                if len(intent) > 1 and intent[1] in vert_counts:
                    vert_counts[intent[1]] += 1

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                bar_chart(list(lat_counts), list(lat_counts.values()),
                          "Lateral intent (per agent)"),
                width="stretch")
        with c2:
            st.plotly_chart(
                bar_chart(list(vert_counts), list(vert_counts.values()),
                          "Vertical intent (per agent)"),
                width="stretch")

        action_counts = index["action"].str.split(" (", regex=False).str[0] \
            .value_counts()
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(
                bar_chart(action_counts.index.tolist(),
                          action_counts.values.tolist(),
                          "Suggested ego action (per scene)", horizontal=True),
                width="stretch")
        with c4:
            agents = (index["n_ped"] + index["n_cyc"]).value_counts().sort_index()
            st.plotly_chart(
                bar_chart([str(i) for i in agents.index],
                          agents.values.tolist(),
                          "VRU agents per scene"),
                width="stretch")

        with st.expander("Underlying data (table view)"):
            st.dataframe(
                index[["id", "risk", "action", "n_ped", "n_cyc"]],
                width="stretch", height=400,
            )


if __name__ == "__main__":
    main()
