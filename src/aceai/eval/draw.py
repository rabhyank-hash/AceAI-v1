"""Draw an Agent 1 module graph as a Mermaid flowchart.

Layout:
- Columns are topological layers ("steps"). A module sits one step after the latest module it
  depends on. Modules in the same step have no dependencies between them, so any order among them
  is valid: these are the ties `topo_sort_modules` reports.
- Arrows point from a prerequisite module to the module that needs it. Solid arrows have LO-level
  evidence (labelled with the number of LO prerequisite links); dashed arrows are declared in
  `Module.depends_on` without any LO edge behind them.
- Colour is the CSV unit most of the module's LOs come from; the label shows how many of its LOs
  that is, so mixed modules stand out.

Mermaid renders on GitHub and in VS Code's Markdown preview; `to_html` wraps it in a standalone
page that loads Mermaid from a CDN.
"""

from __future__ import annotations

from collections import Counter

import networkx as nx

from aceai.ingest.ground_truth import GroundTruth
from aceai.schemas import SequencerOutput
from aceai.tools import build_module_graph, topo_sort_modules

# Fill / stroke / text per colour slot; readable on light and dark Mermaid themes.
_PALETTE = [
    ("#dbe8f7", "#3a6aa8", "#10243d"),
    ("#dff1e3", "#3c8a55", "#12301b"),
    ("#fbe9d4", "#b0701f", "#3a2408"),
    ("#efe0f5", "#7d4a9b", "#2a1535"),
    ("#fde0e0", "#b04444", "#3d1212"),
    ("#dff3f3", "#2f8585", "#0f2e2e"),
    ("#f5f2d3", "#8f8420", "#302b08"),
    ("#e6e6ee", "#5a5a78", "#1c1c28"),
    ("#fbe0ef", "#a8457a", "#3a1229"),
    ("#e3eed8", "#5e7f33", "#1f2b10"),
]
_NO_UNIT = ("#f2f2f2", "#8a8a8a", "#222222")


def _esc(text: str) -> str:
    return text.replace('"', "#quot;").replace("<", "#lt;").replace(">", "#gt;")


def _node_id(module_id: str) -> str:
    return "n_" + "".join(c if c.isalnum() else "_" for c in module_id)


def layers(output: SequencerOutput) -> tuple[list[list[str]], bool]:
    """Topological layers of the module graph (declared + LO-derived edges), each in the model's
    order. Returns ([], False) if the graph has a cycle."""
    order = {m.id: m.order for m in output.modules}
    g = nx.DiGraph()
    g.add_nodes_from(order)
    for e in build_module_graph(output.modules, output.los).edges:
        if e.from_module in order and e.to_module in order:
            g.add_edge(e.to_module, e.from_module)  # prerequisite -> dependent
    if not nx.is_directed_acyclic_graph(g):
        return [], False
    return [sorted(gen, key=order.get) for gen in nx.topological_generations(g)], True


def module_units(
    output: SequencerOutput, id_map: dict[str, str], gt: GroundTruth
) -> dict[str, tuple[str | None, int, int]]:
    """module id -> (dominant CSV unit label, LOs from that unit, LOs in the module)."""
    unit_of = {}
    for u in gt.units:
        for m in u.modules:
            for rid in m.lo_ids:
                unit_of[rid] = f"u{u.unit_no}"
    sources: dict[str, list[str]] = {}
    for e in output.provenance:
        sources.setdefault(e.lo_id, []).append(id_map.get(e.raw_id, e.raw_id))
    out = {}
    for m in output.modules:
        units = Counter(
            unit_of[r] for lid in m.lo_ids for r in sources.get(lid, []) if r in unit_of
        )
        total = sum(units.values())
        if units:
            top, n = units.most_common(1)[0]
            out[m.id] = (top, n, total)
        else:
            out[m.id] = (None, 0, total)
    return out


def module_graph_mermaid(output: SequencerOutput, id_map: dict[str, str], gt: GroundTruth) -> str:
    graph = build_module_graph(output.modules, output.los)
    steps, acyclic = layers(output)
    units = module_units(output, id_map, gt)
    by_id = {m.id: m for m in output.modules}
    unit_names = {f"u{u.unit_no}": u.unit_name.strip() for u in gt.units}
    unit_slots = {u: i for i, u in enumerate(sorted(unit_names, key=lambda x: int(x[1:])))}

    lines = ["flowchart LR"]

    def node(mid: str) -> str:
        m = by_id[mid]
        unit, n, total = units[mid]
        csv = f"CSV {unit} {n}/{total}" if unit else "no CSV unit"
        label = f"<b>{m.order}. {_esc(m.title)}</b><br/>{len(m.lo_ids)} LOs · {csv}"
        return f'    {_node_id(mid)}["{label}"]'

    if acyclic:
        for i, step in enumerate(steps, 1):
            title = "Step 1: no prerequisites" if i == 1 else f"Step {i}"
            lines.append(f'  subgraph s{i}["{title}"]')
            lines += [node(mid) for mid in step]
            lines.append("  end")
    else:
        lines += [node(m.id) for m in sorted(output.modules, key=lambda m: m.order)]

    for e in graph.edges:
        if e.from_module not in by_id or e.to_module not in by_id:
            continue
        a, b = _node_id(e.to_module), _node_id(e.from_module)  # prerequisite -> dependent
        if e.lo_edges:
            n = len(e.lo_edges)
            lines.append(f'  {a} -->|"{n} LO link{"s" if n != 1 else ""}"| {b}')
        else:
            lines.append(f"  {a} -.-> {b}")

    used = sorted({u for u, _, _ in units.values() if u}, key=lambda x: int(x[1:]))
    for u in used:
        fill, stroke, text = _PALETTE[unit_slots[u] % len(_PALETTE)]
        lines.append(f"  classDef {u} fill:{fill},stroke:{stroke},color:{text}")
    fill, stroke, text = _NO_UNIT
    lines.append(f"  classDef nounit fill:{fill},stroke:{stroke},color:{text}")
    for mid, (u, _, _) in units.items():
        lines.append(f"  class {_node_id(mid)} {u or 'nounit'}")

    if used:
        lines.append('  subgraph legend["CSV units (colour)"]')
        lines.append("    direction TB")
        for u in used:
            lines.append(f'    key_{u}["{u}: {_esc(unit_names[u])}"]')
            lines.append(f"    class key_{u} {u}")
        lines.append("  end")
    return "\n".join(lines)


def graph_markdown(output: SequencerOutput, id_map: dict[str, str], gt: GroundTruth) -> str:
    """Mermaid diagram plus the facts needed to read it."""
    topo = topo_sort_modules(output.modules, output.los)
    steps, acyclic = layers(output)
    n_lo_edges = sum(len(lo.depends_on) for lo in output.los)
    graph = build_module_graph(output.modules, output.los)
    solid = sum(1 for e in graph.edges if e.lo_edges)
    dashed = sum(1 for e in graph.edges if not e.lo_edges)
    lines = [
        "## Module graph",
        "",
        "Columns are topological steps: a module sits one step after the latest module it "
        "depends on, and modules in the same step can be taught in any order. Arrows point from "
        "a prerequisite to the module that needs it; solid arrows come from LO-level "
        "prerequisites, dashed arrows are declared by the model with no LO prerequisite behind "
        "them. Colour is the CSV unit most of a module's LOs come from.",
        "",
    ]
    if acyclic:
        widest = max((len(s) for s in steps), default=0)
        lines += [
            f"- {len(output.modules)} modules in {len(steps)} steps (widest step: {widest} "
            "modules)",
            f"- {n_lo_edges} LO prerequisite links; module edges: {solid} solid, {dashed} dashed",
            f"- topological order is {'unique' if topo.unique else 'not unique'}: "
            f"{len(topo.ties)} tie points; model order "
            f"{'matches' if topo.matches_current_order else 'differs from'} the tie-broken sort",
            "",
        ]
    else:
        lines += ["- the module graph has a cycle; shown in model order without steps", ""]
    lines += ["```mermaid", module_graph_mermaid(output, id_map, gt), "```", ""]
    return "\n".join(lines)


def to_html(title: str, markdown_graph: str) -> str:
    """Standalone page for a graph produced by `graph_markdown`."""
    body = markdown_graph.split("```mermaid", 1)[1].split("```", 1)[0]
    notes = markdown_graph.split("```mermaid", 1)[0]
    notes_html = "".join(
        f"<li>{line[2:]}</li>" for line in notes.splitlines() if line.startswith("- ")
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;color:#1b2130;background:#fff}}
h1{{font-size:20px}} ul{{font-size:14px;color:#444}} .mermaid{{overflow-x:auto}}
</style></head><body>
<h1>{title}</h1><ul>{notes_html}</ul>
<pre class="mermaid">{body}</pre>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"></script>
<script>
mermaid.initialize({{startOnLoad:true, securityLevel:"loose", flowchart:{{htmlLabels:true}}}});
</script>
</body></html>
"""
