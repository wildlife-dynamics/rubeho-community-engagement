"""
Preview the full Rubeho community engagement dashboard before building the workflow.

Same outputs as Eden/Olokeri (meetings, participants, median, box plot, gender pie,
age pie, topics bar chart, choropleth) for each grouper:
    none · quarter · meeting type · primary stakeholder · forest cluster · village

Village = form field (meeting_location_level_two_<cluster>) matched to the shapefile,
falling back to the polygon the GPS point falls in, else "Unmatched".

Run with:
    ../eden-community_engagement/.pixi/envs/compile/bin/python .scripts/preview_dashboard.py [villages.zip]

Output: .scripts/output/dashboard.html (+ maps/*.html, open the dashboard in a browser)
"""
import html
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from ecoscope_workflows_ext_ecoscope.tasks.results import draw_pie_chart
from ecoscope_workflows_ext_ecoscope.tasks.transformation import apply_color_map
from ecoscope_workflows_ext_custom.tasks.results import draw_boxplot
from ecoscope_workflows_ext_custom.tasks.results._map import (
    PolygonLayerStyle,
    ScatterplotLayerStyle,
    TextLayerStyle,
    create_polygon_layer_pydeck,
    create_scatterplot_layer,
    create_text_layer_pydeck,
    draw_map,
    set_base_maps_pydeck,
)
from ecoscope_workflows_ext_eden.tasks import apply_choropleth_colormap, create_viewstate_gdf, draw_topic_bar_chart
from ecoscope_workflows_ext_ecoscope.connections import EarthRangerConnection

CONNECTION = "rubeho"
EVENT_TYPE = "community_meeting"  # "Stakeholder meeting"
SINCE, UNTIL = "2025-01-01T00:00:00Z", pd.Timestamp.now(tz="UTC").isoformat()
TZ = "Africa/Dar_es_Salaam"
VILLAGES = sys.argv[1] if len(sys.argv) > 1 else "zip:///Users/zak/Downloads/Rubeho Village Boundaries.zip"
# Form village label -> shapefile Name, where spelling differs
VILLAGE_ALIASES = {"mangaliza": "Kilambo / Matuli / Mangalisa"}  # keys are norm()-ed
OUT = Path(__file__).parent / "output"
(OUT / "maps").mkdir(parents=True, exist_ok=True)

er = EarthRangerConnection.from_named_connection(CONNECTION).get_client()


# --- Schema choice labels -----------------------------------------------------
def choices(field_def):
    out = {}
    for d in field_def.get("anyOf") or field_def.get("items", {}).get("anyOf") or []:
        if ref := d.get("$ref"):
            r = er._get(ref.lstrip("/").removeprefix("api/v2.0/"))
            out.update({c["const"]: c.get("title", c["const"]) for c in r.get("oneOf", [])})
            out.update(zip(r.get("enum", []), r.get("enumNames", [])))
            for val, meta in r.get("x-enumExtra", {}).items():
                out.setdefault(val, meta.get("display", val) if isinstance(meta, dict) else val)
    return out


with er._use_v2_api():
    schema = er._get(f"activity/eventtypes/{EVENT_TYPE}/schema")["json"]
    props = dict(schema["properties"])
    for rule in schema.get("allOf", []):
        for k, v in rule.get("then", {}).get("properties", {}).items():
            props.setdefault(k, v)
    labels = {k: choices(v) for k, v in props.items()}
    labels["topic"] = choices(props["topics"]["items"]["properties"]["topic"])


def lab(field, v):
    return labels.get(field, {}).get(v, v)


# --- Events -------------------------------------------------------------------
et_id = dict(zip(er.get_event_types()["value"], er.get_event_types()["id"]))[EVENT_TYPE]
events = er.get_events(event_type=[et_id], since=SINCE, until=UNTIL, include_details=True, drop_null_geometry=False)
events = events[events["state"] == "resolved"].reset_index()
print(f"{len(events)} resolved meetings {events['time'].min()} → {events['time'].max()}")
d = events["event_details"]

ev = gpd.GeoDataFrame(
    {
        "time": events["time"].dt.tz_convert(TZ),
        **{c: pd.to_numeric(d.str.get(c), errors="coerce") for c in
           ["participants_total", "participants_men", "participants_women",
            "participants_youth_male", "participants_youth_female"]},
        "forest_cluster": d.map(lambda x: lab("meeting_location_level_one", x.get("meeting_location_level_one"))),
        "meeting_type": d.map(lambda x: [lab("meeting_type", t) for t in x.get("meeting_type") or []]),
        "primary_stakeholder": d.map(
            lambda x: lab("primary_stakeholder_level_one", x.get("primary_stakeholder_level_one"))
        ),
        # draw_topic_bar_chart reads topics from an event_details dict
        # topic choices have no display names in ER (enum only), so prettify the key
        "event_details": d.map(lambda x: {"topics": [
            {"topic": str(lab("topic", t.get("topic"))).replace("_", " ").capitalize()} for t in x.get("topics") or []]}),
    },
    geometry=events.geometry.values,
    crs=4326,
)
ev["quarter"] = ev["time"].dt.tz_localize(None).dt.to_period("Q").astype(str).str.replace("Q", " Q")
ev["forest_cluster"] = ev["forest_cluster"].fillna("Unspecified")
ev["primary_stakeholder"] = ev["primary_stakeholder"].fillna("Unspecified")
ev["meeting_type"] = ev["meeting_type"].map(lambda v: v or ["Unspecified"])

# --- Village resolution -------------------------------------------------------
shp = gpd.read_file(VILLAGES).to_crs(4326)
shp_names = set(shp["Name"])


def norm(s):
    return re.sub(r"[^a-z]", "", s.lower())  # ignore case, spaces, apostrophes: Udung'hu == udunghu


parts = {norm(p): n for n in shp_names for p in n.split("/")}


def form_village(x):
    k = next((k for k in x if re.fullmatch(r"meeting_location_level_two_[a-z]+", k) and x[k]), None)
    return lab(k, x[k]) if k else None


def to_polygon(name):
    if name is None:
        return None
    return VILLAGE_ALIASES.get(norm(name)) or parts.get(norm(name))


ev["village_form"] = d.map(form_village)
gps = gpd.sjoin(ev[["geometry"]], shp[["Name", "geometry"]], how="left", predicate="within")
gps = gps[~gps.index.duplicated()]["Name"]
ev["village"] = ev["village_form"].map(to_polygon).fillna(gps).fillna("Unmatched")
ev["village_source"] = ev["village_form"].map(to_polygon).notna().map({True: "form", False: "gps"})
ev.loc[ev["village"] == "Unmatched", "village_source"] = "none"
print("village source:", ev["village_source"].value_counts().to_dict())
print("unmatched:", ev.loc[ev["village"] == "Unmatched", "village_form"].tolist())


# --- Widgets per group --------------------------------------------------------
def split_pie(g, parts_):
    df = pd.DataFrame({"category": list(parts_), "count": [g[c].sum() if isinstance(c, str) else c(g) for c in parts_.values()]})
    df = df[df["count"] > 0]
    df = apply_color_map(df=df, input_column_name="category", output_column_name="color", colormap="Paired")
    return draw_pie_chart.validate()(dataframe=df, value_column="count", label_column="category", color_column="color",
                          plot_style={"textinfo": "label+percent"}, layout_style={"title": None})


def choropleth(g, key):
    counts = g["village"].value_counts().rename("meeting_count")
    poly = shp[["Name", "geometry"]].merge(counts, left_on="Name", right_index=True, how="left")
    poly["meeting_count"] = poly["meeting_count"].fillna(0).astype(int)
    poly = apply_choropleth_colormap(df=poly, count_column="meeting_count", colormap="Blues",
                                     fill_color_column="fill_color", label_column="choropleth_label", max_classes=5)
    cent = gpd.GeoDataFrame(
        {"label": poly["Name"] + "\n" + poly["meeting_count"].astype(str)},
        geometry=poly.geometry.representative_point(), crs=4326,
    )
    pts = g[g.geometry.notna()][["geometry"]].copy()
    pts["c"] = (g["village"] == "Unmatched").map({True: [210, 30, 30, 220], False: [255, 140, 0, 200]})
    layers = [
        create_polygon_layer_pydeck(geodataframe=poly, layer_style=PolygonLayerStyle(
            get_fill_color="fill_color", get_line_color=[0, 0, 0, 200], line_width_min_pixels=1,
            stroked=True, opacity=0.7), legend=None),
        create_scatterplot_layer(geodataframe=pts, layer_style=ScatterplotLayerStyle(
            get_fill_color="c", radius_min_pixels=5, radius_max_pixels=8, opacity=0.8, stroked=True,
            get_line_color=[255, 255, 255, 180], line_width_min_pixels=1), legend=None),
        create_text_layer_pydeck(geodataframe=cent, layer_style=TextLayerStyle(
            get_text="label", get_size=10, get_color=[20, 20, 20, 240], background=False)),
    ]
    path = OUT / "maps" / f"{key}.html"
    path.write_text(draw_map(geo_layers=layers, tile_layers=set_base_maps_pydeck(),
                             view_state=create_viewstate_gdf(gdf=shp, padding=0.1), max_zoom=20))
    return f"maps/{path.name}"


def slim(chart_html):
    """Swap the ~4.6MB inline plotly.js bundle for a CDN tag (else the page is ~900MB)."""
    v = re.search(r"plotly\.js v([\d.]+)", chart_html)
    if not v:
        return chart_html
    return re.sub(r"<script[^>]*>/\*\*\s*\* plotly\.js v.*?</script>",
                  f'<script src="https://cdn.plot.ly/plotly-{v[1]}.min.js"></script>', chart_html, count=1, flags=re.S)


def widgets(g, key):
    w = {
        "meetings": int(len(g)),
        "participants": int(g["participants_total"].sum()),
        "median": float(g["participants_total"].median()),
        "box": draw_boxplot.validate()(dataframe=g[["participants_total"]], y_column="participants_total",
                            show_points="outliers", layout_style={"yaxis": {"title": "Participants"},
                                                                  "plot_bgcolor": "white"}),
        "gender": split_pie(g, {"Men": "participants_men", "Women": "participants_women"}),
        "age": split_pie(g, {
            "Youth male": "participants_youth_male", "Youth female": "participants_youth_female",
            "Adults (36+)": lambda x: (x["participants_total"] - x["participants_youth_male"]
                                       - x["participants_youth_female"]).clip(lower=0).sum()}),
        "topics": draw_topic_bar_chart(df=g, location_column="forest_cluster", title=None,
                                       x_axis_title="Forest cluster", colormap="Paired"),
        "map": choropleth(g, key),
    }
    return {k: slim(v) if k in ("box", "gender", "age", "topics") else v for k, v in w.items()}


GROUPERS = {"All meetings": None, "Quarter": "quarter", "Meeting type": "meeting_type",
            "Primary stakeholder": "primary_stakeholder", "Forest cluster": "forest_cluster", "Village": "village"}
data = {}
for title, col in GROUPERS.items():
    src = ev if col != "meeting_type" else ev.explode("meeting_type")  # multi-select: a meeting counts in each type
    groups = {"All": src} if col is None else dict(tuple(src.groupby(col)))
    data[title] = {}
    for name, g in groups.items():
        print(f"  {title}: {name} ({len(g)})")
        data[title][str(name)] = widgets(g, re.sub(r"\W+", "_", f"{title}_{name}").lower())

# Self-check: every grouper (except multi-select meeting type) partitions the same meetings
for t in ["Quarter", "Primary stakeholder", "Forest cluster", "Village"]:
    assert sum(v["meetings"] for v in data[t].values()) == len(ev), t

# --- Page ---------------------------------------------------------------------
page = """<!doctype html><html><head><meta charset="utf-8"><title>Rubeho Engagement Preview</title>
<style>
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,sans-serif;background:#f0f2f5}
header{padding:12px 20px;background:#1c2b4a;color:#fff;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
header b{margin-right:auto}select{padding:4px 8px;font-size:14px}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;padding:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px;padding:0 12px 12px}
.card{background:#fff;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.1);overflow:hidden}
.card h3{margin:0;padding:8px 14px;font-size:12px;text-transform:uppercase;color:#444;border-bottom:1px solid #eee}
.stat{padding:14px;font-size:28px;font-weight:600}iframe{width:100%;height:380px;border:0}
.map iframe{height:560px}.wide{grid-column:1/-1}
</style></head><body>
<header><b>Rubeho Stakeholder Meetings — Preview</b>
<label>Group by <select id="grp"></select></label><label>Group <select id="val"></select></label></header>
<div class="stats">
 <div class="card"><h3>Total meetings</h3><div class="stat" id="meetings"></div></div>
 <div class="card"><h3>Total participants</h3><div class="stat" id="participants"></div></div>
 <div class="card"><h3>Median participants / meeting</h3><div class="stat" id="median"></div></div>
</div>
<div class="grid">
 <div class="card"><h3>Participants per meeting</h3><iframe id="box"></iframe></div>
 <div class="card"><h3>Gender split</h3><iframe id="gender"></iframe></div>
 <div class="card"><h3>Age split</h3><iframe id="age"></iframe></div>
 <div class="card"><h3>Topics by forest cluster</h3><iframe id="topics"></iframe></div>
 <div class="card map wide"><h3>Meetings per village</h3><iframe id="map"></iframe></div>
</div>
<script>
const D = __DATA__;
const grp = document.getElementById('grp'), val = document.getElementById('val');
Object.keys(D).forEach(k => grp.add(new Option(k, k)));
function fillVals(){ val.innerHTML=''; Object.keys(D[grp.value]).forEach(k => val.add(new Option(`${k} (${D[grp.value][k].meetings})`, k))); show(); }
function show(){
  const w = D[grp.value][val.value];
  ['meetings','participants','median'].forEach(k => document.getElementById(k).textContent = w[k]);
  ['box','gender','age','topics'].forEach(k => document.getElementById(k).srcdoc = w[k]);
  document.getElementById('map').src = w.map;
}
grp.onchange = fillVals; val.onchange = show; fillVals();
</script></body></html>"""
(OUT / "dashboard.html").write_text(page.replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
ev.drop(columns="event_details").assign(meeting_type=ev["meeting_type"].str.join(", ")).to_parquet(OUT / "events_resolved.parquet")
print(f"\nOpen {OUT/'dashboard.html'}")
