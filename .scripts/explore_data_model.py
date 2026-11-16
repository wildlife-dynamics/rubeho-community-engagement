"""
Explore Rubeho's "Stakeholder meeting" (community_meeting) data model.

Pulls the live v2 schema + all events, then reports:
  - every schema field (incl. conditional ones) with title, type, choice list
  - how often each event_details key is filled
  - value counts for the grouper candidates (meeting type, stakeholder, cluster, village)
  - village names found in events vs. the village boundary shapefile

Run with (uses the Eden compile env, which has ecoscope installed):
    ../eden-community_engagement/.pixi/envs/compile/bin/python .scripts/explore_data_model.py

Writes to .scripts/data/: schema.json, events.parquet (details flattened)
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from ecoscope_workflows_ext_ecoscope.connections import EarthRangerConnection

CONNECTION = "rubeho"
EVENT_TYPE = "community_meeting"  # displayed as "Stakeholder meeting"
VILLAGES = sys.argv[1] if len(sys.argv) > 1 else "zip:///Users/zak/Downloads/Rubeho Village Boundaries.zip"
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

er = EarthRangerConnection.from_named_connection(CONNECTION).get_client()


# --- Schema -----------------------------------------------------------------
def choices(field_def):
    """Resolve $ref choice lists to {value: title}."""
    any_of = field_def.get("anyOf") or field_def.get("items", {}).get("anyOf") or []
    out = {}
    for d in any_of:
        ref = d.get("$ref")
        if not ref:
            continue
        ref_schema = er._get(ref.lstrip("/").removeprefix("api/v2.0/"))
        for c in ref_schema.get("oneOf", []):
            out[c["const"]] = c.get("title", c["const"])
        out.update(zip(ref_schema.get("enum", []), ref_schema.get("enumNames", [])))
        for val, meta in ref_schema.get("x-enumExtra", {}).items():
            out.setdefault(val, meta.get("display", val) if isinstance(meta, dict) else val)
    return out


with er._use_v2_api():
    schema = er._get(f"activity/eventtypes/{EVENT_TYPE}/schema")["json"]
    props = dict(schema["properties"])
    for rule in schema.get("allOf", []):  # conditional fields live under allOf[].then
        for k, v in rule.get("then", {}).get("properties", {}).items():
            props.setdefault(k, v)
    for v in props.values():
        v["_choices"] = choices(v)

(OUT / "schema.json").write_text(json.dumps(props, indent=2))

print(f"\n=== SCHEMA: {len(props)} fields ({len(schema['properties'])} top-level) ===")
for k, v in props.items():
    ch = v["_choices"]
    sample = ", ".join(list(ch.values())[:6]) + (" …" if len(ch) > 6 else "")
    print(f"  {k:55s} {str(v.get('type')):7s} {v.get('title', '')!r:50s} {len(ch) or '':>3} {sample}")


# --- Events -----------------------------------------------------------------
et_id = dict(zip(er.get_event_types()["value"], er.get_event_types()["id"]))[EVENT_TYPE]
events = er.get_events(event_type=[et_id], include_details=True, include_notes=False)
print(f"\n=== EVENTS: {len(events)} ===")
if events.empty:
    sys.exit("No events — stopping.")
print(events["state"].value_counts().to_string())
print("time range:", events["time"].min(), "→", events["time"].max())
print("with location:", events.geometry.notna().sum() - events.geometry.is_empty.sum())

details = pd.json_normalize(events["event_details"].tolist(), max_level=0)
details.index = events.index
events = events.reset_index()
details.index = events.index
flat = pd.concat([events[["id", "serial_number", "time", "state", "geometry"]], details], axis=1)
flat.drop(columns="geometry").assign(
    lon=events.geometry.x, lat=events.geometry.y
).astype({c: str for c in details.columns if details[c].map(type).isin([list, dict]).any()}).to_parquet(
    OUT / "events.parquet"
)

print("\n=== DETAIL FIELD FILL RATE ===")
filled = details.notna() & details.astype(str).ne("") & details.astype(str).ne("[]")
for k, n in filled.sum().sort_values(ascending=False).items():
    print(f"  {k:55s} {n:4d}/{len(details)}  {props.get(k, {}).get('title', '(not in schema)')}")


# --- Grouper candidates -----------------------------------------------------
def label(field, val):
    ch = props.get(field, {}).get("_choices", {})
    return [ch.get(x, x) for x in val] if isinstance(val, list) else ch.get(val, val)


keywords = ("type", "stakeholder", "cluster", "level_one", "level_two", "level_three", "village")
candidates = [c for c in details.columns if any(w in c.lower() for w in keywords)]
print("\n=== GROUPER CANDIDATE VALUES ===")
for c in candidates:
    vals = details[c].dropna().map(lambda v: label(c, v)).explode()
    if vals.empty:
        continue
    print(f"\n-- {c}  ({props.get(c, {}).get('title', '?')})")
    print(vals.value_counts().to_string())

# Village = the one meeting_location_level_two_<cluster> field filled for that event
# (level_three_* are free-text sub-villages, level_one is the forest cluster)
loc_cols = [c for c in details.columns if c.startswith("meeting_location_level_two_") and not c.endswith("_other")]
village = details[loc_cols].apply(
    lambda r: next((label(c, r[c]) for c in loc_cols if pd.notna(r[c]) and r[c] != ""), None), axis=1
)

print("\n=== VILLAGE NAMES: events vs shapefile ===")
shp = gpd.read_file(VILLAGES)
shp_names = set(shp["Name"])
shp_parts = {p.strip().lower(): n for n in shp_names for p in n.split("/")}
for name, n in village.value_counts().items():
    hit = name if name in shp_names else shp_parts.get(str(name).strip().lower())
    print(f"  {n:4d}  {name!s:35s} -> {hit or 'NO MATCH'}")
print(f"  {village.isna().sum():4d}  (no village recorded)")
print("\nShapefile villages with no meetings:", sorted(shp_names - set(village.dropna()) - {
    shp_parts.get(str(v).strip().lower()) for v in village.dropna()}))


# Alternative: which polygon does the event's GPS point fall in?
pts = gpd.GeoDataFrame({"form_village": village}, geometry=events.geometry.values, crs=4326)
hit = gpd.sjoin(pts, shp[["Name", "geometry"]], how="left", predicate="within")
hit = hit[~hit.index.duplicated()]
print("\n=== GPS point-in-polygon vs form village ===")
print(f"  points inside a village polygon: {hit['Name'].notna().sum()}/{len(hit)}")
print(hit.groupby(["form_village", "Name"], dropna=False).size().to_string())

print(f"\nSaved {OUT/'schema.json'} and {OUT/'events.parquet'}")
