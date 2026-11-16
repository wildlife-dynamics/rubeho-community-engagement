"""
Generate synthetic mock-IO events for the workflow tests (mock-io/*.parquet).

Only the choice lists come from Rubeho: forest cluster / village pairs (sampled with their real
frequencies), meeting types, stakeholder levels and topics. Everything else is made up: ids,
serial numbers, 2025 dates, attendance, reporters. Notes, objectives, team members and other free
text are left out. Points are random locations inside the chosen village polygon.

Run with an env that has ecoscope-platform + ecoscope-workflows-ext-eden (>= the Rubeho tasks):
    ../eden/src/ecoscope-workflows-ext-eden/.pixi/envs/test-py312/bin/python .scripts/make_mock_io.py
"""
import random
from pathlib import Path

import geopandas as gpd
import pandas as pd
from ecoscope.platform.connections import EarthRangerConnection
from ecoscope_workflows_ext_eden.tasks import load_rubeho_villages, resolve_village
from shapely.geometry import Point

N_RESOLVED, N_ACTIVE = 48, 12
OUT = Path(__file__).parent.parent / "mock-io"
rng = random.Random(42)

er = EarthRangerConnection.from_named_connection("rubeho").get_client()


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
    schema = er._get("activity/eventtypes/community_meeting/schema")["json"]
    props = dict(schema["properties"])
    for rule in schema.get("allOf", []):
        props.update({k: v for k, v in rule.get("then", {}).get("properties", {}).items() if k not in props})
    meeting_types = choices(props["meeting_type"])
    stakeholders = choices(props["primary_stakeholder_level_one"])
    topics = choices(props["topics"]["items"]["properties"]["topic"])
    clusters = choices(props["meeting_location_level_one"])
    village_labels = {k: choices(v) for k, v in props.items() if k.startswith("meeting_location_level_two_")}

# Real (cluster, village field, village key) combinations, to sample with real frequencies
et = er.get_event_types()
real = er.get_events(event_type=[dict(zip(et["value"], et["id"]))["community_meeting"]], include_details=True)
pairs = [
    (
        d.get("meeting_location_level_one"),
        next(((k, v) for k, v in d.items() if k in village_labels and village_labels[k] and v), (None, None)),
    )
    for d in real["event_details"]
]
polygons = load_rubeho_villages()
lookup = resolve_village(  # where each village key lands, so points go inside the right polygon
    df=pd.DataFrame({"event_details": [{k: village_labels[k].get(v, v)} if k else {} for _, (k, v) in pairs]}),
    polygons=polygons,
)["village"].tolist()


def random_point(village):
    poly = polygons.loc[polygons["Name"] == village].geometry
    geom = poly.iloc[0] if len(poly) else polygons.union_all()
    minx, miny, maxx, maxy = geom.bounds
    while True:
        p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if geom.contains(p):
            return p


raw_rows, label_rows = [], []
for i in range(N_RESOLVED + N_ACTIVE):
    j = rng.randrange(len(pairs))
    cluster, (vfield, vkey) = pairs[j]
    total = rng.randint(10, 60)
    men = rng.randint(0, total)
    youth_m, youth_f = rng.randint(0, men // 2), rng.randint(0, (total - men) // 2)
    mtypes = rng.sample(list(meeting_types), rng.choice([0, 1, 1, 2]))
    stake = rng.choice([None, *stakeholders])
    tkeys = rng.sample(list(topics), rng.randint(1, 4))
    point = random_point(lookup[j])
    raw = {
        "meeting_location_level_one": cluster,
        "meeting_type": mtypes,
        "primary_stakeholder_level_one": stake,
        "participants_total": total,
        "participants_men": men,
        "participants_women": total - men,
        "participants_youth_male": youth_m,
        "participants_youth_female": youth_f,
        "topics": [{"topic": t} for t in tkeys],
        **({vfield: vkey} if vfield else {}),
    }
    labels = {
        **raw,
        "meeting_location_level_one": clusters.get(cluster, cluster),
        "meeting_type": [meeting_types[m] for m in mtypes],
        "primary_stakeholder_level_one": stakeholders.get(stake, stake),
        "topics": [{"topic": topics[t]} for t in tkeys],
        **({vfield: village_labels[vfield].get(vkey, vkey)} if vfield else {}),
    }
    base = {
        "id": f"00000000-0000-4000-8000-{i:012d}",
        "time": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(minutes=rng.randrange(365 * 24 * 60)),
        "event_type": "community_meeting",
        "location": {"latitude": point.y, "longitude": point.x},
        "geometry": point,
        "state": "resolved" if i < N_RESOLVED else "active",
        "serial_number": 100000 + i,
        "event_type_display": "Stakeholder meeting",
        "event_category": "None",
        "reported_by": {"name": "Mock Reporter"},
    }
    raw_rows.append({**base, "event_details": raw})
    label_rows.append({**base, "event_details": labels})

OUT.mkdir(exist_ok=True)
events = gpd.GeoDataFrame(raw_rows, geometry="geometry", crs=4326)
processed = gpd.GeoDataFrame(label_rows, geometry="geometry", crs=4326)
events.to_parquet(OUT / "get-events.example-return.parquet")
processed[processed["state"] == "resolved"].to_parquet(OUT / "process-events-details.example-return.parquet")
print(f"{len(events)} events, {N_RESOLVED} resolved -> {OUT}")
print(pd.Series(lookup).value_counts().head(8).to_dict())
