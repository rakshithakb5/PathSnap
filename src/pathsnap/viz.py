"""Folium map builder (mandatory deliverable). Used for EDA, for prediction review, and by the API/UI.

Layers (toggle with the layer control): raw GPS points, road network near the trip,
reference route (green), predicted route (red). A legend is drawn in the corner."""
import numpy as np
import folium


def _arc_latlon(trip, arc_ids):
    out = []
    for a in arc_ids:
        u, v = trip.arcs[a]
        out.append([[trip.nodes[u, 1], trip.nodes[u, 0]], [trip.nodes[v, 1], trip.nodes[v, 0]]])
    return out


def make_map(trip, pred_arcs=None, out_html=None, network_buffer_m=300.0, max_network_arcs=6000,
             max_gps_points=3000, title=None):
    lat_c, lon_c = trip.track[:, 1].mean(), trip.track[:, 0].mean()
    m = folium.Map(location=[lat_c, lon_c], zoom_start=15, tiles="OpenStreetMap", control_scale=True)

    # --- road network, clipped to a box around the track so huge networks stay renderable ---
    lo = trip.xy.min(0) - network_buffer_m; hi = trip.xy.max(0) + network_buffer_m
    a, b = trip.arc_a, trip.arc_b
    inside = ((a >= lo) & (a <= hi)).all(1) | ((b >= lo) & (b <= hi)).all(1)
    ids = np.flatnonzero(inside)
    if len(ids) > max_network_arcs:
        ids = np.random.default_rng(0).choice(ids, max_network_arcs, replace=False)
    fg_net = folium.FeatureGroup(name=f"Road network ({len(ids)} arcs)", show=True)
    for seg in _arc_latlon(trip, ids):
        folium.PolyLine(seg, color="#9aa0a6", weight=1.5, opacity=0.7).add_to(fg_net)
    fg_net.add_to(m)

    # --- raw GPS ---
    fg_gps = folium.FeatureGroup(name="Raw GPS points", show=True)
    step = max(1, len(trip.track) // max_gps_points)
    for i in range(0, len(trip.track), step):
        folium.CircleMarker([trip.track[i, 1], trip.track[i, 0]], radius=2, color="#1a73e8",
                            fill=True, fill_opacity=0.8, weight=0,
                            tooltip=f"pt {i}  t={trip.track[i,2]:.0f}s").add_to(fg_gps)
    fg_gps.add_to(m)

    # --- reference route ---
    if trip.route is not None:
        fg_ref = folium.FeatureGroup(name="Reference route", show=True)
        for seg in _arc_latlon(trip, trip.route):
            folium.PolyLine(seg, color="#188038", weight=5, opacity=0.8).add_to(fg_ref)
        fg_ref.add_to(m)

    # --- predicted route ---
    if pred_arcs is not None and len(pred_arcs):
        fg_pred = folium.FeatureGroup(name="Predicted route", show=True)
        seen = list(dict.fromkeys(int(x) for x in pred_arcs if x >= 0))
        for seg in _arc_latlon(trip, seen):
            folium.PolyLine(seg, color="#d93025", weight=3, opacity=0.9, dash_array="6").add_to(fg_pred)
        fg_pred.add_to(m)

    m.fit_bounds([[trip.track[:, 1].min(), trip.track[:, 0].min()],
                  [trip.track[:, 1].max(), trip.track[:, 0].max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    legend = f"""<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;
      padding:8px 12px;border:1px solid #888;border-radius:6px;font-size:13px">
      <b>{title or 'Trip ' + trip.trip_id}</b><br>
      <span style="color:#1a73e8">&#9679;</span> Raw GPS<br>
      <span style="color:#9aa0a6">&#9473;</span> Road network<br>
      <span style="color:#188038">&#9473;</span> Reference route<br>
      <span style="color:#d93025">&#9548;</span> Predicted route</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    if out_html:
        m.save(str(out_html))
    return m
