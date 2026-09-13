# -*- coding: utf-8 -*-
"""
🌐 Web App for the Spotify Playlist Recommendation System
=========================================================

A tiny Flask website so you can try the recommender in your browser
instead of the terminal.

How to run:
    pip install flask
    python web_app.py
Then open:  http://127.0.0.1:5000

How it works (nice example of code reuse!):
    Instead of copy-pasting the ML code, this file IMPORTS the functions
    from spotify_playlist_recommendationsystem.py, cleans the data the
    same way, and trains the K-Means + KNN models once at startup.
    Every search afterwards is instant.
"""

from flask import Flask, request, render_template_string

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Reuse the exact same logic as the main script — no duplication!
from spotify_playlist_recommendationsystem import (
    AUDIO_FEATURES,
    DATASET_PATH,
    DEFAULT_K,
    RANDOM_STATE,
    load_and_clean_data,
)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Train everything once, at startup (takes ~10 seconds)
# ---------------------------------------------------------------------------
print("⏳ Loading dataset and training models (one-time, ~10 seconds)...")
df = load_and_clean_data(DATASET_PATH)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[AUDIO_FEATURES].values)

kmeans = KMeans(n_clusters=DEFAULT_K, random_state=RANDOM_STATE, n_init="auto")
df["cluster"] = kmeans.fit_predict(X_scaled)

# 45 neighbors is enough to filter out duplicates and still return the top 5
nbrs = NearestNeighbors(n_neighbors=45, algorithm="ball_tree").fit(X_scaled)
print("✅ Ready! Open http://127.0.0.1:5000 in your browser.")


# ---------------------------------------------------------------------------
# The recommendation logic (same dedupe trick as the main script)
# ---------------------------------------------------------------------------
def get_recommendations(track_query: str, n: int = 5):
    """Return (query_song, [recommendations]) or (None, popular_songs)."""
    matches = df[df["track_name"].str.contains(track_query, case=False, na=False)]
    if matches.empty:
        popular = (df.drop_duplicates("track_id")
                     .nlargest(8, "track_popularity"))
        return None, [
            {"name": r["track_name"], "artist": r["track_artist"],
             "genre": r["playlist_genre"]}
            for _, r in popular.iterrows()
        ]

    query_pos = matches["track_popularity"].idxmax()
    query = df.loc[query_pos]

    distances, indices = nbrs.kneighbors(X_scaled[[query_pos]])

    seen_ids = {query["track_id"]}
    seen_songs = {(query["track_name"], query["track_artist"])}
    recs = []
    for idx, dist in zip(indices[0], distances[0]):
        if len(recs) == n:
            break
        cand = df.loc[idx]
        key = (cand["track_name"], cand["track_artist"])
        if cand["track_id"] in seen_ids or key in seen_songs:
            continue                       # same song we already have
        seen_ids.add(cand["track_id"])
        seen_songs.add(key)
        # Convert distance -> a friendlier "% match" number for the UI
        match_pct = max(0, round(100 * (1 - min(dist / 5.0, 1.0))))
        recs.append({
            "name": cand["track_name"], "artist": cand["track_artist"],
            "genre": cand["playlist_genre"], "distance": f"{dist:.2f}",
            "match": match_pct,
        })
    return query, recs


# ---------------------------------------------------------------------------
# The website (one HTML template, styled inline — no external files needed)
# ---------------------------------------------------------------------------
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🎵 Spotify Song Recommender</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background: linear-gradient(160deg, #191414 0%, #1db95433 100%), #191414;
    color:#fff; min-height:100vh; padding:40px 16px;
  }
  .container { max-width:760px; margin:0 auto; }
  h1 { font-size:2rem; margin-bottom:4px; }
  h1 span { color:#1DB954; }
  .subtitle { color:#b3b3b3; margin-bottom:28px; }
  .searchbox { display:flex; gap:8px; margin-bottom:32px; }
  input[type=text] {
    flex:1; padding:14px 18px; border-radius:28px; border:2px solid #2a2a2a;
    background:#282828; color:#fff; font-size:1rem; outline:none;
  }
  input[type=text]:focus { border-color:#1DB954; }
  button {
    padding:14px 26px; border-radius:28px; border:none; cursor:pointer;
    background:#1DB954; color:#191414; font-weight:700; font-size:1rem;
  }
  button:hover { transform:scale(1.04); }
  .card {
    background:#212121; border-radius:12px; padding:18px 20px;
    margin-bottom:12px; display:flex; align-items:center; gap:16px;
    border:1px solid #2a2a2a;
  }
  .card.query { border-color:#1DB954; background:#1db95414; }
  .rank {
    min-width:42px; height:42px; border-radius:50%; background:#1DB954;
    color:#191414; font-weight:800; display:flex; align-items:center;
    justify-content:center; font-size:1.05rem;
  }
  .note {
    min-width:42px; height:42px; border-radius:50%; background:#333;
    display:flex; align-items:center; justify-content:center; font-size:1.2rem;
  }
  .info { flex:1; min-width:0; }
  .name { font-weight:700; font-size:1.02rem; overflow:hidden;
          text-overflow:ellipsis; white-space:nowrap; }
  .artist { color:#b3b3b3; font-size:.9rem; margin-top:2px; }
  .badge {
    display:inline-block; background:#333; color:#1DB954; border-radius:12px;
    padding:2px 10px; font-size:.75rem; font-weight:700; margin-top:6px;
    text-transform:uppercase; letter-spacing:.5px;
  }
  .match { text-align:right; min-width:90px; }
  .match .pct { font-weight:800; color:#1DB954; font-size:1.05rem; }
  .match .lbl { color:#7a7a7a; font-size:.72rem; text-transform:uppercase; }
  .bar { background:#333; border-radius:4px; height:5px; margin-top:5px; }
  .bar > div { background:#1DB954; height:5px; border-radius:4px; }
  .section-title { margin:26px 0 14px; color:#b3b3b3; font-size:.85rem;
                   text-transform:uppercase; letter-spacing:1.5px; }
  .hint { color:#7a7a7a; font-size:.85rem; margin-top:8px; }
  a { color:#1DB954; text-decoration:none; }
</style>
</head>
<body>
<div class="container">
  <h1>🎵 Spotify Song <span>Recommender</span></h1>
  <p class="subtitle">K-Means clustering + KNN on {{ total_tracks }} tracks —
     type a song, get 5 similar ones.</p>

  <form class="searchbox" method="get" action="/">
    <input type="text" name="track" placeholder="Try: Shape of You, Believer, Dance Monkey…"
           value="{{ query_text }}" autofocus>
    <button type="submit">Recommend</button>
  </form>

  {% if query_text %}
    {% if query %}
      <div class="card query">
        <div class="note">🎧</div>
        <div class="info">
          <div class="name">{{ query['track_name'] }}</div>
          <div class="artist">{{ query['track_artist'] }} · cluster #{{
              query['cluster'] }}</div>
          <span class="badge">{{ query['playlist_genre'] }}</span>
        </div>
      </div>
      <div class="section-title">Because you like this…</div>
      {% for r in recs %}
        <div class="card">
          <div class="rank">{{ loop.index }}</div>
          <div class="info">
            <div class="name">{{ r['name'] }}</div>
            <div class="artist">{{ r['artist'] }} · distance {{ r['distance'] }}</div>
            <span class="badge">{{ r['genre'] }}</span>
          </div>
          <div class="match">
            <div class="pct">{{ r['match'] }}%</div>
            <div class="lbl">match</div>
            <div class="bar"><div style="width: {{ r['match'] }}%"></div></div>
          </div>
        </div>
      {% endfor %}
    {% else %}
      <div class="section-title">No song called “{{ query_text }}” — try one of these:</div>
      {% for s in recs %}
        <div class="card">
          <div class="note">💡</div>
          <div class="info">
            <div class="name">{{ s['name'] }}</div>
            <div class="artist">{{ s['artist'] }}</div>
            <span class="badge">{{ s['genre'] }}</span>
          </div>
        </div>
      {% endfor %}
    {% endif %}
    <p class="hint"><a href="/">← clear search</a></p>
  {% endif %}
</div>
</body>
</html>
"""


@app.route("/")
def home():
    query_text = request.args.get("track", "").strip()
    query, recs = None, []
    if query_text:
        query, recs = get_recommendations(query_text)
        if query is not None:
            # Jinja can't test the truthiness of a pandas Series
            # (it raises "truth value is ambiguous") — use a plain dict
            query = query.to_dict()
    return render_template_string(
        TEMPLATE, query_text=query_text, query=query, recs=recs,
        total_tracks=f"{len(df):,}")


if __name__ == "__main__":
    # host="0.0.0.0" makes the site reachable from other devices on your network
    app.run(host="0.0.0.0", port=5000, debug=False)
