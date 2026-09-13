# -*- coding: utf-8 -*-
"""
🎵 Spotify Playlist Recommendation System — Beginner-Friendly Version
=====================================================================

A simple, content-based music recommender built with pandas + scikit-learn.

What it does, step by step:
    1. Loads and cleans the Spotify dataset (removes missing / duplicate rows)
    2. Explores the audio features (summary stats + correlation heatmap)
    3. Clusters songs with K-Means (Elbow Method helps you choose k)
    4. Recommends similar songs using K-Nearest Neighbors (KNN)

How to run:
    python spotify_playlist_recommendationsystem.py                     # demo mode
    python spotify_playlist_recommendationsystem.py --track "Shape of You"
    python spotify_playlist_recommendationsystem.py --k 6 --neighbors 8

All charts are saved to  outputs/figures/  so it works everywhere
(including servers with no screen).

What was fixed compared to the original Colab notebook:
    ❌ from google.colab import files  →  ✅ reads the CSV directly from disk
    ❌ IndexError crash in the KNN step →  ✅ reset_index(drop=True) after cleaning
    ❌ display(plt.show())              →  ✅ plots are saved as PNG files
    ❌ duplicated pipeline code         →  ✅ small, commented functions + main()
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration — the only place you need to change things to experiment!
# ---------------------------------------------------------------------------
DATASET_PATH = Path(__file__).parent / "spotify_dataset.csv"
FIGURES_DIR = Path(__file__).parent / "outputs" / "figures"

# The audio features we use to describe "what a song sounds like"
AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
]

DEFAULT_K = 4          # number of clusters (try changing this!)
RANDOM_STATE = 42      # fixed seed so results are reproducible
SHOW_PLOTS = False     # set True if you run this locally and want pop-up windows


# ---------------------------------------------------------------------------
# Step 1 — Load & clean the data
# ---------------------------------------------------------------------------
def load_and_clean_data(csv_path: Path) -> pd.DataFrame:
    """Read the Spotify CSV and remove broken / duplicate rows."""
    print(f"\n[1/5] Loading dataset: {csv_path.name}")
    df = pd.read_csv(csv_path)
    rows_before = len(df)

    # Drop rows that are missing basic info we need
    df = df.dropna(subset=["track_name", "track_artist", "track_album_name"])

    # The same track can appear in many playlists — that's fine!
    # We only remove exact (track, playlist) duplicates.
    df = df.drop_duplicates(subset=["track_id", "playlist_id"])

    # ⭐ THE MOST IMPORTANT FIX in this project ⭐
    # dropna/drop_duplicates leave GAPS in the index (e.g. 0, 1, 5, 9, ...).
    # Later we turn the features into a NumPy array, which is POSITIONAL
    # (0, 1, 2, 3, ...). If we don't reset the index, row label #32828 gets
    # looked up in an array of 32,246 rows → IndexError crash!
    # reset_index(drop=True) renumbers rows 0..N-1 so labels == positions.
    df = df.reset_index(drop=True)

    # Turn the release-date text into a real date, extract the year
    df["track_album_release_date"] = pd.to_datetime(
        df["track_album_release_date"], errors="coerce"
    )
    df["release_year"] = df["track_album_release_date"].dt.year

    print(f"      ✅ Kept {len(df):,} of {rows_before:,} rows "
          f"({rows_before - len(df):,} empty/duplicate rows removed)")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Quick exploratory data analysis (EDA)
# ---------------------------------------------------------------------------
def explore_data(df: pd.DataFrame) -> None:
    """Print simple statistics and save a correlation heatmap."""
    print("\n[2/5] Exploring the data (EDA)")
    print(f"      Unique tracks : {df['track_id'].nunique():,}")
    print(f"      Unique artists: {df['track_artist'].nunique():,}")
    print(f"      Genres        : {', '.join(sorted(df['playlist_genre'].unique()))}")

    print("\n      Summary of key audio features:")
    key_features = ["track_popularity", "danceability", "energy", "valence", "tempo"]
    print(df[key_features].describe().round(3).to_string())

    # Correlation heatmap — shows which audio features move together.
    # (Values near +1 = move together, near -1 = move opposite, near 0 = unrelated)
    corr = df[AUDIO_FEATURES + ["track_popularity"]].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                annot_kws={"size": 7})
    plt.title("Correlation Matrix of Audio Features")
    plt.tight_layout()
    save_chart("correlation_heatmap.png")


# ---------------------------------------------------------------------------
# Step 3 — Find a good number of clusters with the Elbow Method
# ---------------------------------------------------------------------------
def plot_elbow_method(X_scaled) -> None:
    """
    The Elbow Method: run K-Means for k = 1..10 and plot the inertia
    (how tightly songs fit their cluster). Lower is better, but it always
    decreases with more clusters — so we look for the 'elbow' where the
    line bends and extra clusters stop helping much.
    """
    print("\n[3/5] Elbow Method — searching for a good number of clusters...")
    k_range = range(1, 11)
    inertia = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5)
        km.fit(X_scaled)
        inertia.append(km.inertia_)
        print(f"      k={k:>2}  inertia={km.inertia_:,.0f}")

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia, marker="o")
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (within-cluster distance)")
    plt.xticks(k_range)
    plt.grid(True, alpha=0.4)
    save_chart("elbow_method.png")


# ---------------------------------------------------------------------------
# Step 3b (optional) — Score every k with the Silhouette method
# ---------------------------------------------------------------------------
def find_best_k_with_silhouette(X_scaled, k_range=range(2, 11)) -> int:
    """
    The Silhouette Score measures how well each song fits its cluster:
        +1  = perfectly matched to its own cluster, far from others
         0  = right on the border between two clusters
        -1  = probably placed in the wrong cluster
    We try every k in the range and keep the one with the highest
    average score. This is more objective than eyeballing the elbow!

    (Tip: we score a random sample of 10,000 songs, not all 32,246 —
    the result is nearly identical but much faster.)
    """
    print("\n      Scoring each k with the Silhouette method...")
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE,
                        n_init=5).fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels, sample_size=10_000,
                                     random_state=RANDOM_STATE)
        best_marker = "  <- best so far" if scores[k] == max(scores.values()) else ""
        print(f"      k={k:>2}  silhouette={scores[k]:.3f}{best_marker}")

    best_k = max(scores, key=scores.get)
    print(f"      🏆 Best k = {best_k} (highest silhouette score)")

    plt.figure(figsize=(8, 5))
    plt.bar(list(scores.keys()), list(scores.values()), color="#1DB954")
    plt.axvline(best_k, color="red", linestyle="--", label=f"best k = {best_k}")
    plt.title("Silhouette Score for Each k (higher = better)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.legend()
    save_chart("silhouette_scores.png")
    return best_k


# ---------------------------------------------------------------------------
# Step 4 — Cluster the songs with K-Means
# ---------------------------------------------------------------------------
def cluster_songs(df: pd.DataFrame, X_scaled, k: int) -> pd.DataFrame:
    """
    Group songs into k clusters based on their audio features.

    Key idea: we cluster on the FULL feature set (all 10 dimensions),
    and use PCA only to draw a 2-D picture of the result.
    """
    print(f"\n[4/5] Clustering songs with K-Means (k={k})")
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
    df["cluster"] = kmeans.fit_predict(X_scaled)

    print("      Cluster sizes:")
    for cluster_id, size in df["cluster"].value_counts().sort_index().items():
        print(f"      Cluster {cluster_id}: {size:,} songs")

    # Which genres ended up in which cluster? (fraction of each cluster)
    print("\n      Top genre in each cluster:")
    genre_mix = df.groupby("cluster")["playlist_genre"].value_counts(normalize=True)
    for cluster_id in sorted(df["cluster"].unique()):
        top_genre = genre_mix.loc[cluster_id].idxmax()
        share = genre_mix.loc[cluster_id].max()
        print(f"      Cluster {cluster_id}: mostly '{top_genre}' ({share:.0%})")

    # For the picture only: squeeze 10-D data down to 2-D with PCA
    X_pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)
    plt.figure(figsize=(9, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df["cluster"],
                    palette="viridis", s=12, alpha=0.6, legend="full")
    plt.title(f"Song Clusters (k={k}, 2-D PCA view)")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.tight_layout()
    save_chart("song_clusters.png")

    # Average feature values per cluster — good for interpreting clusters
    summary = df.groupby("cluster")[AUDIO_FEATURES].mean().round(3)
    summary.to_csv(FIGURES_DIR.parent / "cluster_summary.csv")
    print(f"\n      💾 Cluster summary saved to outputs/cluster_summary.csv")
    return df


# ---------------------------------------------------------------------------
# Step 5 — Build the KNN recommender and get suggestions
# ---------------------------------------------------------------------------
def recommend_similar_songs(df: pd.DataFrame, X_scaled, track_name: str,
                            n_recommendations: int = 5) -> None:
    """
    Find the n songs 'closest' to a query song in audio-feature space.

    Why this works: every song is a point in 10-D space (one dimension per
    audio feature, all scaled). Songs that sound alike sit close together,
    so K-Nearest Neighbors literally finds the nearest points.
    """
    print(f"\n[5/5] Recommending songs similar to: '{track_name}'")

    # Find matching tracks (case-insensitive partial match)
    matches = df[df["track_name"].str.contains(track_name, case=False, na=False)]
    if matches.empty:
        print(f"      ❌ No track matching '{track_name}' found.")
        print("      💡 Try one of these popular tracks instead:")
        # nlargest on the whole table gives the same song many times
        # (once per playlist) — keep only the first copy of each track
        sample = (df.drop_duplicates("track_id")
                    .nlargest(10, "track_popularity")[["track_name", "track_artist"]])
        for _, row in sample.iterrows():
            print(f"         • {row['track_name']} — {row['track_artist']}")
        return

    # If several rows match (same song, many playlists), pick the most popular
    query_pos = matches["track_popularity"].idxmax()
    query = df.loc[query_pos]
    print(f"      🎧 Query: {query['track_name']} — {query['track_artist']} "
          f"[{query['playlist_genre']}]")

    # Ask for a bigger pool of neighbors than we need, then skip:
    #   1) the query song itself and every copy of it, and
    #   2) songs we've already shown. Careful #1: one recording can sit under
    #      DIFFERENT track_ids, so we also compare song name + artist.
    #      Careful #2: each song appears once PER PLAYLIST, so the neighbor
    #      list itself can contain duplicates — we dedupe as we walk it.
    pool_size = n_recommendations + 40
    nbrs = NearestNeighbors(n_neighbors=pool_size,
                            algorithm="ball_tree").fit(X_scaled)
    distances, indices = nbrs.kneighbors(X_scaled[[query_pos]])

    seen_ids = {query["track_id"]}
    seen_songs = {(query["track_name"], query["track_artist"])}

    print(f"\n      🎵 Top {n_recommendations} similar songs:")
    shown = 0
    for idx, dist in zip(indices[0], distances[0]):
        if shown == n_recommendations:
            break
        candidate = df.loc[idx]
        song_key = (candidate["track_name"], candidate["track_artist"])
        if candidate["track_id"] in seen_ids or song_key in seen_songs:
            continue                        # same song we already have
        seen_ids.add(candidate["track_id"])
        seen_songs.add(song_key)
        shown += 1
        print(f"       {shown}. {candidate['track_name']} — {candidate['track_artist']} "
              f"[{candidate['playlist_genre']}] (similarity distance: {dist:.2f})")


# ---------------------------------------------------------------------------
# Helper — save every chart to outputs/figures/
# ---------------------------------------------------------------------------
def save_chart(filename: str) -> None:
    """Save the current figure as a PNG (and optionally show it)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"      📊 Chart saved: outputs/figures/{filename}")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Put it all together
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="🎵 Spotify song recommender for beginners")
    parser.add_argument("--track", type=str,
                        default="City Of Lights - Official Radio Edit",
                        help="Song name to base recommendations on")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help=f"Number of clusters (default {DEFAULT_K})")
    parser.add_argument("--auto-k", action="store_true",
                        help="Score k=2..10 with the Silhouette method "
                             "and pick the best one automatically")
    parser.add_argument("--neighbors", type=int, default=5,
                        help="How many similar songs to show (default 5)")
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        raise SystemExit(
            f"❌ Could not find {DATASET_PATH}\n"
            f"   Make sure spotify_dataset.csv sits next to this script.")

    df = load_and_clean_data(DATASET_PATH)

    # Scale features so each one contributes fairly to distances.
    # (Without scaling, duration_ms (~200,000) would drown out valence (0–1)!)
    print("\n      Scaling audio features with StandardScaler...")
    X_scaled = StandardScaler().fit_transform(df[AUDIO_FEATURES].values)

    explore_data(df)
    plot_elbow_method(X_scaled)

    if args.auto_k:
        k = find_best_k_with_silhouette(X_scaled)
    else:
        k = args.k

    df = cluster_songs(df, X_scaled, k=k)
    recommend_similar_songs(df, X_scaled, args.track, args.neighbors)

    print("\n✅ Done! Check outputs/figures/ for charts "
          "and outputs/cluster_summary.csv for cluster averages.")


if __name__ == "__main__":
    main()
