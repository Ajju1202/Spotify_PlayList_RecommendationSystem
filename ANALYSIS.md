# 📋 Repository Analysis — Spotify_PlayList_RecommendationSystem

> ## ✅ UPDATE APPLIED — 2026-09-13
> All findings below have been **fixed in the code** in the commit
> *"Make project beginner-friendly"*. See the rewritten
> `spotify_playlist_recommendationsystem.py`, `README.md`, `requirements.txt`,
> and `architecture.txt`. This document is kept as the record of the
> original state of the repo.

**Repo:** https://github.com/Ajju1202/Spotify_PlayList_RecommendationSystem
**Author:** Arjun (Ajju1202) · **License:** MIT · **Size:** ~7.7 MB (mostly dataset)
**Analyzed:** 2026-09-13 · Verified by executing the full pipeline locally

---

## 1. What the project is

A **content-based music recommendation system** built in a Google Colab notebook (later exported to `.py`). It uses the well-known public **Spotify 32k-tracks dataset** (23 columns of track metadata + audio features, 6 playlist genres, releases from 1957–2020) to:

1. Explore and visualize audio-feature relationships (EDA)
2. Cluster songs with **K-Means** (Elbow Method, PCA visualization)
3. Recommend similar songs with **K-Nearest Neighbors (KNN)** on scaled audio features

> ⚠️ Note: despite the README's wording ("KNN collaborative filtering"), this is purely **content-based filtering** — it uses audio features, not user interaction data.

## 2. Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3 (Colab notebook export) |
| Data | pandas 2.2, numpy 1.26 |
| ML | scikit-learn 1.4 — `StandardScaler`, `PCA`, `KMeans`, `NearestNeighbors` |
| Visualization | matplotlib, seaborn |
| Environment | Google Colab (hard dependency — see bugs) |

## 3. Repo structure (actual)

```
Spotify_PlayList_RecommendationSystem/
├── spotify_dataset.csv                        # 32,833 rows × 23 cols (7.7 MB)
├── spotify_playlist_recommendationsystem.py   # entire codebase (152 lines, Colab export)
├── README.md                                  # truncated mid-document (69 lines)
├── architecture.txt                           # describes structure that DOESN'T exist
├── requirements.txt                           # 30+ pinned packages, ~26 unused
├── gitignore                                  # ← misnamed, missing leading dot
└── LICENSE                                    # MIT
```

Flat, single-script repo. No `src/`, `tests/`, `models/`, or notebooks — even though `architecture.txt` and the README imply otherwise.

## 4. How the pipeline works

| Step | What happens |
|---|---|
| 1. Load & clean | Read CSV → drop NaN in `track_name/artist/album_name` → dedupe on (`track_id`,`playlist_id`) → **32,246 rows** |
| 2. EDA | `.describe()`, correlation heatmap of 11 numeric features (seaborn) |
| 3. Clustering v1 | `StandardScaler` → `PCA(2)` → **KMeans k=3 on the 2-D PCA output** → scatter (danceability vs energy) |
| 4. Elbow | Inertia for k=1…10 on full 10-D scaled features (visual only — result unused) |
| 5. Clustering v2 | **(duplicated block)** scale+PCA+KMeans again, then final **KMeans k=4 on full 10-D features** → overwrites `df['cluster']` |
| 6. Recommender | `NearestNeighbors(n_neighbors=6, ball_tree)` on 10-D scaled features; one **hardcoded** query track → print 5 nearest songs |

Dataset verified stats: 32,833 rows · 28,356 unique track IDs · 10,692 artists · genres: edm 6,043 / rap 5,746 / pop 5,507 / r&b 5,431 / latin 5,155 / rock 4,951.

---

## 5. 🔴 Critical bug (verified — the script crashes as-is)

The recommendation step throws:

```
IndexError: index 32828 is out of bounds for axis 0 with size 32246
```

**Root cause:** after `dropna()` + `drop_duplicates()`, the DataFrame keeps its **original index labels** (0…32832 with 587 gaps), but `X_scaled` is a positional numpy array of 32,246 rows. The code does:

```python
query_idx = query_df.index[0]           # label = 32828  ❌
distances, indices = nbrs.kneighbors(X_scaled[query_idx].reshape(1, -1))  # 💥 crash
```

**Latent second bug:** even when a label happens to fit, `indices` from `kneighbors()` are **positional**, while the follow-up `df.loc[idx]` uses **labels** → silently **wrong songs** would be displayed.

**One-line fix** (right after cleaning):

```python
df = df.reset_index(drop=True)
```

✅ With this fix the recommender runs and gives sensible results (verified):

```
Query: City Of Lights - Official Radio Edit by Lush & Simon
 → Call Me by In This Moment (0.78)
 → Helena by My Chemical Romance (0.86)
 → Living Again by Ryos (0.93)
 → Collide (feat. Collin McLoughlin) - Radio Edit by Laidback Luke (0.93)
 → I'll Be Here For You by Sick Individuals (0.94)
```

## 6. 🟠 Other defects

| # | Issue | Impact |
|---|---|---|
| 1 | `from google.colab import files; files.upload()` at the top | Script **won't run anywhere except Colab**; `google-colab` isn't even in requirements.txt |
| 2 | `display(plt.show())` | `display()` is IPython-only → `NameError` in plain Python |
| 3 | K-Means k=3 fitted on **2-D PCA output**, then overwritten by k=4 on full 10-D features | Two inconsistent clusterings; the scatter plot's clusters ≠ the final `cluster` column |
| 4 | Elbow method computed but result never used | k=4 is hardcoded; README overstates "optimal k selection via Elbow Method" |
| 5 | `release_year` computed | Never used anywhere |
| 6 | `key` and `mode` features | Present in dataset, ignored in clustering |
| 7 | README claims "32,246 **unique tracks**" | False — that's row count; only **28,356 unique track IDs** (same track in multiple playlists survives dedupe) |
| 8 | README cut off mid-document | Demo section is empty; Installation / Usage / Documentation sections linked in the nav **don't exist** |
| 9 | `architecture.txt` documents `src/`, `tests/`, `models/`, notebooks | **None of it exists** — aspirational, misleading |
| 10 | requirements.txt pins 30+ packages (sphinx, mypy, black, pylint, plotly, statsmodels, xlrd, jupyterlab…) | Actual needs: `pandas`, `scikit-learn`, `matplotlib`, `seaborn` only |
| 11 | File named `gitignore` (no leading dot) | Git **ignores nothing** — the file does nothing |
| 12 | Duplicated imports & pipeline blocks; hardcoded query track; no functions, no `main()`, no CLI args | Notebook-export artifact; poor reusability |

## 7. ✅ What's done well

- **Correct core idea**: scaling → distance-based KNN on standardized audio features is the right approach for content-based song similarity; recommendations (verified) are musically plausible.
- **Sensible cleaning**: dedupe on `(track_id, playlist_id)` correctly preserves a track's presence across playlists while removing exact row duplicates.
- **Reproducibility**: `random_state` set consistently on PCA/KMeans.
- **Good pedagogy**: pipeline covers EDA → clustering → elbow → recommender, a solid learning project structure.

## 8. 🚀 Recommendations (priority order)

1. **Add `df = df.reset_index(drop=True)`** — fixes the crash (critical).
2. Remove/guard Colab-only code (`files.upload()`, `display()`); read the CSV path directly.
3. Choose **one** clustering scheme: KMeans on the full 10-D scaled features (not the PCA-2D projection); keep PCA strictly for visualization.
4. Use the Elbow result (or **silhouette score**) to pick k programmatically instead of hardcoding 4.
5. Accept the query track via `argparse`/function param instead of hardcoding; handle multi-match by preferring highest `track_popularity`.
6. Slim `requirements.txt` to the 4 actually-used libraries.
7. Rename `gitignore` → `.gitignore`; fix README (unique-track count, finish Demo/Install/Usage sections); delete or actually implement `architecture.txt` layout.
8. Nice-to-haves: persist models with joblib, add silhouette/cluster evaluation, dedupe by `track_id` for recommendations, wrap pipeline in functions with a `main()`.

## 9. Verdict

**A decent beginner/learning ML project with a genuinely sound core algorithm — but it does not run as published.** The KNN stage crashes on any dataset large enough to leave index gaps after cleaning, the docs oversell the repo's structure (nonexistent `src/`, `tests/`, "32k unique tracks"), and the Colab coupling makes it non-portable. With the one-line index fix and ~1 hour of cleanup, it becomes a clean, reproducible content-based recommender.

| Aspect | Rating |
|---|---|
| Concept & ML approach | ★★★★☆ |
| Code quality | ★★☆☆☆ |
| Runs as-is | ✗ (crashes at recommendation step) |
| Documentation accuracy | ★★☆☆☆ |
| Overall (as a learning project) | ★★★☆☆ |
