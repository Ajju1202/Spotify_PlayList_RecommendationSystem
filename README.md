# 🎵 Spotify Playlist Recommendation System

<div align="center">

![Spotify](https://img.shields.io/badge/Spotify-Music%20Analysis-1DB954?style=for-the-badge&logo=spotify&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A beginner-friendly, content-based music recommender using K-Means clustering and K-Nearest Neighbors**

</div>

---

## 🎯 What is this project?

This project takes **32,833 Spotify tracks** and learns which songs sound alike,
using only their audio features — danceability, energy, tempo, valence, and more.

It does three things:

1. **Clusters** songs into groups using **K-Means** (picks k with the Elbow Method, visualizes with PCA)
2. **Recommends** similar songs using **K-Nearest Neighbors (KNN)** in audio-feature space
3. **Visualizes** correlations, clusters, and the elbow curve (saved as PNG files)

> 💡 This is **content-based filtering**: "you like song X, here are songs that *sound* like X."
> (No user data is involved, so it's not collaborative filtering.)

## 📚 What you'll learn

- Cleaning a real dataset with **pandas** (missing values, duplicates, index gotchas)
- Why **feature scaling** matters before any distance-based ML
- How **K-Means** clustering works and how to choose k with the **Elbow Method**
- Why **PCA** is used for *visualization only*, not for the actual clustering
- Building a song recommender with **K-Nearest Neighbors** in 10-D feature space

## 🚀 Quick start (3 commands)

```bash
git clone https://github.com/Ajju1202/Spotify_PlayList_RecommendationSystem.git
cd Spotify_PlayList_RecommendationSystem
pip install -r requirements.txt
```

Then run it:

```bash
python spotify_playlist_recommendationsystem.py
```

That's it! The script uses a built-in demo song. Expected output:

```
[1/5] Loading dataset: spotify_dataset.csv
      ✅ Kept 32,246 of 32,833 rows (587 empty/duplicate rows removed)
...
[5/5] Recommending songs similar to: 'City Of Lights - Official Radio Edit'
      🎧 Query: City Of Lights - Official Radio Edit — Lush & Simon [edm]

      🎵 Top 5 similar songs:
       1. Call Me — In This Moment [rock] (similarity distance: 0.78)
       2. Helena — My Chemical Romance [rock] (similarity distance: 0.86)
       3. Living Again — Ryos [edm] (similarity distance: 0.93)
       ...
✅ Done! Check outputs/figures/ for charts and outputs/cluster_summary.csv
```

### Try it with your own song

```bash
python spotify_playlist_recommendationsystem.py --track "Shape of You"
python spotify_playlist_recommendationsystem.py --track "Believer" --neighbors 8
python spotify_playlist_recommendationsystem.py --k 6        # change # of clusters
python spotify_playlist_recommendationsystem.py --auto-k     # let silhouette pick k
```

### 🌐 Or use it as a website!

```bash
python web_app.py
```

Then open **http://127.0.0.1:5000** — type any song into the search box and get
5 similar tracks with match scores, right in your browser.

## 📁 Project structure

```
├── spotify_dataset.csv                       # the data (32,833 tracks × 23 columns)
├── spotify_playlist_recommendationsystem.py  # the ML pipeline — one readable script
├── web_app.py                                # 🌐 Flask website for the recommender
├── requirements.txt                          # the 6 libraries needed
├── .gitignore
├── LICENSE
└── outputs/                                  # created when you run the script
    ├── figures/correlation_heatmap.png
    ├── figures/elbow_method.png
    ├── figures/silhouette_scores.png
    ├── figures/song_clusters.png
    └── cluster_summary.csv
```

## 🧠 How it works (the 5 steps)

| Step | What happens | Why |
|------|-------------|-----|
| 1. Load & clean | Remove rows with missing names, dedupe `(track, playlist)` pairs, **reset the index** | Index gaps would crash the KNN step — this was the original repo's bug! |
| 2. EDA | Stats + correlation heatmap | Get a feel for the data before modeling |
| 3. Elbow Method | K-Means for k = 1…10, plot inertia | Find a sensible number of clusters |
| 3b. Silhouette (`--auto-k`) | Score k = 2…10, keep the highest | An objective, automatic way to choose k |
| 4. K-Means | Cluster on all **10 scaled audio features** | PCA is used **only** to draw the 2-D picture |
| 5. KNN recommender | Find nearest songs in feature space | Similar audio features ≈ similar-sounding songs |

## 🔬 Audio features used

| Feature | Meaning | Range |
|---------|---------|-------|
| Danceability | How suitable for dancing | 0.0 – 1.0 |
| Energy | Intensity and activity | 0.0 – 1.0 |
| Loudness | Overall loudness (dB) | −60 – 0 |
| Speechiness | Presence of spoken words | 0.0 – 1.0 |
| Acousticness | Acoustic vs electronic | 0.0 – 1.0 |
| Instrumentalness | Likelihood of no vocals | 0.0 – 1.0 |
| Liveness | Audience presence | 0.0 – 1.0 |
| Valence | Musical positiveness | 0.0 – 1.0 |
| Tempo | Beats per minute | ~50 – 250 |
| Duration_ms | Track length | milliseconds |

## 🧪 Experiments to try (great for learning!)

1. Change `DEFAULT_K` (or `--k`) from 4 to 6 — the dataset has 6 genres; do clusters match genres?
2. Remove `duration_ms` from `AUDIO_FEATURES` — do the recommendations improve?
3. Add `key` and `mode` — the dataset has them, the original code ignored them
4. Try 10 different query songs — do you always agree with the recommender?
5. Set `SHOW_PLOTS = True` at the top of the script to get pop-up charts locally

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: spotify_dataset.csv` | Run the script from the repo folder (or keep the CSV next to the `.py` file) |
| `ModuleNotFoundError: sklearn` | Run `pip install -r requirements.txt` |
| Charts don't pop up | They're saved to `outputs/figures/` — or set `SHOW_PLOTS = True` |
| First run feels slow | The Elbow Method trains K-Means 10 times — that's normal (~30 s) |
| Web app won't start: `Address already in use` | Something else uses port 5000 — change `port=5000` at the bottom of `web_app.py` |

## 📜 License

MIT — free to use, learn from, and build upon.
