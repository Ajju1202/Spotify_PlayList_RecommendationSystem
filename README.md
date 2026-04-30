# 🎵 Spotify Playlist Recommendation System

<div align="center">

![Spotify](https://img.shields.io/badge/Spotify-Music%20Analysis-1DB954?style=for-the-badge&logo=spotify&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An intelligent music recommendation system using K-Means clustering and K-Nearest Neighbors**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-documentation)

</div>

---


## 🎯 Overview

This project implements a **content-based music recommendation system** that analyzes Spotify track features to cluster similar songs and provide personalized recommendations. Using machine learning techniques including K-Means clustering and K-Nearest Neighbors (KNN), the system identifies music patterns based on audio characteristics like danceability, energy, tempo, and valence.

### 🎬 What This Project Does:

1. **Analyzes** 32,246 unique Spotify tracks across multiple genres
2. **Clusters** songs based on audio features using K-Means algorithm
3. **Recommends** similar tracks using KNN collaborative filtering
4. **Visualizes** music patterns through correlation matrices and cluster plots

---

## ✨ Features

### 🔍 Data Analysis
- ✅ Comprehensive exploratory data analysis (EDA)
- ✅ Statistical summaries of 11+ audio features
- ✅ Correlation analysis between musical attributes
- ✅ Genre distribution visualization

### 🤖 Machine Learning
- ✅ **K-Means Clustering** with optimal k selection via Elbow Method
- ✅ **PCA Dimensionality Reduction** for 2D visualization
- ✅ **KNN Recommendation Engine** with configurable neighbors
- ✅ Feature scaling with StandardScaler

### 📊 Visualizations
- ✅ Heatmap of audio feature correlations
- ✅ 2D cluster scatter plots (Danceability vs Energy)
- ✅ Elbow curve for optimal cluster determination
- ✅ Genre distribution bar charts

### 🎼 Audio Features Analyzed
| Feature | Description | Range |
|---------|-------------|-------|
| **Danceability** | How suitable for dancing | 0.0 - 1.0 |
| **Energy** | Intensity and activity measure | 0.0 - 1.0 |
| **Loudness** | Overall loudness in dB | -60 to 0 |
| **Speechiness** | Presence of spoken words | 0.0 - 1.0 |
| **Acousticness** | Acoustic vs electronic | 0.0 - 1.0 |
| **Instrumentalness** | Likelihood of no vocals | 0.0 - 1.0 |
| **Liveness** | Presence of audience | 0.0 - 1.0 |
| **Valence** | Musical positiveness | 0.0 - 1.0 |
| **Tempo** | Beats per minute | Variable |

---

## 🎬 Demo

### Example Output: Song Recommendations
