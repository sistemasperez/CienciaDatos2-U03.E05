from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

from inf8239_u03.config import MOVIELENS_DIR, RANDOM_STATE, ROOT
from inf8239_u03.data import load_movielens
from inf8239_u03.metrics import (
    catalog_coverage,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from inf8239_u03.recommenders import MatrixFactorization, temporal_leave_one_out


def run_experiments():
    print("Cargando dataset...")
    ratings, movies = load_movielens(MOVIELENS_DIR)
    train, test = temporal_leave_one_out(ratings)

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    print("Entrenando Factorización Matricial (20 factores, 12 épocas)...")
    t0 = perf_counter()
    mf_model = MatrixFactorization(factors=20, seed=RANDOM_STATE).fit(train, epochs=12)
    mf_train_time = perf_counter() - t0

    evaluable = test[test["userId"].isin(mf_model.user_index) & test["movieId"].isin(mf_model.item_index)]
    predictions = [mf_model.predict(row.userId, row.movieId) for row in evaluable.itertuples()]
    rmse = root_mean_squared_error(evaluable["rating"], predictions)
    print(f"RMSE de predicción de rating (evaluable n={len(evaluable)}): {rmse:.4f}")

    # Popularity Baseline
    popular = train.groupby("movieId")["rating"].agg(["mean", "count"]).sort_values(["count", "mean"], ascending=False)
    popular_ranked = popular.index.tolist()

    pop_recs = {}
    for user_id in evaluable["userId"].unique():
        seen = set(train.loc[train["userId"].eq(user_id), "movieId"])
        cands = [m for m in popular_ranked if m not in seen][:10]
        pop_recs[int(user_id)] = cands

    # Precomputar perfiles y matrices de candidatos para recomendación híbrida
    print("Precomputando perfiles TF-IDF...")
    movie_table = movies.reset_index(drop=True).copy()
    movie_table["genres_text"] = movie_table["genres"].str.replace("|", " ", regex=False)
    genre_matrix = TfidfVectorizer().fit_transform(movie_table["genres_text"])
    movie_row = {int(movie_id): index for index, movie_id in enumerate(movie_table["movieId"])}

    user_candidates = {}
    user_collab_norm = {}
    user_content_norm = {}

    for user_id in evaluable["userId"].unique():
        seen = set(train.loc[train["userId"].eq(user_id), "movieId"])
        collab_df = mf_model.top_n(user_id, seen, 100)
        c_min, c_max = collab_df["collaborative_score"].min(), collab_df["collaborative_score"].max()
        norm_collab = (collab_df["collaborative_score"] - c_min) / max(c_max - c_min, 1e-9)

        history = train.loc[train["userId"].eq(user_id) & train["movieId"].isin(movie_row), ["movieId", "rating"]]
        history_rows = [movie_row[int(item)] for item in history["movieId"]]
        weights = np.clip(history["rating"].to_numpy() - 2.5, 0.1, None)
        profile = np.asarray(genre_matrix[history_rows].multiply(weights[:, None]).sum(axis=0) / weights.sum())
        cand_rows = [movie_row[int(item)] for item in collab_df["movieId"]]
        c_scores = cosine_similarity(profile, genre_matrix[cand_rows]).ravel()
        cnt_min, cnt_max = c_scores.min(), c_scores.max()
        norm_content = (c_scores - cnt_min) / max(cnt_max - cnt_min, 1e-9)

        user_candidates[user_id] = collab_df["movieId"].to_numpy()
        user_collab_norm[user_id] = norm_collab.to_numpy()
        user_content_norm[user_id] = norm_content

    alphas = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    results = []

    # Registrar Popularidad
    results.append({
        "Modelo": "Popularidad Ponderada",
        "Alpha": np.nan,
        "RMSE": np.nan,
        "HitRate@10": hit_rate_at_k(pop_recs, evaluable, 10),
        "Precision@10": precision_at_k(pop_recs, evaluable, 10),
        "Recall@10": recall_at_k(pop_recs, evaluable, 10),
        "NDCG@10": ndcg_at_k(pop_recs, evaluable, 10),
        "CatalogCoverage": catalog_coverage(pop_recs, len(mf_model.items)),
        "Descripcion": "Baseline no personalizado basado en conteo y rating promedio de entrenamiento",
    })

    for alpha in alphas:
        rec_alpha = {}
        for user_id in evaluable["userId"].unique():
            cands = user_candidates[user_id]
            scores = alpha * user_collab_norm[user_id] + (1.0 - alpha) * user_content_norm[user_id]
            top10_idx = np.argsort(scores)[::-1][:10]
            rec_alpha[int(user_id)] = [int(cands[i]) for i in top10_idx]

        name = "Contenido Puro" if alpha == 0.0 else ("Colaborativo Puro (MF)" if alpha == 1.0 else f"Hibrido (alpha={alpha})")
        results.append({
            "Modelo": name,
            "Alpha": alpha,
            "RMSE": rmse if alpha > 0 else np.nan,
            "HitRate@10": hit_rate_at_k(rec_alpha, evaluable, 10),
            "Precision@10": precision_at_k(rec_alpha, evaluable, 10),
            "Recall@10": recall_at_k(rec_alpha, evaluable, 10),
            "NDCG@10": ndcg_at_k(rec_alpha, evaluable, 10),
            "CatalogCoverage": catalog_coverage(rec_alpha, len(mf_model.items)),
            "Descripcion": f"Ponderacion lineal: {alpha:.2f} CF + {1.0-alpha:.2f} Content",
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv(reports_dir / "pareto_experiments.csv", index=False)
    
    # Escribir tabla en Markdown
    md_lines = [
        "| Modelo | Alpha | NDCG@10 | HitRate@10 | Precision@10 | Recall@10 | Cobertura Catalogo | RMSE |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for _, r in df_results.iterrows():
        a_str = f"{r['Alpha']:.2f}" if pd.notna(r['Alpha']) else "-"
        rmse_str = f"{r['RMSE']:.4f}" if pd.notna(r['RMSE']) else "-"
        md_lines.append(
            f"| {r['Modelo']} | {a_str} | {r['NDCG@10']:.4f} | {r['HitRate@10']:.4f} | {r['Precision@10']:.4f} | {r['Recall@10']:.4f} | {r['CatalogCoverage']*100:.2f}% | {rmse_str} |"
        )
    (reports_dir / "main_results_table.md").write_text("\n".join(md_lines), encoding="utf-8")
    
    print("\nResultados comparativos:")
    for _, r in df_results.iterrows():
        print(f"{r['Modelo']} | NDCG@10={r['NDCG@10']:.4f} | HitRate@10={r['HitRate@10']:.4f} | Cobertura={r['CatalogCoverage']*100:.2f}%")

    # Grafico 1: Trade-off de Metricas segun Alpha
    hybrid_only = df_results[df_results["Alpha"].notna()].sort_values("Alpha")
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    p1 = ax1.plot(hybrid_only["Alpha"], hybrid_only["NDCG@10"], "b-o", linewidth=2, label="NDCG@10 (Ranking)")
    p2 = ax1.plot(hybrid_only["Alpha"], hybrid_only["HitRate@10"], "g--s", linewidth=1.5, label="HitRate@10 (Acierto)")
    p3 = ax2.plot(hybrid_only["Alpha"], hybrid_only["CatalogCoverage"] * 100, "r-^", linewidth=2, label="Cobertura Catalogo (%)")

    ax1.set_xlabel("Parametro de Hibridacion alpha (0=Contenido, 1=Colaborativo)", fontsize=11)
    ax1.set_ylabel("Metricas de Ranking (NDCG / HitRate)", color="b", fontsize=11)
    ax2.set_ylabel("Cobertura del Catalogo (%)", color="r", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.title("Trade-off de Rendimiento vs Cobertura segun Parametro alpha", fontsize=13, pad=12)

    lines = p1 + p2 + p3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    plt.savefig(reports_dir / "alpha_tradeoff.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Grafico 2: Frontera de Pareto (Cobertura vs NDCG@10)
    plt.figure(figsize=(8, 6))
    for _, row in df_results.iterrows():
        x = row["CatalogCoverage"] * 100
        y = row["NDCG@10"]
        label = row["Modelo"]
        if row["Modelo"] == "Popularidad Ponderada":
            plt.scatter(x, y, color="black", s=130, marker="X", label="Popularidad")
            plt.annotate(f"{label}", (x, y), textcoords="offset points", xytext=(8, -5), fontsize=9, fontweight="bold")
        elif row["Alpha"] in [0.0, 0.5, 0.75, 1.0]:
            color = "blue" if row["Alpha"] == 0.75 else "purple"
            plt.scatter(x, y, color=color, s=110, marker="o")
            plt.annotate(f"{label}", (x, y), textcoords="offset points", xytext=(8, 3), fontsize=9)
        else:
            plt.scatter(x, y, color="gray", s=60, alpha=0.6)

    opt = df_results[df_results["Alpha"] == 0.75].iloc[0]
    plt.scatter(opt["CatalogCoverage"] * 100, opt["NDCG@10"], facecolors="none", edgecolors="red", s=300, linewidth=2.5, label="Decision Pareto (alpha=0.75)")

    plt.xlabel("Cobertura del Catalogo (%) [Diversidad / Espacio Exploratorio]", fontsize=11)
    plt.ylabel("NDCG@10 [Calidad del Ranking]", fontsize=11)
    plt.title("Frontera de Pareto: Calidad del Ranking vs Cobertura del Catalogo", fontsize=13, pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(reports_dir / "pareto_frontier.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nGraficos generados exitosamente en {reports_dir}")


if __name__ == "__main__":
    run_experiments()
