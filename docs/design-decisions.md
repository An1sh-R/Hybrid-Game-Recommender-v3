# Hybrid Steam Recommender — Brainstorm Design Decisions

**Date:** 2026-10-04
**Status:** Brainstorm complete for product, data, modelling, pipeline, API, frontend and engineering practice. Data decisions settled from the EDA (2026-10-05). **Deployment is mostly left open** (the API host is decided: Render) and will be finished in a later session.

---

## 1. Goal and constraints

| | |
|---|---|
| **Purpose** | Résumé project. The main aim is to show ML and MLOps engineering skills, with a working live demo. |
| **Timeline** | Under 2 weeks. Getting it done smoothly matters more than clever extras. |
| **Budget** | **No recurring monthly cost.** AWS is used to show deployment engineering, not to host an always-on service. The public demo runs on a free host. |
| **Machine** | Windows 11, Python 3.11.9, Docker 29.6, RTX 4050 (6 GB), 16 GB RAM, about 166 GB free disk |

---

## 2. Product

- A visitor searches for and selects **1–5 games they liked** (about 3 typical) and gets **12 recommendations**.
- The demo has to show off **hybrid recommendation and cold-start handling**. The visitor is never in the training data.
- Each recommendation card shows:
  - "**Because you liked X**" (from item-to-item CF)
  - a small two-colour bar showing how much of the score came from **CF vs content**
- A **"hybrid dial" slider** (from "people who liked this" to "similar content") re-ranks the results live by changing α.
- Known dataset users (`user_id`) are used **only for offline evaluation**, not in the demo.
- **Out of scope for v1:** looking up a real Steam profile by SteamID.

---

## 3. Data

**Source:** Kaggle dataset [`antonkozyriev/game-recommendations-on-steam`](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)
- `games.csv`: app_id, title, release date, platforms, rating, positive ratio, review count, price, Steam Deck
- `users.csv`: user_id, number of products, number of reviews
- `recommendations.csv`: about 41M rows (app_id, user_id, is_recommended, hours, helpful, funny, date)
- `games_metadata.json`: **description and tags** for each app_id, which is the source for content-based filtering

| Decision | Choice | Why |
|---|---|---|
| Extra metadata from the Steam Storefront API or SteamSpy | **No.** Tags and descriptions are enough. | Tags already cover genre and mechanics. Avoids a crawler that would take about a day. Could be added later as "data v2". |
| Positive signal | `is_recommended = True`, with **confidence = 1 + c · log(1 + hours)**, `c` tuned over {0, 0.5, 1} | Standard implicit-feedback setup. Separates a 2-hour thumbs-up from a 400-hour one. `c = 0` treats every thumbs-up the same, which tests whether hours help. The EDA showed median hours of 30.3 for positive reviews vs 11.1 for negative. Hours are capped at 1,000 in the source data, which the log handles. |
| Duplicates | Drop the **21 duplicate `(user_id, app_id)` reviews** in the `filter` stage | Found in the EDA. Each user–game pair should appear once. |
| Raw data | **`data/raw/` is never edited by hand.** Cleaning happens in code. | Keeps the pipeline reproducible. Re-running `download` would overwrite manual edits anyway. |
| Negative reviews | Kept in the data, **ignored in the similarity for v1** | Subtracting them adds complexity for little expected gain. They can still be used as an experiment. |
| Data tooling | **pandas + numpy + scipy.sparse** (pyarrow only for Parquet). CSVs are read with `usecols` and small types (`int32`/`float32`/`bool`) and converted to **Parquet once**. **No Polars, DuckDB or SQLite.** | Simple, familiar tools. With explicit types the 41M-row table takes 535 MB in memory (measured in the EDA). Sparse matrices are required for similarity. A database adds nothing for batch processing. |
| Data size | **Filter to an active core** (min reviews per user, min reviews per game) | Fits in 16 GB of RAM, and active users carry most of the CF signal. Cold users are handled by the content side. |
| Filter thresholds | **`min_user_reviews = 5`, `min_game_reviews = 50`** (positive reviews; users filtered first, then games among those users). Keeps **17.7M interactions (50% of positive reviews), 1.62M users and 13,160 games**. Stored in `params.yaml`. | Held-out evaluation users need at least 5 reviews anyway (3 seeds plus at least 2 hidden). The median user has only 1 positive review, and a user with a single review adds nothing to item-to-item co-occurrence. |
| Heavy users | **Drop users with more than 168 positive reviews** (`max_user_reviews`, the top 0.1% of active users: 1,616 users, 2.7% of reviews) | One user with n liked games links n·(n−1)/2 game pairs. In the EDA, the top 0.1% produced 41% of all pairs, so a handful of users would decide the similarities. Dropping them is simpler than weighting them. |
| Hours outliers | Nothing extra. Only 26 reviews sit at the 1,000-hour cap. | The log in the confidence formula already squashes large values. |
| Planned experiment | Train on users with **3 or more** reviews (23.3M interactions, 3.3M users), with evaluation users still drawn from those with 5 or more. Compared using `dvc exp`. | Tests whether lighter users improve recall. |
| Adult content | **No filter. All games are kept.** | The user's decision. The EDA showed the obvious tags ("Sexual Content", "Nudity", "Mature") also cover mainstream games like The Witcher 2 and Far Cry 3. |
| Selectable seeds | The **~13k games that pass the CF filter** (at least 50 positive reviews from active users), popular games first in search | People pick games they recognise, and every seed has CF neighbours. One rule, no extra parameter. |
| Recommendation candidates | **Every game with tags** (~49.6k of 50,872) | The ~1.2k games without tags have no content signal. The ~36k games outside the CF set can only come through the content side, which is the item cold-start case the popularity-aware blend handles. |

---

## 4. Models

### 4.1 Collaborative filtering: item-to-item ("people who liked X also liked Y")
- **Cosine similarity between game columns** of the confidence-weighted user-game matrix.
- **Shrinkage** (e.g. `sim × n_common / (n_common + λ)`) down-weights similarities computed from only a few shared users.
- Keep the **top-K neighbours per game** (K ≈ 100–200) as a sparse matrix, which keeps the serving file small.
- A visitor's CF score for a candidate is the sum of its similarity to each seed game. The "Because you liked X" text comes from whichever seed contributed most.
- Rejected: ALS fold-in (less explainable) and LightFM (poorly maintained, painful to build in Docker).

### 4.2 Content-based
- **Tags:** plain **TF-IDF** over tags, where every tag a game has counts the same. Tags are the main content signal: 97.6% of games have them (441 distinct tags, up to 20 per game).
  - Planned experiment: weight tags by their position in the list (`1 / log2(position + 2)`) before applying IDF. Steam probably orders tags by player votes, but this isn't verified for this dataset. Kept only if Recall@10 improves.
- **Descriptions:** `sentence-transformers/all-MiniLM-L6-v2` embeddings (384 dimensions) of the text **"title: description"**, computed once in the pipeline on the local GPU. Including the title lets series and franchise names carry signal (*Dark Souls III* lands near *Dark Souls II*). Tags are kept out of this text so the two signals stay separate.
  - Whitespace in descriptions is normalised in code before embedding (including invisible U+2028 line separators found in 3 descriptions).
  - Only 79.6% of games have a description (median 35 words). **Games without one get their content score from tags only.**
- Content similarity is a weighted combination of tag cosine and embedding cosine. The weight is a parameter.
- **The serving container never loads the transformer.** Only the precomputed vectors or neighbour lists are shipped.

### 4.3 Hybrid blend: popularity-aware weighting
- `score = α(g)·CF + (1 − α(g))·content`, where α depends on the candidate game's review count. Games with few reviews lean on content, and well-known games lean on CF.
- Both scores are normalised before blending.
- α and the popularity curve are tuned by **grid search on a validation split**.
- The UI slider overrides α for each request. The default is the tuned value.
- Future work: a learned re-ranker (e.g. LightGBM on the CF score, content score, popularity, price and year).

### 4.4 Offline evaluation: simulated visitors
- Hold out a set of **users entirely** (split into validation and test users).
- For each held-out user, give the model **3 of their liked games as seeds**, hide the rest, and check whether the model recovers them.
- Metrics: **Recall@10, NDCG@10, catalog coverage**.
- Baselines: **popularity, CF-only, content-only, hybrid**, all on the same split.
- Tuning happens on validation users and the final results come from test users, so the reported numbers aren't optimistic.
- The final comparison table goes in the README.

---

## 5. Pipeline and data versioning

**DVC pipeline (`dvc.yaml`), with all settings in `params.yaml`:**

1. `download`: fetch the dataset with **`kagglehub`** (`kagglehub.dataset_download("antonkozyriev/game-recommendations-on-steam")`) and copy the files from kagglehub's cache into `data/raw/`, which DVC tracks
2. `validate`: **Pandera** schemas and sanity checks (columns, types, unique IDs, metadata coverage). Fails loudly if the data is wrong.
3. `filter`: drop duplicate reviews, keep positive reviews, apply the activity thresholds, drop heavy users
4. `split`: train, validation and test users
5. `content_features`: tag TF-IDF and description embeddings (GPU locally, CPU in CI)
6. `cf_similarity`: confidence-weighted, shrunk cosine similarity, top-K neighbours
7. `evaluate`: grid search on validation users, final metrics on test users, written to `metrics.json`
8. `export`: serving bundle (game catalog, CF neighbours, content features or neighbours, `manifest.json` with model version, data version and metrics)

**EDA** happens in Jupyter notebooks (`notebooks/`), for exploration only. Notebooks load data through the same `steamrec` loader functions, and anything they decide (e.g. filter thresholds) goes into `params.yaml`. Pipeline stages are always scripts, never notebooks. The findings behind the data decisions in section 3 are in `notebooks/01_eda.ipynb`.

| Decision | Choice |
|---|---|
| Experiment tracking | **DVC only:** `params.yaml`, `metrics.json`, `dvc exp show` and `dvc metrics diff`. **No MLflow** (kept simple). |
| DVC remote | **DagsHub** (free for public repos, permanent, no card needed). The remote must be permanent because CI pulls the serving bundle on every image build. |

---

## 6. Serving: API

**FastAPI**, in the same monorepo. It imports the same `steamrec` package as the pipeline, so the scoring code you evaluate is the code you serve (no training/serving skew).

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Status, model version and build commit (from `manifest.json`) |
| `GET /api/search?q=&limit=` | Fuzzy title search (`rapidfuzz`), popular games first, seed-eligible games only |
| `GET /api/popular` | Starter games for the landing page |
| `POST /api/recommend` | Body `{ "app_ids": [1..5], "k": 12, "alpha": optional }`. Returns games with `score`, `cf_score`, `content_score` and `because: [seed titles]`. |

**Guardrails:** 1–5 seeds, `k ≤ 50`, unknown app_ids give a clear `422` error, CORS limited to the frontend's domain and localhost, no rate limiter.

**Logging:** structured JSON request logs (latency, number of seeds, errors).

**Model delivery (provisional, revisit with deployment):** the serving bundle is baked into the API image during the CI build, so **the image tag identifies both the code and the model version**, and rollback means redeploying the previous tag.

---

## 7. Serving: frontend

- **Plain HTML, CSS and JS** (no framework).
- **Dark, Steam-adjacent theme** (deep navy and charcoal, one accent colour, artwork as the main colour). Colours are CSS variables, so a light theme could be added later.
- One page: header → search box with autocomplete → selected games shown as chips with small cover images → "Recommend" → responsive card grid.
- Each card: header artwork (from Steam's CDN by app_id, with a fallback image), title, top 3 tags, price, % positive reviews, "Because you liked X", the CF-vs-content bar, and a link to the Steam store page.
- Hybrid dial slider re-ranks the results.
- A "waking up the recommender…" state for when the free API host is cold-starting.
- A "not affiliated with Valve" note in the footer.

---

## 8. Engineering practice

| Area | Choice |
|---|---|
| Repo | **Monorepo:** `src/steamrec/` (core package), `pipeline/` (stage scripts), `api/`, `web/`, `infra/`, `deploy/`, `tests/`, `.github/workflows/` |
| Tooling | **uv** (lockfile), **ruff** (lint and format), **pytest**, **pre-commit**, `Makefile` |
| Docker | **Required.** The API runs in Docker. Locally, Docker Compose runs the API and a reverse-proxy container serving the frontend. **One image runs everywhere** (locally, on the demo host and on AWS). Base image `python:3.11-slim`. |
| CI (GitHub Actions) | ruff → unit and API tests → **`dvc repro` on a tiny fixture dataset** (smoke test) → build and push the image. **CI does not retrain** on real data. Real training happens locally and is shared with `dvc push`. |
| CD | Merging to `main` automatically deploys the free demo. Deploys to AWS are triggered manually. (Details depend on the deployment decision.) |
| Tests | Unit tests (similarity, shrinkage, blending, filtering, on hand-checkable synthetic data). API tests (FastAPI test client with fixture model files). Pipeline smoke test in CI. **Smoke test against the live URL after each deploy.** Stretch goal: one Playwright end-to-end test. |
| Git workflow | **GitHub flow:** feature branches, then pull requests into `main` with required CI checks (branch protection), squash merges, plain human-style commit messages with no type prefix (e.g. "Add item similarity stage") |
| Secrets | GitHub Actions secrets only. No long-lived AWS keys (use GitHub OIDC). |
| Monitoring | `/health` endpoint and JSON request logs. CloudWatch logs and alarms while AWS is up. Stretch goal: Prometheus and Grafana. |
| Visibility | Public GitHub repo (also needed for free tiers) |
| README | Architecture diagram, metrics comparison table, recording of the AWS deployment, demo link |

---

## 9. Deployment: OPEN, to be discussed later

Principles that are still agreed:
- **No recurring monthly cost.**
- **An always-on free demo**: static frontend plus the API container.
- **AWS is used to show engineering skill.** The preferred pattern is a **temporary environment**: one-click "up" (Terraform → deploy → smoke test → capture screenshots or video for the README) and one-click "down" (destroy). New AWS accounts come with signup credits, and a billing alarm acts as a backstop.
- **One Docker image** across all environments.

**Decided:** the free, always-on API runs on **Render** (free web service). The 512 MB / 0.1 CPU limits in section 9.2 are now **hard design constraints** on the serving bundle.

**Undecided, for a later session:** which AWS service (EC2 + Compose, ECS Fargate, Lambda container, …), Terraform scope, container registry (GHCR and/or ECR), HTTPS and reverse proxy, and where the frontend is hosted (Vercel or a Render static site).

### 9.1 Research: can Hugging Face Spaces host our API for free? **No.**

Checked against HF's official docs on 2026-10-04:

- **Docker Spaces now need a paid plan.** From the Spaces overview: *"Static Spaces are free for everyone. Gradio and Docker Spaces run on compute and require a paid plan to create: PRO for personal accounts, Team or Enterprise for organizations. Free personal accounts in good standing can still host up to 2 Gradio Spaces running on ZeroGPU."*
- The hardware itself would have been fine: CPU Basic is 2 vCPU, 16 GB RAM, 50 GB non-persistent disk, no hourly cost. **But a PRO subscription is needed to create a Docker Space**, which breaks the "no recurring cost" constraint.
- Other notes, relevant only if we paid for PRO: the default port is 7860 (`app_port`), the container runs as UID 1000, the disk is wiped on restart, outbound traffic is limited to ports 80, 443 and 8080, and free hardware sleeps when unused.
- The free options left on HF (Static Spaces, ZeroGPU Gradio Spaces) don't fit a FastAPI container.

**Conclusion:** HF Spaces is ruled out for the API. **Render is now the leading candidate** for the free API host.

### 9.2 Research: Render's free tier (**chosen** for the API, one item still to verify)

- Free web service: **512 MB RAM, 0.1 CPU** (from third-party summaries; Render's free-tier doc doesn't state the specs).
- **Spins down after 15 minutes without traffic.** Cold start takes **about 1 minute** (Render docs).
- **750 free instance hours per workspace per month**, enough for one service running all month.
- No persistent disk, single instance, no SSH, and *"Render might restart a Free web service at any time."*
- **To verify:** whether a free service can deploy a **prebuilt image from a registry** (sources disagree). The fallback is building from a one-line Dockerfile (`FROM ghcr.io/<user>/steamrec-api:<sha>`), which keeps the "one image everywhere" property.

**What 512 MB and 0.1 CPU would mean for the design:**
- The serving bundle has to stay small: store vectors as float16, keep only top-K neighbours, and avoid pandas at serving time if memory is tight.
- **Precompute content neighbours (top-K) as well as CF neighbours**, so serving is just cheap sparse lookups and the slider re-ranking costs almost nothing. With this, the serving memory estimate is roughly 150–300 MB.
- The frontend must handle the ~1-minute cold start well: a static frontend served from a CDN, a "waking up" state, and calling `/api/health` on page load.
- An image-size budget for the API (no torch or transformers in the serving image).

**Other free hosts worth comparing in the deployment session** (not yet researched; confirm current terms first): Google Cloud Run (scales to zero, has a free tier, needs a billing account), AWS Lambda with a container image plus a Function URL (would keep the demo on AWS), Koyeb, and Railway.

---

## 10. Out of scope for v1

Steam Storefront and SteamSpy crawling, SteamID profile lookup, learned re-ranker, MLflow, custom domain, Prometheus and Grafana, scheduled retraining, Kubernetes, feature store.

---

## 11. Manual steps only the user can do

- [ ] Kaggle credentials for `kagglehub`, only if the download asks for them (`kagglehub.login()`, or a token in `C:\Users\Anish Ray\.kaggle\kaggle.json`)
- [ ] GitHub account and a public repo for this project
- [ ] DagsHub account linked to the GitHub repo
- [ ] AWS account (root MFA, an admin user, billing alarm) when the deployment phase starts
- [ ] Accounts on the demo hosts once chosen (e.g. Render, Vercel)

---

## 12. Next steps

1. ~~Download with kagglehub, do EDA, settle the filter thresholds.~~ Done 2026-10-05 (section 3, `notebooks/01_eda.ipynb`).
2. Deployment session: choose the AWS service and the frontend host (the API host is decided: Render).
3. Write the formal design spec and implementation plan.

### Sources
- Hugging Face — Spaces Overview: https://huggingface.co/docs/hub/spaces-overview
- Hugging Face — Docker Spaces: https://huggingface.co/docs/hub/spaces-sdks-docker
- Render — Deploy for Free: https://render.com/docs/free
