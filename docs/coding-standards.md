# Coding Standards

**The rule behind all the others:** write *plain* code. Someone in their second year of university should be able to read any line and explain what it does and why. When a shorter version and a clearer version both exist, use the clearer one.

Design decisions (what we build) are in [`design-decisions.md`](design-decisions.md). This document covers how the code is written.

---

## 1. Functions first, one stateful class

Write **plain functions** that take data in and return data out. Use `@dataclass` to group related values (settings, results). Write a class only when something genuinely holds loaded state. In this project that's just `Recommender`, which keeps the loaded model files so each API request doesn't reload them.

```python
# Good: a function. Easy to test: pass inputs, check the output.
def shrink_similarity(similarity: float, n_common: int, shrinkage: float) -> float:
    """Down-weight similarities computed from only a few shared users."""
    # shrunk cosine: sim * n / (n + λ)
    return similarity * n_common / (n_common + shrinkage)


# Good: a dataclass for grouped values.
@dataclass
class EvalResult:
    recall_at_10: float
    ndcg_at_10: float
    coverage: float


# Avoid: a class that only wraps one function.
class SimilarityShrinker:
    def __init__(self, shrinkage):
        self.shrinkage = shrinkage

    def shrink(self, similarity, n_common):
        return similarity * n_common / (n_common + self.shrinkage)
```

---

## 2. Type hints on every function signature

Annotate every argument and return value. Local variables don't need hints. There's no type checker (no mypy); hints are there as documentation and to help your editor.

```python
def top_k_neighbours(similarity: csr_matrix, k: int) -> csr_matrix:
```

---

## 3. Docstrings and comments

- **Public functions** get a docstring: one summary line, plus `Args:` and `Returns:` only when the meaning isn't obvious from the names and types.
- **Comments say *why*, not *what*.** The code already says what.
- **Math gets a one-line formula comment**, written the way you'd write it on a whiteboard. That's the line you'll point to in an interview.

```python
# Good: explains why
# Log-scale hours so a 2000-hour player doesn't outweigh 50 normal players.
confidence = 1 + confidence_scale * np.log1p(hours)

# Avoid: repeats the code
# take log1p of hours and multiply by scale and add 1
confidence = 1 + confidence_scale * np.log1p(hours)
```

---

## 4. Formatting and linting

- **`ruff format`** handles all formatting automatically (Black style). Never format by hand.
- **`ruff check`** with a modest rule set: pyflakes errors (`F`), pycodestyle errors (`E`), import sorting (`I`), common bug patterns (`B`), modern syntax (`UP`).
- **Line length: 100.**
- Both run in **pre-commit** (before every commit) and in **CI** (on every pull request).

---

## 5. Naming and the glossary

Standard Python naming: `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_CASE` for module-level constants.

**Glossary.** Each concept has exactly one name, used everywhere: code, column names, API fields and docs.

| Term | Meaning |
|---|---|
| `app_id` | Steam's ID for a game. Never `game_id`, `appid` or `item`. |
| `user_id` | A dataset user's ID |
| `seed` | A game the visitor picked as one they liked |
| `candidate` | A game that could be recommended |
| `cf_score` | Score from item-to-item collaborative filtering |
| `content_score` | Score from content similarity (tags and description) |
| `alpha` | Blend weight: `score = alpha * cf_score + (1 - alpha) * content_score` |
| `confidence` | Interaction weight derived from hours played |
| `neighbours` | The top-K most similar games to a game |

**Matrix shapes** go in the variable name or a comment:

```python
user_game: csr_matrix  # (n_users, n_games), values = confidence
```

**One-letter names** are only for `i`/`j` loop counters and for variables that match a written formula exactly (e.g. `k` in Recall@k).

---

## 6. Configuration: no magic numbers

Every tunable number lives in **`params.yaml`**. It's loaded **once** into a typed `@dataclass`, and functions get values **as arguments**. Functions never open the config file themselves.

```python
# Good: the value comes in as an argument, so a test can pass any number.
def filter_users(reviews: pd.DataFrame, min_reviews: int) -> pd.DataFrame:
    ...

filtered = filter_users(reviews, min_reviews=params.filter.min_user_reviews)

# Avoid: a magic number hidden in the code.
def filter_users(reviews: pd.DataFrame) -> pd.DataFrame:
    counts = reviews.groupby("user_id").size()
    keep = counts[counts >= 5].index   # where did 5 come from?
```

Why it matters: tests can pass any value, a typo like `min_revews` fails as soon as the config loads, and DVC re-runs exactly the stages affected by a changed parameter.

---

## 7. Errors: validate at the edges, fail loudly

- **Data coming in** is checked by **Pandera** schemas (`validate.py`).
- **API requests** are checked by **Pydantic** models (FastAPI does this automatically).
- **Inside the core code**, trust data that has already been validated, and raise clear errors when something is wrong:

```python
if app_id not in catalog.index:
    raise ValueError(f"unknown app_id {app_id}")
```

Always catch a **specific** exception and either handle it or re-raise it with context. A silently swallowed error is the worst ML bug: the pipeline "works" but the numbers are wrong.

```python
# Good
try:
    params = load_params(path)
except FileNotFoundError as error:
    raise FileNotFoundError(f"params file missing: {path}") from error

# Never
try:
    params = load_params(path)
except:
    pass
```

---

## 8. Logging

Use Python's built-in **`logging`** module in `src/`, `pipeline/` and `api/`. One setup function lives in `logs.py`, and the API logs in JSON (latency, number of seeds, errors).

```python
logger = logging.getLogger(__name__)
logger.info("filtered users: %d -> %d", n_before, n_after)
```

`print` is only for notebooks.

---

## 9. Tests

- **pytest.** `tests/` mirrors `src/`, so `src/steamrec/cf.py` is tested by `tests/test_cf.py`.
- **Tiny hand-made data**, e.g. 4 users × 3 games, small enough that you can **work out the expected answer on paper**. If you can't, the test data is too big.
- **Every math function** gets a test: similarity, shrinkage, confidence, blending, metrics.
- **Test names describe behaviour:** `test_shrinkage_downweights_rare_pairs`, not `test_shrink_1`.
- Tests never touch the real dataset or the network.
- Writing the test first (TDD) is encouraged for the math functions.

```python
def test_shrinkage_downweights_rare_pairs():
    # Same raw similarity, but one pair shares 2 users and the other shares 200.
    rare = shrink_similarity(0.9, n_common=2, shrinkage=10)
    common = shrink_similarity(0.9, n_common=200, shrinkage=10)

    assert rare == pytest.approx(0.9 * 2 / 12)    # 0.15
    assert common == pytest.approx(0.9 * 200 / 210)
    assert rare < common
```

---

## 10. Size: small functions, focused modules

These are judgement calls, not lint rules:
- A **function does one thing** and fits on a screen (about 40 lines).
- A **module has one clear purpose** (roughly under 300 lines).
- **Return early** instead of nesting `if`s.

```python
# Good: return early
def validate_seeds(app_ids: list[int], max_seeds: int) -> None:
    if len(app_ids) == 0:
        raise ValueError("pick at least one game")
    if len(app_ids) > max_seeds:
        raise ValueError(f"pick at most {max_seeds} games")
```

---

## 11. Notebooks

- **Numbered names:** `notebooks/01_eda.ipynb`, `02_cf_sanity_check.ipynb`, …
- Before committing, **restart the kernel and run all cells top to bottom**. This catches cells that only worked because of an earlier out-of-order run.
- **Keep outputs** so plots display on GitHub.
- Notebooks load data through `steamrec` functions. Any function you'd reuse **moves into `src/steamrec/`**.
- Notebooks are for **exploring only**. Decisions go into `params.yaml`, and the real pipeline stages are scripts in `pipeline/`.

---

## 12. Dependencies and paths

- **Keep dependencies minimal.** Every new package needs a one-line reason in the pull request description. Smaller images matter on Render's 512 MB free tier.
- Dev-only tools (pytest, ruff, jupyter, pre-commit) go in the **uv dev group**.
- **The serving image never includes torch or transformers.** Description embeddings are computed in the pipeline, and the API only loads the precomputed results.
- Use **`pathlib`** everywhere. Paths are **relative to the repo root** and come from config, never `C:\Users\...`.

---

## 13. Module layout (`src/steamrec/`)

| Module | Job |
|---|---|
| `config.py` | Typed settings dataclasses, loaded from `params.yaml` |
| `data.py` | Load raw CSV/JSON → Parquet, filter, split users |
| `validate.py` | Pandera schemas for raw and filtered data |
| `content.py` | Tag TF-IDF, description embeddings, content neighbours |
| `cf.py` | User-game matrix, confidence weights, shrunk cosine, top-K neighbours |
| `hybrid.py` | Score normalisation, alpha per game, blending, "because" text |
| `evaluate.py` | Simulated visitors, Recall@10 / NDCG@10 / coverage, baselines, alpha search |
| `artifacts.py` | Save/load the serving bundle and `manifest.json` |
| `recommender.py` | `Recommender` class: holds loaded artifacts, offers `search()` and `recommend()` |
| `logs.py` | Logging setup |

**Import rules:**
- `cf.py` and `content.py` are independent and never import each other. `hybrid.py` combines them.
- The API imports **only** `recommender.py`.

---

## 14. Plain code: what's banned and what's allowed

**Write instead:** a plain `for` loop, a named intermediate variable, a regular `if/else`, or a small named function.

**Banned:**
- Nested comprehensions: `[x for row in matrix for x in row if x > 0]`
- The walrus operator `:=`
- Nested ternaries: `a if x else b if y else c`
- `functools.reduce`
- `lambda`, except as a trivial sort key (`key=lambda game: game.score`)
- Boolean tricks used for values: `name = given or default and fallback`
- Custom decorators, metaclasses, monkey-patching
- `*args` / `**kwargs` passthrough
- One-letter names (except `i`/`j` and formula variables, see section 5)

**Allowed:**
- Simple comprehensions with **one** `for` and at most **one** `if`: `[game.app_id for game in games if game.is_seedable]`
- f-strings
- Decorators that frameworks need: `@dataclass`, `@app.get`, `@pytest.fixture`
- **Vectorised numpy/scipy operations.** With 41M rows, vectorising is necessary rather than clever. **Every vectorised line gets a shape comment:**

```python
# (n_games, n_users) @ (n_users, n_games) -> (n_games, n_games): co-occurrence weights
co_occurrence = game_user @ game_user.T

# (n_games,): number of users who interacted with each game
game_counts = np.asarray((game_user > 0).sum(axis=1)).ravel()
```

```python
# Banned: clever one-liner
because = {c: max(seeds, key=lambda s: sim[s, c]) for c in cands if any(sim[s, c] for s in seeds)}

# Plain version
because = {}
for candidate in candidates:
    best_seed = None
    best_similarity = 0.0
    for seed in seeds:
        similarity = neighbours[seed, candidate]
        if similarity > best_similarity:
            best_seed = seed
            best_similarity = similarity
    if best_seed is not None:
        because[candidate] = best_seed
```

---

## 15. pandas style

- **Step by step, with a named variable after each meaningful step.** Short chains of up to about **3 calls** are fine.
- **No row-wise `.apply(lambda ...)`.** Use vectorised column operations, or an explicit loop if it truly has to go row by row. Row-wise apply is both a clever lambda and very slow on 41M rows.
- **Always pass `usecols` and `dtype`** when reading files.
- Column names use glossary terms.

```python
# Good
reviews = pd.read_csv(
    path,
    usecols=["app_id", "user_id", "is_recommended", "hours"],
    dtype={"app_id": "int32", "user_id": "int32", "is_recommended": "bool", "hours": "float32"},
)
positive = reviews[reviews["is_recommended"]]
reviews_per_user = positive.groupby("user_id").size()
active_users = reviews_per_user[reviews_per_user >= min_user_reviews].index
active_reviews = positive[positive["user_id"].isin(active_users)]

# Avoid: one long chain with lambdas
active = (pd.read_csv(path).query("is_recommended").pipe(lambda d: d[d.groupby("user_id")
          ["app_id"].transform("size") >= 5]).assign(c=lambda d: d.hours.apply(lambda h: 1 + log1p(h))))
```

---

## 16. Pull requests and git

- **GitHub flow:** a branch per change, then a pull request into `main`. CI must pass (branch protection), merges are squash merges.
- **Branch names:** `feat/cf-similarity`, `fix/search-encoding`, `docs/readme-metrics`, `chore/ruff-config`.
- **Commit messages:** a plain sentence written the way a person would describe the change. Start with a capital letter and an action word, no type prefix (`feat:`, `chore:`, `fix:` …), no full stop, about 60 characters at most. If the reason isn't obvious, add a blank line and a short body explaining *why*.
  - Good: `Add shrunk cosine similarity for item neighbours`
  - Good: `Fix search returning games without tags`
  - Avoid: `feat: add shrunk cosine similarity`, `update stuff`, `wip`
- **Keep pull requests small:** roughly under 400 lines changed, excluding lockfiles.
- **Self-review checklist** (in the pull request template):
  - [ ] Tests pass locally (`make test`) and lint is clean (`make lint`)
  - [ ] New tunable numbers are in `params.yaml`
  - [ ] Docs updated if behaviour changed
  - [ ] No banned constructs (section 14)
  - [ ] New dependencies have a stated reason
  - [ ] **I can explain every line of this pull request**

The last item matters most. If AI-written code contains a line you can't explain, simplify it or get it explained **before** merging.

---

## 17. Frontend (`web/`)

- **Plain modern JavaScript, no build step, no framework.** ES modules:
  - `api.js`: every `fetch` call to the API
  - `ui.js`: functions that build and update DOM elements
  - `app.js`: wiring (event listeners, page state)
- `const` by default, `let` when a value changes. `async/await` for fetch calls.
- **Data from the API goes in with `textContent`**, never `innerHTML`. This prevents HTML injection.
- **CSS colours, spacing and radii are variables** in `:root`, so the theme can change in one place.
- Simple, descriptive class names (`.game-card`, `.seed-chip`, `.score-bar`).
- No JS linter or formatter.
