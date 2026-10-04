# Hybrid Steam Recommender

Hybrid game recommender (item-to-item CF + content) with an MLOps pipeline. The user is an undergrad building this for their résumé: every line must be one they can explain.

## Read before acting
- **What we build and why:** `docs/design-decisions.md`. Read it before designing, adding a feature, or changing the pipeline/API/deployment.
- **How code is written:** `docs/coding-standards.md`. Read it before writing or reviewing code; it holds the examples behind every rule below.

## Write *plain* code
- Each step is its own statement with a named variable. A plain `for` loop or `if/else` beats a compact expression.
- Plain functions + `@dataclass`. The only stateful class is `Recommender`.
- Type hints on every signature. Short docstring on public functions. Comments say *why*; math gets a one-line formula comment.
- Guardrail (banned): nested comprehensions, `:=`, nested ternaries, `reduce`, lambdas other than trivial sort keys, custom decorators/metaclasses, `*args/**kwargs` passthrough, one-letter names (except `i`/`j` and formula variables).
- Vectorised numpy/scipy is required for the 41M-row data: put a shape comment on every vectorised line.
- pandas: named intermediate variables, chains of ≤3 calls, vectorised column ops instead of row-wise `.apply`, explicit `usecols`/`dtype` on every read.

## Structure and boundaries
- Tunables live in `params.yaml`, loaded once into a typed dataclass and passed in as arguments.
- Validate at the edges (Pandera for data, Pydantic for the API), then trust the data inside; raise specific errors with context.
- `logging` in `src/`, `pipeline/`, `api/`; `print` only in notebooks. `pathlib` with repo-relative paths.
- `cf.py` and `content.py` stay independent; `hybrid.py` combines them; the API imports only `recommender.py`.
- Serving image targets Render free tier (512 MB RAM): the API loads precomputed artifacts only, with torch/transformers kept out of `api/`.
- New dependency: state the reason to the user before adding it.
- Every math function gets a pytest test on tiny data whose answer can be checked by hand.

## Glossary (one name per concept, everywhere)
`app_id`, `user_id`, `seed`, `candidate`, `cf_score`, `content_score`, `alpha`, `confidence`, `neighbours`. Definitions are in `docs/coding-standards.md` §5.

## Working with the user
- Small PRs (~≤400 changed lines), `feat/…` / `fix/…` branches.
- Commit messages are plain human sentences with no type prefix: `Add shrunk cosine similarity for item neighbours` (rules in `docs/coding-standards.md` §16).
- After writing non-obvious code, explain it briefly so the user can pass the "I can explain every line" PR check.
