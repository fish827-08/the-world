# the-world

**A deterministic spherical digital-life simulation platform — Python orchestration, a Rust/PyO3 numerical core, and a versioned scientific record.**

[中文说明](README.md) | English

Status: **archived research project (closed 2026-10-24), maintenance mode**.
License: **Apache-2.0**. No claim of emergent language, intelligence, or "L0–L4" success is made.

---

## What this is

`the-world` simulates a minimal digital-life population on a latitude–longitude sphere
(**480 × 960 = 460,800 cells** in the formal configuration). The world provides terrain, a
seasonal energy-balance climate, resource/signal/odour fields, and sub-cell movement. Organisms
have small real-valued genomes and fixed behavioural formulas; **no fitness function and no LLM
are used in the simulation loop**. The research question was whether signal use, memory, and other
behavioural structure could arise and stabilise under selection alone.

The project was closed on 2026-10-24. It is published in full — including its failures — because
the failures are part of the result.

## What was established (and what was not)

| Question | Result |
|---|---|
| End-to-end emergence of language-like signalling (L0–L4) | **Not established.** The atmospheric criteria gates J1 and J2 are **FAIL**; the J3 end-to-end coupling test was **NOT-RUN** because its P0 prerequisites failed, producing **zero coupling data**. |
| Do memory and odour mechanisms change foraging? | **Yes, at the mechanism layer.** Four positive mechanism-layer findings are listed below; they are **not** evidence of emergence. |
| Is the world simulated deterministically? | **Yes.** The Python and Rust paths are bit-exact, and the frozen digests are reproducible. |

Mechanism-layer positives (scope and caveats in [docs/公开/02-科研账本-20261024.md](docs/公开/02-科研账本-20261024.md)):

- **Memory improves patch coverage** — S3 memory batch (run on the s2 device): 24/24 seeds reduced
  unvisited patch cells (median on−off −2.92 pp).
- **Memory raises carrying capacity** — flagship 60k, 4/4 paired seeds positive; the first three
  rose +23.1% to +176.1%, while seed 172 went 82 → 18,634 (denominator near extinction, so the
  ratio is distorted; read the absolute value). A wider flagship set of 8 seeds (5 on branch A,
  3 on branch B) read K ratios of 1.231–227.244, with seed 172 again affected by that denominator.
  The B-side main gate did **not** pass (recorded separately).
- **Odour makes a zero-food desert traversable** — batch A main gate 23/24. The separate
  four-channel device E-048A reads 24/24 on a different criterion and a different device; the two
  are **not interchangeable**.
- **Memory enlarges individual range** — 4/4 seeds; the share of moves longer than 5 cells rose
  4.7–7.5×.

Negative results are recorded with equal weight: g22 memory-weight perturbation was neutral; the
g22 evolutionary-era test was underpowered; LTO-1 never reached its checkpoint; the flagship 60k
B gate failed; the C-batch natural-climate arm failed; K convergence held in only 3/5 seeds; the
chain and adhesion probes were mostly negative; multicellularity (W4b) and the deterrent/arms-race
line (C3/B16) were negative or underpowered; J1 FAIL; J2 FAIL.

Structural findings (the most reusable output):

1. **Signal marginal value was zero by construction** — the signal encoded the emitter's state
   with a formula the receiver could already read, so emitting was pure cost and selection
   switched it off.
2. **An expressivity ceiling** — finite-dimensional real-valued genes plus fixed behavioural
   formulas span only a finite-dimensional behavioural manifold, which is structurally mismatched
   with the combinatorial unboundedness required for language.
3. **The K=2 "bistability" was an apparatus bug** — it came from a background-multiplier lottery,
   not from the mechanism under test.

## Architecture

```text
world/            spherical grid, insolation/temperature, resource/signal/odour fields
simulation/       engine and configuration (single state machine, deterministic step)
biosphere/        minimal organism layer (genome → development → controller)
sim_core/         Rust/PyO3 hot paths, bit-exact with the Python reference
atmo/             atmospheric coupling interface (off by default)
atmo_bridge/      atmospheric remapping/bridge helpers
atmo_core/        Rust dry Held–Suarez core (frozen after P1 FAIL)
atmo_julia/       optional SpeedyWeather.jl integration (external EUPL-1.2 dependency)
s0_kernel/        zero-dependency channel-driven reference world for conservation tests
observatory/      experiment matrix, statistics, WebSocket snapshots
viz/              snapshot exporter + Vite/React/three.js replay front end
benchmarks/       atmospheric criteria, metrics, gate documents
experiments/      one-off experiment runners (historical; some need adaptation)
tools/            reproducible probes, digest and audit utilities
tests/            three-tier pytest suite
```

## Quick start

Requires Python ≥ 3.10 (tested on 3.12 and 3.14), a Rust toolchain, and `maturin`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Build the Rust extension used by the mainline tests
cd sim_core && maturin develop --release && cd ..

# Tier-2 mainline tests (this command reads 1,580 passed / 0 failed / 10 skipped / 48 deselected)
python -m pytest tests/ -q -m "not legacy and not slow" --ignore=tests/test_broker.py

# Small deterministic smoke run (not the formal 480x960 device)
python -X utf8 experiments/run_bare_world.py --arm 4 --rows 96 --cols 192 --patches 40 \
    --ticks 100 --sample 25 --out _rerun_logs/bare_smoke
```

The `--ignore` above excludes 6 optional broker tests: that run is **1,580 passed / 0 failed /
10 skipped / 48 deselected** (1,638 collected). Without `--ignore`, with `fastapi`, `uvicorn` and
`websockets` installed, the same suite reads **1,586 passed** (1,644 collected). Re-measured at
2026-10-24: **111.4 s** idle; under concurrent load on a shared machine, **130.5 s** (broker
ignored) / **175.4 s** (broker included), which can exceed the 150 s budget — timing is
load-dependent, not a performance guarantee. The closure record reports **109.36 s** and
**140.23 s** for the 1,644-case set.

Windows: replace the activation line with `.venv\Scripts\activate` and prefix commands with
`PYTHONIOENCODING=utf-8` if your console code page is not UTF-8.

Determinism check (must reproduce the frozen digests):

```bash
python tools/env_field_digest.py --device s2 --mode classic   # 31d6acce…4072
python tools/env_field_digest.py --device s2 --mode default   # db9ccbb3…d295
```

The optimised path requires the compiled extension; `tests/test_broker.py` additionally needs
`fastapi`, `uvicorn`, and `websockets`.

## Data and scope

Experimental data (`.npz`, `.csv`, snapshots, figures) are **not** in this repository; they live in
a separate data repository. The code here is the apparatus, not the evidence bundle; the evidence
readings and file anchors are summarised in the public reports under `docs/公开/`.

This release is a **curated snapshot** of a closed project: it contains the source, tests,
design/preregistration/gate documents, and public reports. Internal operations material (task
dispatch, team administration, raw run logs, archived working files) is excluded.

## Documentation

- [docs/公开/README.md](docs/公开/README.md) — public report index and one-minute summary (Chinese)
- [01-最终汇报](docs/公开/01-最终汇报-20261024.md) — final report (Chinese; [English version](docs/公开/01-最终汇报-20261024.en.md))
- [02-科研账本](docs/公开/02-科研账本-20261024.md) — scientific ledger of positive and negative results (Chinese)
- [03-工程与复现](docs/公开/03-工程与复现-20261024.md) — engineering assets, build, tests, reproduction (Chinese)
- [04-方法与纪律](docs/公开/04-方法与纪律-20261024.md) — preregistration, review, and audit discipline (Chinese)
- [05-大气线技术纪要](docs/公开/05-大气线技术纪要-20261024.md) — atmospheric gate record (J1/J2/J3) (Chinese)
- [06-发布与表述边界](docs/公开/06-发布与表述边界-20261024.md) — what may and may not be claimed (Chinese)

## License and third-party components

Original code and documentation are licensed under the **Apache License 2.0** (see `LICENSE`).
Third-party components are not vendored; they are installed by the user and remain under their
own licenses. Notably:

- **SpeedyWeather.jl** (optional atmospheric integration) is **EUPL-1.2**, not Apache-2.0. It is an
  external dependency and is not distributed here.
- **exoplasim** (GPL-2.0) is **not part of this distribution**; the corresponding optional
  experiment script is excluded from the public snapshot.
- Rust crates, npm packages, and Python dependencies are used under their own licenses; the direct
  dependencies declared here are MIT/BSD/Apache-2.0-compatible. Transitive and optional
  dependencies must be checked against their own license texts.

See `NOTICE` and `THIRD-PARTY-NOTICES.md` for details.

## Citation

If you reference this project, use the metadata in `CITATION.cff`:

> the-world contributors (2026). *the-world: a deterministic spherical digital-life simulation
> platform* (Version 1.0.0) [Computer software]. https://github.com/fish827-08/the-world

## Maintenance

The project is **closed**. The repository is kept read-only in spirit: no new experiments, no
re-opening of the J1/J2 gates, no restart of J3 (archived NOT-RUN), and no retroactive changes to
frozen criteria or thresholds. Reopening requires a new preregistration and a new owner ruling,
and must not reinterpret the results recorded here.
