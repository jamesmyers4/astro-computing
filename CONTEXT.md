# Astro-Computing Project — CONTEXT.md

This file is the living project doc. Every Claude Code session should read this at start and append a short status block at the end (per Jimmy's `/grill-with-docs` convention). CLAUDE.md does not exist yet — it gets written once, at the end of the roadmap, distilling whatever conventions actually proved stable across sessions.

## Who's working on this

Jimmy (James Myers) — QA/SDET engineer, Knoxville TN. Strong in Playwright/TypeScript; Python is newer but actively used via a parallel DNA-computing project (`github.com/jamesmyers4/dna-computing`) that established the working pattern this project follows: pull real public data, run a model/analysis against it, formalize the result as pytest assertions. Comfortable reading code without comments — do not add inline comments to code in this repo.

## Machine setup

Linux laptop (HP, i7-7500U, 12GB RAM, Linux Mint Cinnamon) — worked on directly, physically, not remoted into. Native Linux, so none of the WSL/Windows headaches from the original plan apply: no Microsoft Store Python stub, no Remote-WSL layer, and critically, `lalsuite`/`pycbc` (needed for Session 2b) only ship prebuilt binaries for Linux/macOS — this machine is the actual reason that phase is viable at all. VS Code's integrated terminal works normally here, no special setup.

This machine already runs a local model (qwen2.5:3b-instruct via Ollama) — smaller than the desktop's llama3.1:8b. Relevant for the later agent-loop session (3b): decide then whether to use this local model or reach across the network to the desktop's Ollama instance.

Specs are modest but adequate for everything in this roadmap — nothing here is compute-heavy (light curve fitting, a single-event matched filter, a 5-body integration).

This is a different machine than where Session 0 was first attempted (JimmyPC/WSL2, abandoned due to WSL friction), so treat this as a clean start:
- Fresh clone: `git clone https://github.com/jamesmyers4/astro-computing.git` — the repo/README already exist on GitHub, nothing local from the JimmyPC attempt carries over.
- Git identity and GitHub auth need to be set up on this machine specifically (`git config --global user.name`/`user.email`, plus a PAT or SSH key) — the JimmyPC WSL credentials don't transfer.
- Verify `python3 -m venv` works out of the box; Mint sometimes splits this into a separate `python3-venv` apt package if it's missing.
- Quick `df -h` check before Session 2b, given `lalsuite` is a large prebuilt binary bundle.

## Code style for this repo

No inline `//` or `#` comments. Line break after a function or major block ends. No line breaks within a function body. Minimal formatting in chat/status updates — prose over bullets where possible.

## Golden pattern from the DNA-computing project

Pull real data → run analysis/model → write pytest assertions on the result, not print statements. Assert *relationships* between values, not just absolute numbers (e.g. `assert end_mfe < internal_mfe`), and double-check the direction of the relationship before asserting it — the DNA project's known trap was an MFE sign error (more negative = stronger binding). The equivalent traps here: Kepler flux dips are a *decrease*, LIGO significance should be higher *inside* the event window than outside it, orbital semi-major axes should stay *within* bounds not just below/above one.

## Roadmap and decisions (from grill-me session, 2026-08-27)

Order decided: Kepler-22b first (lower risk, matches the established pattern), then LIGO, then a black-hole physics toy simulation (added scope beyond the original handoff — Jimmy's actual passion point), then N-body, then SDSS. LIGO scope is full `pycbc` matched filtering, not just a `gwpy` visualization — confirmed pip-installable via `pip install lalsuite pycbc` with prebuilt binaries, no compiler toolchain or sudo needed, on Linux/macOS only (no Windows wheel). That constraint plus WSL friction on the original machine (JimmyPC) is why this project moved to the Linux laptop before Session 0 completed. Session granularity decided per-item below.

## Session catalog

## DO NOT Commit or work on more than one session at a time.
## Jimmy will commit manually

**Session 0 — environment bootstrap.** `python3 -m venv .venv`, activate, `pip install lightkurve astropy matplotlib pytest`, `.gitignore` (`.venv/`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `data/`), `requirements.txt`. Verify `import lightkurve` succeeds. Confirm the existing GitHub repo is what gets cloned into, not re-initialized.

**Session 1 — Kepler-22b transit.** Pull the light curve via `lightkurve`, flatten/normalize (data is noisy), plot, pytest assertion that minimum normalized flux drops below a threshold during the transit window. Reference: transit period ~289.9 days, transit depth ~0.049%. One session, same shape as `test_hybridization.py`.

**Session 2a — LIGO visualization.** `pip install gwpy`. Pull GW150914 strain data from GWOSC (public, no account needed). Bandpass filter ~35–350Hz. Plot the chirp. Get this working and visible before touching matched filtering.

**Session 2b — LIGO matched filtering.** `pip install lalsuite pycbc`. Build/compare against template waveforms, compute SNR, pytest assertion that peak significance inside the event window exceeds significance outside it. Split from 2a deliberately — heavier, more finicky library (older academic codebase, tighter numpy/scipy version constraints), worth isolating so a rough patch here doesn't block 2a's visual win.

**Session 3a — black hole toy sim, Schwarzschild.** Photon orbits, photon sphere, ISCO, static plot. New roadmap item, not in the original handoff — added because this is Jimmy's actual point of excitement in the field.

**Session 3b — black hole toy sim, Kerr + lensing.** Extends 3a with spin. Tentatively the home for the LLM agent-loop pattern later (perturb spin/impact parameter, score against a target lensing configuration) — this is the default choice over N-body because it has more physically interesting parameters to perturb; flip it if it doesn't pan out.

**Session 4 — N-body Solar System.** `pip install rebound`. Sun + 4 rocky planets, integrate 100 years, pytest assertion that each planet's semi-major axis stays within known bounds.

**Session 5 — SDSS / galaxy morphology.** Exploratory — pull catalog data via `astropy`'s SQL interface, Galaxy Zoo as the citizen-science angle. No clean pass/fail test expected here the way the other phases have one.

**LLM agent loop** — not a standalone phase. Wraps around Session 3b (default) or Session 4: randomize parameters, score against a target, log results, let a local model (Ollama, llama3.1:8b) reason about the next perturbation. Attach once the underlying session has a working baseline.

Test automation beyond the inline pytest assertions above is explicitly out of scope for these sessions — Jimmy has a separate custom skill for that, to be applied later.

## Status log

(Each session appends a short entry here after completion — what was built, what broke, what changed from the plan above.)

**Session 0 (2026-08-27) — done.** Cloned the real `jamesmyers4/astro-computing` repo in place (moved `.git` and `README.md` from a temp clone into this directory, keeping `CONTEXT.md` untracked alongside it, rather than re-initializing). Confirmed git identity/auth already configured on this machine (`jamesmyers4`, noreply email) and that `python3 -m venv` works out of the box on this Mint install, no separate `python3-venv` apt package needed. Created `.venv`, installed `lightkurve` (2.6.0), `astropy` (8.0.1), `matplotlib` (3.11.1), `pytest` (9.1.1) plus dependencies — install took a few minutes, pulled in scipy/pandas/scikit-learn/bokeh/astroquery among others. `import lightkurve` succeeds; one harmless warning that the `tpfmodel` submodule is unavailable without `oktopus`/`autograd` (PRF modeling, not needed for Session 1's transit work). Wrote `.gitignore` (`.venv/`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `data/`) and `requirements.txt` via `pip freeze` (75 pinned packages, for reproducibility). Nothing committed — left for Jimmy to review and commit manually. Next: Session 1, Kepler-22b transit.

**Session 1 (2026-08-27) — done.** Wrote `kepler22b.py` (pulls all 18 long-cadence quarters for Kepler-22 via `lightkurve.search_lightcurve(..., exptime=1800)`, caches FITS into `data/`, stitches, flattens with a 901-cadence window, removes outliers, phase-folds, bins, plots, saves `kepler22b_transit.png`) and `test_kepler22b.py` (same pull/fold logic inlined per DNA-computing's flat no-helper-function style, asserts the in-transit minimum flux both drops below an absolute threshold and sits below the out-of-transit baseline — the relational half of the assertion is the safeguard against the MFE-sign-error-style trap, i.e. asserting the dip is actually a decrease). Period/epoch/duration (289.863876 d, BJD 2454966.7001, 7.415 h) pulled from the NASA Exoplanet Archive (`pscomppars` table) rather than memorized from CONTEXT.md's approximate reference numbers, and hardcoded into both files like the DNA project hardcodes its sequences — matches CONTEXT's ~289.9 d / ~0.049% depth closely. One gotcha: the raw period-long fold has other noise-driven dips elsewhere in phase (visible in `kepler22b_transit.png` near phase -0.008, itself a leftover systematic from flattening, not a second transit) that are deeper than a naive flat threshold would expect — the test only checks flux inside a tight window around phase 0 sized off the real transit duration, so that unrelated dip can't accidentally satisfy the assertion. First threshold attempt (0.999) was wrong direction of tight — actual in-transit minimum (~0.99930) didn't clear it; loosened to 0.9995, still well below the ~0.99998 out-of-transit baseline. `pytest test_kepler22b.py` passes (~39s wall time, dominated by the MAST search query, not download — the 18 FITS files are cached in `data/`, 7.3MB total, gitignored). Nothing committed — left for Jimmy to review and commit manually. Next: Session 2a, LIGO visualization.