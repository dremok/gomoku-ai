# Gomoku AI (15×15, Standard 5-only) — Development Plan

## 🚀 Current Progress

**Status**: M1 — Core Engine (In Progress)
- ✅ **Board Initialization**: Empty 15×15 board creation with proper data types
- ✅ **Move Application**: Stone placement with validation (bounds, empty cells, valid players)  
- 🔄 **Next**: Legal moves generation and basic win detection

**Recent Milestones Completed**:
- Initial project structure with conda environment (`gomoku`)
- Board class with comprehensive test coverage (3/3 tests passing)
- Basic move validation and application functionality

---

## 1) Goals & Deliverables

**Goal:** A playable Gomoku app on a 15×15 board plus a Deep Q-Learning (DQN) agent trained via self-play, all CPU-friendly.

**Deliverables**

* **App:** Play **PvP / PvAI / AIvAI** with a simple **GUI (Tkinter)** and a CLI fallback.
* **Training tool:** Self-play generator + DQN trainer (Double-DQN + Dueling) with experience replay.
* **Model artifacts:** Trained weights (`.pt`), `metadata.json`, and training logs.
* **Docs:** README with install, play, train, evaluate; configs; acceptance criteria.
* **License:** MIT.

---

## 2) Decisions (Locked-In)

* **Rules:** **Standard Gomoku (5-only)**
  Win requires **exactly 5** contiguous stones in any direction; **overlines (≥6)** do **not** count. No Renju bans.
* **Board:** 15×15. Black plays first.
* **Framework:** PyTorch.
* **UI:** Tkinter GUI (minimal, no extra deps) + CLI.
* **Compute:** CPU (no GPU).
* **Telemetry:** Local logs only (Rich/TQDM); no external services.

> ✅ **PAUSE-0 CHECK** — These are final. Any change here impacts all subsequent tasks.

---

## 3) Repository Structure

```
gomoku-ai/
  gomoku/
    __init__.py
    core/
      board.py          # state, moves, win/draw check
      rules.py          # Standard(5-only) helpers
      game.py           # turn loop, transcript
    ui/
      cli.py            # CLI fallback
      tk_gui.py         # Tkinter board + controls
    ai/
      agents/
        random_agent.py
        heuristic_agent.py
        dqn_agent.py
      models/
        dqn_model.py    # dueling double-DQN net
      training/
        replay_buffer.py
        self_play.py
        train_dqn.py
        evaluate.py
    utils/
      config.py
      log.py
      symmetry.py
  scripts/
    play.py             # entry for GUI/CLI
    train.py            # entry for training
    evaluate.py         # entry for arenas
  models/
    README.md           # explains model files
  config/
    train.yaml
    eval.yaml
  tests/
    test_board.py
    test_rules.py
    test_game.py
    test_masking.py
  README.md
  LICENSE
  pyproject.toml or requirements.txt
```

> ✅ **PAUSE-1 CHECK** — Project builds, `pytest -q` runs (even if most tests are TODO/xfail initially), skeleton scripts exist.

---

## 4) Milestones with Checkpoints

### M1 — Core Engine (Deterministic & Tested)

* **Board state:** `np.int8[15,15]` with {0: empty, 1: black, -1: white}.
* **Moves:** `(row, col)` 0-based; legal if empty.
* **Apply/undo:** Fast in-place apply; optional undo for tree search experiments.
* **Win check (5-only):**

  * For the **last move**, count contiguous stones both directions for 4 axes (→, ↓, ↘, ↗).
  * **Win iff** count == 5 **and** neither side extends to 6+ of same color. (Overline invalid.)
* **Draw:** board full (225 moves) and no 5-exact.
* **Masks:** Generate boolean mask of legal moves (225 outputs).

**Unit tests**

* All four directions (3–6 in a row) → only 5 counts as win.
* Overline (6+) → **not** a win.
* Edge/corner sequences.
* Draw after 225 moves.
* Mask excludes occupied cells.

> ✅ **PAUSE-2 CHECK** — All tests pass; winner detection is O(1) around last move.

---

### M2 — CLI + Game Wrapper

* **CLI app (`scripts/play.py`):** modes `pvp | pvai | aivai`; ASCII board; move input `e.g. "H8" or "7 7"`.
* **Transcripts:** JSONL with `{game_id, move_no, player, row, col, ts}` into `runs/games/`.

**Acceptance**

* Complete PvP games without crash; transcripts saved.

> ✅ **PAUSE-3 CHECK** — Manual PvP session OK; 2–3 transcripts on disk.

---

### M3 — GUI (Tkinter)

* **Tk board:** 15×15 grid, click to place; highlight last move; status bar (turn, result).
* **Modes:** PvP / PvAI / AIvAI; drop-downs for agent type (Random/Heuristic/DQN), ε for DQN, side selection.
* **Controls:** New game, Undo (PvP only), Save transcript, Load model.

**Acceptance**

* Human can play **PvAI** end-to-end; no freezes; reasonable responsiveness on CPU.

> ✅ **PAUSE-4 CHECK** — GUI smoke test done across all 3 modes.

---

### M4 — Baseline Agents

* **RandomAgent:** uniform over legal moves.
* **HeuristicAgent:** simple pattern scoring on local neighborhoods:

  * Immediate win/lose detection (take 5 or block opponent’s 4-open/4-semi).
  * Scores for open/closed 2/3/4; prefer center proximity tie-break.

**Arena**

* `scripts/evaluate.py --p1 heuristic --p2 random --games 200 --swap-colors`
* Expect **Heuristic > Random** (>80% wins).

> ✅ **PAUSE-5 CHECK** — Result meets threshold with 95% CI.

---

### M5 — DQN Model & Inference

* **State encoding (channels, 15×15):**
  `C1` current player stones, `C2` opponent stones, `C3` turn plane (1 for black-to-play else 0), `C4` last-move plane (optional).
* **Network (Dueling, Double-DQN compatible):**
  4× Conv(3×3, padding=1) + ReLU + (BatchNorm or LayerNorm) → flatten →
  Dueling heads:

  * Value: MLP → scalar V
  * Advantage: MLP → 225 logits → reshape to board → mask illegal → valid Q
    Combine Q = V + (A − mean(A\_valid)).
* **Action selection:** ε-greedy on masked Q.
* **Save/load:** `.pt` + `metadata.json` (board\_size, channels, rule\_variant, config hash).

**Tests**

* Forward shape (B, 225).
* Masking never returns illegal action.
* Save → Load → identical outputs on fixed input.

> ✅ **PAUSE-6 CHECK** — `dqn_agent` acts legally on random states; round-trip weights OK.

---

### M6 — Self-Play & Replay

* **Replay buffer (uniform to start):**
  Stores `(s, a_idx, r, s', done, legal_mask_next, game_id, move_no)`.
  Disk snapshots every N steps (`.npz` or `.pt`).
* **Self-play driver:**
  Two agents (same net, different ε: e.g., 0.25 vs 0.05), alternate colors per game.
  Rewards: +1 win / −1 loss / 0 draw; optional small per-move penalty (−0.001).
* **Symmetry augmentation:** 8 board symmetries applied to `(s,a)` pairs (rotate/flip) to multiply samples.

**Acceptance**

* Generate ≥50 episodes and persist buffer; print basic stats (avg len, results split).

> ✅ **PAUSE-7 CHECK** — `scripts/train.py --self-play-only --episodes 50` produces files & summary.

---

### M7 — Training Loop (CPU-Friendly)

* **Algorithm:** Double DQN + Dueling + Target network.
* **Targets:**
  `a* = argmax_a' Q_online(s',a')`
  `y = r + γ * (1−done) * Q_target(s', a*)`
* **Hyperparams (start point, CPU-aware):**

  * γ = 0.99
  * lr = 1e-3 (Adam)
  * batch\_size = **256** (lower if needed), grad clip 1.0
  * replay\_capacity = 200k, min\_replay = 5k
  * target update = soft τ=0.01 (or hard every 2k steps)
  * ε schedule: start 0.30 → 0.02 over 200k env steps
  * Eval matches every 5k steps vs Random + Heuristic (200 games each)
* **Logging:** CSV/JSONL for loss, q\_mean, q\_std, ε, steps/sec; Rich console table.
* **Checkpointing:** every N steps; keep symlink `models/best.pt` by eval win-rate.

**Acceptance**

* Short run (e.g., 30–60 min on CPU) shows decreasing loss & improved win-rate vs Random (>70%).
* Checkpoint load resumes training identically (seeded).

> ✅ **PAUSE-8 CHECK** — Metrics trend in right direction; checkpoints valid.

---

### M8 — Arena, Elo & Regression Safety

* **Arena:** Round-robin DQN (current) vs Random, Heuristic, and prior snapshots.
* **Color balancing:** swap colors half the games.
* **Elo (optional):** Simple logistic update per match-up.
* **Report:** `reports/eval_{ts}.json` with win/draw/loss, 95% CI, optional Elo changes.

**Gate**

* Promote checkpoint to `best.pt` only if it **beats Heuristic ≥55%** over **≥500** games.

> ✅ **PAUSE-9 CHECK** — Promotion criteria satisfied; `best.pt` updated.

---

### M9 — User App Integration

* **GUI integration:** menu option to load `models/best.pt`.
* **Hints:** optional top-k Q suggestions overlay.
* **CLI:**
  `python scripts/play.py --mode pvai --model models/best.pt --epsilon 0.02`

**Acceptance**

* Smooth human play; no stalls; move latency acceptable on CPU (<300ms typical).

> ✅ **PAUSE-10 CHECK** — Manual QA of PvAI & AIvAI with `best.pt`.

---

### M10 — Packaging, Docs & Release

* **README:** install, play, train, evaluate, known limits.
* **`pyproject.toml` or `requirements.txt`** pinned minor versions.
* **License:** MIT.
* **Reproducibility:** seed, env dump, config snapshot in `metadata.json`.

**Release bundle**

* `models/best.pt`, `models/metadata.json`, top eval report, example transcripts.

> ✅ **PAUSE-11 CHECK** — Fresh machine can follow README and reach “play a game” in <10 minutes.

---

## 5) Technical Specs & Notes

### 5.1 Win Detection (Standard 5-only)

For the last move `(r,c,player)`:

* For each direction `d ∈ {(0,1),(1,0),(1,1),(-1,1)}`:
  count contiguous stones both ways; let `k = left + 1 + right`.
* If `k == 5`, **win candidate**.
* **Overline test:** check immediate cells beyond both ends; if either end continues with same color → overline → **invalidate**.
* Overall **win iff** any direction yields 5-exact (not overline).

### 5.2 DQN Model (CPU)

* Prefer **few filters** (e.g., 64) and 4 conv blocks to stay fast.
* Use **torch.compile(False)** on older CPUs; keep it simple.
* Batch states during training; single-state forward for inference in GUI.

### 5.3 Replay & Symmetries

* Map action index ↔ (row,col): `idx = 15*row + col`.
* For each symmetry, transform both the planes and the action coordinate.

### 5.4 Evaluation Protocol

* Deterministic seeds per arena to reproduce results.
* Color swap and starting player balance.
* Report CI via Wilson interval.

---

## 6) Configs

**`config/train.yaml`**

```yaml
seed: 42
device: cpu
board_size: 15
rule_variant: standard_5_only

dqn:
  gamma: 0.99
  lr: 0.001
  batch_size: 256
  replay_capacity: 200000
  min_replay: 5000
  target_update: soft
  tau: 0.01
  double: true
  dueling: true
  grad_clip: 1.0

epsilon:
  start: 0.30
  end: 0.02
  decay_steps: 200000

self_play:
  episodes: 200000
  epsilon_black: 0.25
  epsilon_white: 0.05
  symmetry_augment: true
  step_penalty: -0.001

eval:
  every_steps: 5000
  games_per_opponent: 200
  opponents: [random, heuristic]
  promote_threshold_vs_heuristic: 0.55
  promote_games: 500
```

**`config/eval.yaml`**

```yaml
games: 500
opponents: [random, heuristic]
swap_colors: true
seed: 123
```

---

## 7) Commands Cheat-Sheet

```bash
# Install (venv optional)
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy pyyaml rich click tqdm matplotlib pytest black ruff

# Tests
pytest -q

# Play (GUI by default; CLI fallback via flag)
python scripts/play.py --mode pvai --ai1 dqn --model models/best.pt --epsilon 0.02
python scripts/play.py --mode pvp --cli

# Evaluate baselines
python scripts/evaluate.py --p1 heuristic --p2 random --games 200 --swap-colors

# Self-play data only (sanity)
python scripts/train.py --config config/train.yaml --self-play-only --episodes 50

# Full training
python scripts/train.py --config config/train.yaml

# Arena vs baselines & snapshots
python scripts/evaluate.py --config config/eval.yaml --model models/best.pt
```

---

## 8) Testing Matrix

**Unit**

* `test_board.py`: placement, masks, immutability guarantees if used.
* `test_rules.py`: all win/overline cases; draw; edge lines.
* `test_game.py`: turn order, terminal states.
* `test_masking.py`: illegal moves never chosen.

**Integration**

* CLI PvP end-to-end.
* GUI PvAI end-to-end.
* DQN save/load parity.
* Self-play produces non-empty buffer.
* Training reduces loss on a fixed small buffer (overfitting test).

---

## 9) Acceptance Criteria (Summary)

* **Engine:** exact-5 wins only; overlines invalid; tests green.
* **Heuristic > Random:** ≥80% win (200 games).
* **Training:** Double-DQN stable; vs Random ≥70% after short run; checkpoints reproducible.
* **Arena Gate:** DQN ≥55% vs Heuristic over ≥500 games to promote `best.pt`.
* **App:** GUI playable (PvP/PvAI/AIvAI), load `best.pt`, no crashes.
* **Artifacts:** `models/best.pt` + `metadata.json` + top eval report and transcripts.

---

## 10) Risks & Mitigations

* **Slow CPU training:** keep model small; reduce batch; more self-play wall-time; use symmetries for sample efficiency.
* **Overfitting to self-play quirks:** mix in heuristic opponent games (e.g., 10–30%) during training.
* **Rule ambiguity:** explicitly enforce 5-exact + overline invalid in tests.

---

## 11) Roadmap (Optional Enhancements)

* Prioritized replay; curiosity reward.
* Hybrid DQN + shallow MCTS for move lookahead.
* Web UI (FastAPI + small React board).
* Opening fairness (swap2 rule) for evaluation only.
* AlphaZero-style policy/value (bigger scope).

---

## 12) File Stubs (Agent can scaffold immediately)

* `gomoku/core/board.py`: `Board(state, last_move, to_play)`, `apply(move)`, `legal_mask()`, `check_winner(last_move)`
* `gomoku/core/rules.py`: win/overline helpers
* `gomoku/core/game.py`: `Game(reset, step, is_terminal, result)`
* `gomoku/ui/tk_gui.py`: `GomokuApp` with modes and event handlers
* `gomoku/ai/agents/{random,heuristic,dqn}_agent.py`: `select_action(state, legal_mask, epsilon)`
* `gomoku/ai/models/dqn_model.py`: `DuelingDQN`
* `gomoku/ai/training/{replay_buffer,self_play,train_dqn,evaluate}.py`
* `scripts/{play,train,evaluate}.py` (Click/Rich CLIs)
* `tests/*` (as detailed above)
