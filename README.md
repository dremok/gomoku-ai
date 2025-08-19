# Gomoku AI (15×15, Standard 5-only) — Development Plan

## 🚀 Current Progress

**Status**: M4 — Baseline Agents (COMPLETED) + M5 — DQN Model & Inference (COMPLETED) + M6 — Self-Play & Replay (COMPLETED)

### ✅ **Completed Milestones**:

**M1-M2: Core Engine & CLI** ✅
- ✅ **Board System**: Complete 15×15 board with move validation, win detection (all 4 directions), draw detection
- ✅ **Game Engine**: Full game flow with turn management, state tracking (ongoing/win/draw)
- ✅ **Win Detection**: Standard 5-only rules implemented - exactly 5 stones win, overlines (6+) rejected
- ✅ **CLI Interface**: Fully playable command-line Gomoku with 6 game modes

**M4: Baseline Agents** ✅
- ✅ **RandomAgent**: Uniform random move selection (7/7 tests passing)
- ✅ **HeuristicAgent**: Pattern recognition AI with immediate win/block detection (14/14 tests passing)
- ✅ **Agent Evaluation**: HeuristicAgent achieves 100% win rate vs RandomAgent (exceeds 80% requirement)

**M5: DQN Model & Inference** ✅
- ✅ **State Encoding**: 4-channel tensor format (current/opponent stones, turn plane, last-move plane) (11/11 tests)
- ✅ **Neural Network**: Dueling Double-DQN architecture with legal move masking (12/12 tests)
- ✅ **Action Selection**: ε-greedy policy with masked Q-values (16/16 tests)
- ✅ **DQN Agent**: Complete PyTorch integration with save/load functionality (17/17 tests)
- ✅ **CLI Integration**: 6 game modes including DQN vs Human, DQN vs AI

**M6: Self-Play & Replay** ✅
- ✅ **Replay Buffer**: Experience storage with prioritized sampling option (19/19 tests)
- ✅ **Self-Play System**: Multi-game generation with opponent rotation (15/15 tests)
- ✅ **Training Data**: Complete data pipeline with batch generation (18/18 tests)
- ✅ **Integration**: Full training infrastructure ready

### 🔄 **Next: M7 — Training Loop & Optimization**
- Implementing DQN training algorithm with experience replay
- Loss functions, target network updates, optimizer setup
- Training monitoring and checkpointing

**Current Test Status**: **160/160 tests passing** ✅

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
      __init__.py
      board.py          # ✅ Complete Board implementation (23/23 tests)
      game.py           # ✅ Complete Game implementation (8/8 tests)
    ai/
      __init__.py
      agents/
        __init__.py
        random_agent.py # ✅ RandomAgent implementation (7/7 tests)
        heuristic_agent.py # ✅ HeuristicAgent implementation (14/14 tests)
        dqn_agent.py    # ✅ DQNAgent with PyTorch integration (17/17 tests)
      models/
        __init__.py
        state_encoding.py # ✅ 4-channel state encoding (11/11 tests)
        dqn_model.py    # ✅ Dueling Double-DQN architecture (12/12 tests)
        action_selection.py # ✅ ε-greedy action selection (16/16 tests)
      training/
        __init__.py
        replay_buffer.py # ✅ Experience replay system (19/19 tests)
        self_play.py    # ✅ Self-play game generation (15/15 tests)
        data_utils.py   # ✅ Training data management (18/18 tests)
  scripts/
    play.py             # ✅ Full CLI with 6 game modes (PvP/PvAI/AIvAI)
    evaluate_agents.py  # ✅ Agent evaluation framework
    demo_self_play.py   # ✅ Self-play system demonstration
    quick_data_demo.py  # ✅ Local data directory demo
  tests/
    __init__.py
    test_board.py       # ✅ 23 Board tests (all passing)
    test_game.py        # ✅ 8 Game tests (all passing)
    test_random_agent.py # ✅ 7 RandomAgent tests (all passing)
    test_heuristic_agent.py # ✅ 14 HeuristicAgent tests (all passing)
    test_dqn_agent.py   # ✅ 17 DQNAgent tests (all passing)
    test_state_encoding.py # ✅ 11 state encoding tests (all passing)
    test_dqn_model.py   # ✅ 12 DQN model tests (all passing)
    test_action_selection.py # ✅ 16 action selection tests (all passing)
    test_replay_buffer.py # ✅ 19 replay buffer tests (all passing)
    test_self_play.py   # ✅ 15 self-play tests (all passing)
    test_data_utils.py  # ✅ 18 data utilities tests (all passing)
  requirements.txt      # numpy, pytest, torch
  CLAUDE.md            # Development guidance
  README.md            # This file
  .gitignore           # Ignores training data and temporary files
  
# Local Data Storage (git-ignored):
  data/                # Local training data, models, and logs
    training/          # Self-play session data
    models/            # Saved model checkpoints
    logs/              # Training logs and metrics
    replays/           # Game replay files
  
# Pending Implementation:
  config/              # Training configurations (M7)
  gomoku/ui/           # GUI components (Optional - CLI works excellently)
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
# Setup environment (conda recommended)
conda activate gomoku  # or create: conda create -n gomoku python=3.10

# Install dependencies
pip install -r requirements.txt
# Core: numpy pytest torch

# Run tests
pytest -q  # All 160 tests should pass ✅

# Play Gomoku (6 game modes available)
python scripts/play.py
# 1. Player vs Player (PvP)
# 2. Player vs AI - Random (Easy)
# 3. Player vs AI - Heuristic (Hard)  
# 4. Player vs AI - DQN (Neural Network - Untrained)
# 5. AI vs AI - Random vs Heuristic
# 6. AI vs AI - Heuristic vs DQN

# Evaluate agent performance
python scripts/evaluate_agents.py  # HeuristicAgent vs RandomAgent benchmark

# Demo self-play system
python scripts/demo_self_play.py   # Shows training data generation
python scripts/quick_data_demo.py  # Quick demo of local data storage

# Future commands (M7+):
# python scripts/train.py --config config/train.yaml # Train DQN
# python scripts/evaluate.py --p1 dqn --p2 heuristic --games 500 # DQN evaluation
```

---

## 8) Testing Matrix

**Unit Tests (160/160 passing)** ✅

**Core System:**
* `test_board.py` (23): Board state, legal moves, win detection (all 4 directions), overline rejection, draw detection
* `test_game.py` (8): Game flow, turn management, move validation, terminal states

**AI Agents:**
* `test_random_agent.py` (7): Random move selection, reproducibility, game integration
* `test_heuristic_agent.py` (14): Pattern recognition, win/block detection, center preference
* `test_dqn_agent.py` (17): Neural network integration, action selection, save/load functionality

**DQN Neural Network:**
* `test_state_encoding.py` (11): 4-channel state representation, legal move masking
* `test_dqn_model.py` (12): Dueling Double-DQN architecture, forward pass shapes, gradient flow
* `test_action_selection.py` (16): ε-greedy policy, action masking, coordinate conversion

**Training Infrastructure:**
* `test_replay_buffer.py` (19): Experience storage, sampling, prioritized replay, persistence
* `test_self_play.py` (15): Game generation, experience collection, opponent rotation
* `test_data_utils.py` (18): Data loading, batch generation, training dataset creation

**Integration Tests**

* ✅ CLI PvP/PvAI/AIvAI end-to-end (6 game modes working)
* ✅ DQN agent legal move enforcement 
* ✅ Self-play produces training data (demonstrated in demo)
* ✅ Agent save/load functionality with model persistence
* ✅ Training data pipeline from games to neural network batches

---

## 9) Acceptance Criteria (Summary)

**Completed Criteria** ✅
* ✅ **Engine:** Exact-5 wins only; overlines invalid; all 160 tests passing
* ✅ **Heuristic > Random:** 100% win rate achieved (exceeds 80% requirement)
* ✅ **DQN Infrastructure:** Complete neural network, state encoding, action selection ready
* ✅ **Self-Play System:** Training data generation and experience replay working
* ✅ **CLI App:** 6 game modes playable (PvP/PvAI/AIvAI), DQN integration complete

**Pending Criteria** (M7+)
* 🔄 **Training:** DQN training loop with loss optimization and checkpointing
* 🔄 **Arena Gate:** DQN ≥55% vs Heuristic over ≥500 games to promote `best.pt`
* 🔄 **Artifacts:** `models/best.pt` + `metadata.json` + eval reports

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
