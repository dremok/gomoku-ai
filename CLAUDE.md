# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**Important**: This project uses a conda environment named `gomoku`. 

Always check if this conda environment is active before trying to activate it.
If not active, try to activate it before running any commands.

# Workarounds, Fallbacks & Heuristics
 
- Avoid "workarounds", "heuristics", and other garbage you might reach for when you don't understand the issue.
- If you think a heuristic is useful, then you are definitely wrong and this means we are taking a wrong approach.
- You MUST consider breaking out of local maxima in case you try to use heuristics.
- If necessary, backtrack (i.e. this didn't improve; delete this code)
- Don't be afraid to make edits that compromise the integrity of the code willy-nilly
- DO NOT keep tests/code paths just because you want to "get it done" or just because they align with your hypothesis!

## Development Approach

**Critical**: This project must be developed incrementally with verification at each small checkpoint. Never implement multiple components at once without testing.

### Development Workflow
1. **Small Steps**: Implement one small component at a time (e.g., single function, single class method)
2. **Immediate Testing**: After each implementation, write and run tests to verify it works correctly
3. **Checkpoint Verification**: Before moving to the next component, ensure current code runs without errors
4. **Incremental Building**: Each new component should build on verified, working foundations

### Example Incremental Approach
- Implement `Board.__init__()` → test board creation
- Add `Board.apply_move()` → test move application  
- Add `Board.legal_moves()` → test legal move generation
- Add basic win detection for one direction → test with simple cases
- Extend to all four directions → test edge cases
- Only then move to next major component

**Never skip verification steps**. If something doesn't work at a checkpoint, fix it immediately before proceeding.

### Critical: Always Stay in Baby Steps Mode
**NEVER** ask whether to move to "the next major component" vs "another baby step". This is a false choice that misunderstands the methodology.

- **Every step is a baby step** - this never changes
- Moving to a new major component IS a baby step if that's the natural next progression
- Baby steps continue throughout the entire development process
- The question is always: "What is the next natural baby step?" - whether that's within current functionality or starting something new

Example correct thinking:
- ✅ "The next baby step is to add basic draw detection"  
- ✅ "The next baby step is to start implementing Game class initialization"
- ❌ "Should we move to the next major component or continue with baby steps?"

Stay in baby step mode always. Each step builds incrementally on verified, working foundations.

### Communication Style for Next Steps
**Don't ask open-ended questions** like "What's the next natural baby step?" 

Instead:
- **Suggest the specific next step** you think makes most sense
- **Briefly explain why** it's the logical progression  
- **Ask for permission** to proceed in that direction

Example:
❌ "What's the next natural baby step?"
✅ "I think the next step should be implementing a simple RandomAgent class that can make moves automatically. This would let us test PvAI gameplay and provides the foundation for all AI agents. Should I proceed with this?"

Be decisive and provide direction while still asking for confirmation.

## Current Development Status

**Milestone Completed: M4 - Baseline Agents** ✅
- ✅ Board system (15×15, Standard 5-only rules) - 23/23 tests passing
- ✅ Game engine (turn management, state tracking) - 8/8 tests passing  
- ✅ CLI interface (PvP/PvAI modes) - fully playable
- ✅ RandomAgent (baseline AI) - 7/7 tests passing
- ✅ HeuristicAgent (pattern recognition AI) - 14/14 tests passing
- ✅ Agent evaluation (HeuristicAgent: 100% win rate vs RandomAgent > 80% requirement)

**Next Milestone: M5 - DQN Model & Inference** 🔄
- Implementing Deep Q-Network for Gomoku AI
- State encoding, network architecture, legal move masking
- DQNAgent class with PyTorch integration

**Decision: GUI Optional**
- M3 GUI (Tkinter) marked as optional since CLI works excellently
- Focus on core AI functionality first
- Can implement GUI later if needed for user experience

### Critical: Dependency Management
**ALWAYS** update `requirements.txt` when installing new packages. This is essential for:
- Reproducible environments
- Team collaboration
- Deployment consistency

After installing any package with `pip install <package>`, immediately add it to `requirements.txt` with appropriate version constraints.

### Progress Tracking in README
**ALWAYS** update the "Current Progress" section in README.md after completing any milestone or significant component. This helps track development status and provides visibility into what's been accomplished.

**When to update README progress**:
- After completing a major component (e.g., Board class, win detection, agent implementation)
- After reaching a milestone checkpoint 
- After adding significant functionality that changes project status
- Before moving to a new development phase

**How to update**:
- Mark completed items with ✅
- Update "Next" item with 🔄 
- Add recent accomplishments to "Recent Milestones Completed"
- Update overall "Status" to reflect current milestone progress

## Project Overview

This is a Gomoku AI project implementing a playable 15×15 board game with Deep Q-Learning (DQN) agent trained via self-play. The project follows Standard Gomoku rules (5-only) where wins require exactly 5 contiguous stones and overlines (≥6) do not count as wins.

## Key Architecture Decisions

- **Rules**: Standard Gomoku (5-only) - exactly 5 contiguous stones win, overlines invalid
- **Board**: 15×15 grid, Black plays first
- **Framework**: PyTorch for AI components
- **UI**: Tkinter GUI with CLI fallback
- **Compute**: CPU-only (no GPU requirements)
- **State Representation**: `np.int8[15,15]` with {0: empty, 1: black, -1: white}

## Current Repository Structure

```
gomoku-ai/
  gomoku/
    __init__.py
    core/
      __init__.py
      board.py          # ✅ Complete Board implementation with win/draw detection
      game.py           # ✅ Complete Game implementation with turn management  
    ai/
      __init__.py
      agents/
        __init__.py
        random_agent.py # ✅ RandomAgent implementation (7/7 tests passing)
        heuristic_agent.py # ✅ HeuristicAgent with pattern recognition (14/14 tests passing)
  scripts/
    play.py             # ✅ Full CLI with PvP/PvAI/AIvAI modes  
    evaluate_agents.py  # ✅ Agent evaluation framework (HeuristicAgent: 100% vs Random)
  tests/
    __init__.py
    test_board.py       # ✅ 23 Board tests (all passing)
    test_game.py        # ✅ 8 Game tests (all passing)
    test_random_agent.py # ✅ 7 RandomAgent tests (all passing)
    test_heuristic_agent.py # ✅ 14 HeuristicAgent tests (all passing)
  requirements.txt      # numpy, pytest
  CLAUDE.md            # This file
  README.md            # Development plan with current progress

# Next: DQN Implementation
  gomoku/ai/models/    # DQN network architecture (pending)
  gomoku/ai/training/  # Training and replay components (pending)
  models/              # Trained model weights (pending)
  config/              # Configuration files (pending)
  gomoku/ui/           # GUI components (OPTIONAL - CLI works well)
```

## Common Development Commands

```bash
# Activate environment (required for all commands)
conda activate gomoku

# Install dependencies
pip install -r requirements.txt

# Additional dependencies for full development
pip install torch pyyaml rich click tqdm matplotlib pytest black ruff

# Run tests
pytest -q

# Play Gomoku (fully working)
python scripts/play.py                        # Interactive CLI with 4 modes:
                                              # 1. PvP, 2. PvAI-Random, 3. PvAI-Heuristic, 4. AIvAI

# Evaluate agents (working)
python scripts/evaluate_agents.py            # Test agent performance (HeuristicAgent vs RandomAgent)

# Future commands (not yet implemented):
# python scripts/train.py --config config/train.yaml # Train DQN
# python scripts/evaluate.py --p1 dqn --p2 heuristic --games 500 # DQN evaluation
```

## Critical Implementation Details

### Win Detection Algorithm
For each move at position (r,c), check 4 directions: horizontal (→), vertical (↓), diagonal (↘), anti-diagonal (↗):
- Count contiguous stones in both directions
- Win condition: exactly 5 stones AND no extension to 6+ (overline check)
- Must validate both ends don't continue with same color

### DQN Model Specifications
- **Input**: 4-channel state encoding (current player stones, opponent stones, turn plane, last-move plane)
- **Architecture**: Dueling Double-DQN with 4 conv layers + dueling heads (value + advantage)
- **Output**: 225 logits (15×15 board positions) with illegal move masking
- **Action mapping**: `idx = 15 * row + col`

### Training Configuration
Key hyperparameters from the development plan:
- Gamma: 0.99, Learning rate: 1e-3, Batch size: 256
- Replay capacity: 200k, Target update: soft τ=0.01
- Epsilon schedule: 0.30 → 0.02 over 200k steps
- Self-play with symmetry augmentation (8 board symmetries)

## Testing Requirements

All implementations must include comprehensive tests:
- `test_board.py`: Board state, legal moves, placement validation
- `test_rules.py`: Win detection including overline cases, edge conditions
- `test_game.py`: Game flow, turn management, terminal states
- `test_masking.py`: Legal move masking for AI agents

## Performance Benchmarks

- **Heuristic vs Random**: Must achieve ≥80% win rate over 200 games
- **DQN vs Random**: Target ≥70% win rate after initial training
- **DQN vs Heuristic**: ≥55% win rate over ≥500 games for model promotion
- **GUI responsiveness**: Move latency <300ms on CPU

## File Naming and Conventions

- Use snake_case for Python files and functions
- Model files: `.pt` format with accompanying `metadata.json`
- Config files: YAML format in `config/` directory
- Transcripts: JSONL format with game metadata
- Reports: JSON format with timestamps and evaluation metrics