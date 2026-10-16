## PROJECT DEFINITION (shared by every role)
**What "post-training" means here.** We take an already pretrained small instruction-tuned language
model and improve one narrow skill with supervised fine-tuning (SFT) using LoRA adapters. It is
not pretraining, and it is not RLHF or reinforcement learning (Nebius offers those only as a
separate, restricted service; we do not use them).

**What is being post-trained.** The base model named in your brief (default `Qwen/Qwen3-0.6B`,
set by `POSTTRAIN_BASE_MODEL`), trained as a LoRA on Nebius Token Factory. Only the adapter is
trained; the base weights are frozen.

**The skill.** Tool calling on small deterministic games: given a game state and one tool, emit
exactly one correct tool call with correct arguments. Games: tic-tac-toe (`place_mark`, win now else
block), Nim (`take_stones`, leave a multiple of 4), number guessing (`guess_number`, midpoint rounded
down) and rock-paper-scissors (`play_move`, beat the predicted next move).

**The training data.** Chat-format rows whose last message is the correct assistant tool call,
built only from TRAIN-split tasks and failures, plus rule-derived variations whose answers are
recomputed from the game rule. Every row is checked by a deterministic verifier. Nothing from the
held-out or regression splits is ever used.

**How success is measured.** Exact-match accuracy of the tool call (tool name AND arguments) on a
sealed held-out split, with a sealed regression split to catch forgetting. A deterministic gate
promotes a candidate only if it beats the champion by at least 10% relative on the same suite,
split and seed, with verified evidence on both sides, and loses at most 0.05 absolute on regression.

**Principles.** Measurements come only from tools; reasoning is labelled `EXPLANATION`; one
training job per experiment, approved by a human; honest negative results are valid outcomes.
