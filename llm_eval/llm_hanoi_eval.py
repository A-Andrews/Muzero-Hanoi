#!/usr/bin/env python3
"""LLM evaluation on Tower of Hanoi problems.

Evaluates an instruction-tuned LLM on the same Close / Moderate / Far problem
sets used for the MuZero ablation experiments and saves results in a format
compatible with the existing MuZero plotting pipeline.

Metrics saved per condition:
  • Mean excess moves (steps_taken − optimal_moves)  — same as MuZero error
  • Solve rate
  • Illegal-move rate

Output files (under stats/Hanoi/<timestamp>/<difficulty>/):
  LLM_<label>_<prompting>_actingAccuracy.pt       shape (1, 2):  [[0, mean_error]]
  LLM_<label>_<prompting>_actingAccuracy_error.pt  shape (1, 3):  [[0, mean_error, std_error]]
  LLM_<label>_<prompting>_results.json            full per-episode data

Usage example:
    python llm_eval/llm_hanoi_eval.py \\
        --timestamp 1748875208 \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --model_label llama3_8b \\
        --prompting zero_shot \\
        --start 0 \\
        --episodes 50 \\
        --temperature 0.7
"""

import argparse
import itertools
import json
import logging
import os
import re
import signal
import sys
import time

import numpy as np
import torch

# Make project root importable regardless of CWD
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from env.hanoi import TowersOfHanoi
from env.hanoi_utils import hanoi_solver
from llm_eval.prompts import PROMPT_BUILDERS, state_to_text

# ---------------------------------------------------------------------------
# Move encoding helpers
# ---------------------------------------------------------------------------

MOVES = list(itertools.permutations([0, 1, 2], 2))
# [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)] — 0-indexed pegs


def pegs_to_action(from_peg: int, to_peg: int) -> int:
    """Return action index for a move between 0-indexed pegs."""
    return MOVES.index((from_peg, to_peg))


def _last_match(pattern: str, text: str, flags: int = 0) -> re.Match | None:
    """Return the *last* match of ``pattern`` in ``text``."""
    matches = list(re.finditer(pattern, text, flags))
    return matches[-1] if matches else None


def parse_move(response: str) -> tuple[int, int] | None:
    """Extract (from_peg, to_peg) from LLM text — all 0-indexed.

    Uses the **last** match in the response so that CoT reasoning about
    intermediate or illegal moves doesn't shadow the final answer.

    Tries several patterns in decreasing specificity.  Returns None if no
    valid move can be extracted.
    """
    # Pattern 1: canonical "Move from Peg X to Peg Y"
    m = _last_match(
        r"move\s+from\s+peg\s+([123])\s+to\s+peg\s+([123])",
        response,
        re.IGNORECASE,
    )
    if m:
        fp, tp = int(m.group(1)) - 1, int(m.group(2)) - 1
        if fp != tp:
            return fp, tp

    # Pattern 2: looser — "from X to Y" with optional "Peg"
    m = _last_match(
        r"from\s+(?:peg\s+)?([123])\s+to\s+(?:peg\s+)?([123])",
        response,
        re.IGNORECASE,
    )
    if m:
        fp, tp = int(m.group(1)) - 1, int(m.group(2)) - 1
        if fp != tp:
            return fp, tp

    # Pattern 3: two lone digits separated by arrow or dash
    m = _last_match(
        r"\b([123])\s*(?:→|->|to|-)\s*([123])\b",
        response,
        re.IGNORECASE,
    )
    if m:
        fp, tp = int(m.group(1)) - 1, int(m.group(2)) - 1
        if fp != tp:
            return fp, tp

    return None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_id: str, cache_dir: str | None, dtype: str):
    """Load model and tokenizer directly (no pipeline wrapper).

    Avoids importing transformers.pipeline which pulls in torchvision image
    utilities that conflict with the torchvision version in this environment.

    Returns:
        (model, tokenizer) tuple.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    logging.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logging.info(f"Loading model: {model_id}  dtype={dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    model.eval()
    logging.info("Model loaded successfully.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Layer intervention hooks
# ---------------------------------------------------------------------------

def register_layer_hooks(model, ablate_layer: int, noise_scale: float, noise_layer: int) -> list:
    """Register forward hooks for layerwise ablation or noise injection.

    Args:
        ablate_layer: transformer block index to bypass (-1 = disabled).
            The hook replaces the layer output with its input, effectively
            skipping that layer's computation (residual pass-through only).
        noise_scale: std dev of Gaussian noise to inject (0.0 = disabled).
        noise_layer: which layer to inject noise at (-1 = all layers).

    Returns:
        list of hook handles — call handle.remove() to clean up.
    """
    handles = []

    # Access transformer blocks.  Works for LlamaForCausalLM and most
    # decoder-only models (model.model.layers or model.transformer.h).
    try:
        layers = model.model.layers
    except AttributeError:
        try:
            layers = model.transformer.h
        except AttributeError:
            raise ValueError(
                "Cannot locate transformer layers on this model. "
                "Expected model.model.layers or model.transformer.h."
            )

    if ablate_layer >= 0:
        if ablate_layer >= len(layers):
            raise ValueError(
                f"--ablate_layer {ablate_layer} is out of range "
                f"(model has {len(layers)} layers)."
            )

        def _ablate_hook(module, inputs, outputs):
            # outputs is a tuple; first element is the hidden state.
            # Return the input hidden state unchanged (bypass this layer).
            in_hidden = inputs[0]
            return (in_hidden,) + outputs[1:]

        handles.append(layers[ablate_layer].register_forward_hook(_ablate_hook))
        logging.info(f"Layer ablation hook registered on layer {ablate_layer}.")

    if noise_scale > 0.0:
        target_layers = range(len(layers)) if noise_layer < 0 else [noise_layer]
        for idx in target_layers:
            def _noise_hook(module, inputs, outputs, _scale=noise_scale):
                hidden = outputs[0]
                noisy = hidden + torch.randn_like(hidden) * _scale
                return (noisy,) + outputs[1:]

            handles.append(layers[idx].register_forward_hook(_noise_hook))
        logging.info(
            f"Noise hooks registered (scale={noise_scale}) on "
            f"{'all' if noise_layer < 0 else f'layer {noise_layer}'} layer(s)."
        )

    return handles


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate_response(
    model,
    tokenizer,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
) -> str:
    """Tokenise prompt using the model's chat template, generate, decode only the new tokens."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=0.9 if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: TowersOfHanoi,
    model,
    tokenizer,
    prompt_fn,
    temperature: float,
    max_new_tokens: int,
    history_length: int,
    n_disks: int,
) -> dict:
    """Run one episode and return per-episode statistics.

    The env must have been reset before calling this function.

    Returns:
        dict with keys:
            error          – steps_taken − optimal_moves (int; includes unsolved)
            solved         – whether the goal was reached (bool)
            illegal_count  – number of illegal moves attempted (int)
            total_steps    – total steps taken (int)
            illegal_rate   – illegal_count / total_steps (float)
            parse_failures – number of steps where LLM output couldn't be parsed (int)
    """
    state_tuple = tuple(env.current_state())
    min_moves = hanoi_solver(state_tuple)

    steps = 0
    illegal_count = 0
    parse_failures = 0
    move_history: list[str] = []
    done = False
    solved = False

    while not done:
        current_state = tuple(env.current_state())

        # Build and execute prompt
        hist = move_history[-history_length:] if history_length > 0 else None
        prompt = prompt_fn(current_state, n_disks, hist)
        response = generate_response(model, tokenizer, prompt, temperature, max_new_tokens)

        # Parse move
        parsed = parse_move(response)
        if parsed is None:
            parse_failures += 1
            # Random fallback — treated as an attempted move
            action = int(np.random.randint(6))
            move_str = f"[PARSE_FAIL] random action {action}"
            logging.debug(f"Step {steps}: parse failure. Response: {response!r}")
        else:
            from_peg, to_peg = parsed
            action = pegs_to_action(from_peg, to_peg)
            move_str = f"Move from Peg {from_peg + 1} to Peg {to_peg + 1}"

        # Take environment step
        _, rwd, done, illegal_move = env.step(action)
        steps += 1

        if illegal_move:
            illegal_count += 1

        if rwd == 100:
            solved = True

        move_history.append(move_str)

    error = steps - min_moves
    illegal_rate = illegal_count / steps if steps > 0 else 0.0

    return {
        "error": error,
        "solved": solved,
        "illegal_count": illegal_count,
        "total_steps": steps,
        "illegal_rate": illegal_rate,
        "parse_failures": parse_failures,
        "min_optimal_moves": min_moves,
    }


# ---------------------------------------------------------------------------
# Multi-episode runner
# ---------------------------------------------------------------------------

def _aggregate(episode_results: list[dict]) -> dict:
    """Compute aggregate statistics from a list of episode results."""
    errors = np.array([r["error"] for r in episode_results], dtype=float)
    solve_rates = np.array([r["solved"] for r in episode_results], dtype=float)
    illegal_rates = np.array([r["illegal_rate"] for r in episode_results], dtype=float)

    n = len(episode_results)
    se = lambda x: float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    return {
        "mean_error": float(errors.mean()),
        "std_error": float(errors.std(ddof=1)) if n > 1 else 0.0,
        "se_error": se(errors),
        "solve_rate": float(solve_rates.mean()),
        "se_solve_rate": se(solve_rates),
        "mean_illegal_rate": float(illegal_rates.mean()),
        "se_illegal_rate": se(illegal_rates),
        "n_episodes": n,
        "episodes": episode_results,
    }


def run_evaluation(
    env: TowersOfHanoi,
    model,
    tokenizer,
    prompt_fn,
    start: int | None,
    episodes: int,
    temperature: float,
    max_new_tokens: int,
    history_length: int,
    n_disks: int,
    seed: int,
) -> dict:
    """Run ``episodes`` episodes and return aggregate statistics.

    Installs a SIGTERM handler so that partial results are returned
    when SLURM kills the job (sends SIGTERM before SIGKILL).
    """
    np.random.seed(seed)

    episode_results = []
    t0 = time.time()
    terminated = False

    def _sigterm_handler(signum, frame):
        nonlocal terminated
        logging.warning(
            f"Received SIGTERM after {len(episode_results)}/{episodes} episodes — "
            "will save partial results."
        )
        terminated = True

    prev_handler = signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        for ep in range(episodes):
            if terminated:
                break

            if start is not None:
                env.reset()
            else:
                env.random_reset()

            result = run_episode(
                env,
                model,
                tokenizer,
                prompt_fn,
                temperature,
                max_new_tokens,
                history_length,
                n_disks,
            )
            episode_results.append(result)

            elapsed = time.time() - t0
            avg_s = elapsed / (ep + 1)
            eta = avg_s * (episodes - ep - 1)
            logging.info(
                f"  ep {ep+1:3d}/{episodes}  "
                f"error={result['error']:4d}  "
                f"solved={result['solved']}  "
                f"illegal_rate={result['illegal_rate']:.2f}  "
                f"parse_fail={result['parse_failures']}  "
                f"ETA {eta/60:.1f} min"
            )
    finally:
        signal.signal(signal.SIGTERM, prev_handler)

    if not episode_results:
        raise RuntimeError("No episodes completed before termination.")

    if terminated:
        logging.warning(
            f"Partial run: {len(episode_results)}/{episodes} episodes completed."
        )

    return _aggregate(episode_results)


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def save_results(
    agg: dict,
    save_dir: str,
    label: str,
    prompting: str,
) -> None:
    """Save results in MuZero-compatible .pt format and as JSON.

    The .pt files use a dummy n_simulations value of 0 so that they can be
    read by load_accuracy() in plot_startingState_results.py.
    The bar-chart plotting code averages over the n_sim axis, so a single row
    gives the correct mean.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_stem = f"LLM_{label}_{prompting}"

    mean_e = agg["mean_error"]
    std_e = agg["se_error"]   # use SE for error bars (consistent with multi-run code)

    # actingAccuracy.pt  — shape (1, 2): [[n_sims, mean_error]]
    acc_tensor = torch.tensor([[0.0, mean_e]])
    acc_path = os.path.join(save_dir, file_stem + "_actingAccuracy.pt")
    torch.save(acc_tensor, acc_path)

    # actingAccuracy_error.pt  — shape (1, 3): [[n_sims, mean_error, se_error]]
    err_tensor = torch.tensor([[0.0, mean_e, std_e]])
    err_path = os.path.join(save_dir, file_stem + "_actingAccuracy_error.pt")
    torch.save(err_tensor, err_path)

    # Full JSON with solve rate, illegal move rate, and per-episode data
    json_path = os.path.join(save_dir, file_stem + "_results.json")
    with open(json_path, "w") as f:
        json.dump(agg, f, indent=2)

    logging.info(f"Saved: {acc_path}")
    logging.info(f"Saved: {err_path}")
    logging.info(f"Saved: {json_path}")
    logging.info(
        f"Summary  mean_error={mean_e:.2f}  se={std_e:.2f}"
        f"  solve_rate={agg['solve_rate']:.2%}"
        f"  illegal_rate={agg['mean_illegal_rate']:.2%}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_starting_state(env: TowersOfHanoi, start: int | None) -> str:
    """Mirror the logic from acting_ablations.py."""
    if start is not None:
        if start == 0:
            init_state = (2, 2, 0)   # 7 moves from goal — Far (ES)
            file_indx = "ES"
        elif start == 1:
            init_state = (0, 0, 2)   # 3 moves from goal — Moderate (MS)
            file_indx = "MS"
        elif start == 2:
            init_state = (1, 2, 2)   # 1 move from goal — Close (LS)
            file_indx = "LS"
        else:
            raise ValueError(f"--start must be 0, 1, or 2; got {start}")
        env.init_state_idx = env.states.index(init_state)
    else:
        file_indx = "RandState"
    return file_indx


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an instruction-tuned LLM on Tower of Hanoi"
    )
    # --- experiment identity ---
    parser.add_argument("--timestamp", required=True, help="Model timestamp directory (e.g. 1748875208)")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--model_label", required=True, help="Short label for output files (e.g. llama3_8b)")
    parser.add_argument("--prompting", choices=list(PROMPT_BUILDERS.keys()), default="zero_shot")
    # --- problem setup ---
    parser.add_argument("--start", type=int, default=None, choices=[0, 1, 2],
                        help="0=Far(ES), 1=Moderate(MS), 2=Close(LS); omit for random")
    parser.add_argument("--N", type=int, default=3, help="Number of disks")
    parser.add_argument("--max_steps", type=int, default=200, help="Max steps per episode")
    # --- evaluation ---
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history_length", type=int, default=0,
                        help="Number of previous moves to include in prompt (0=no history)")
    # --- LLM inference ---
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=300,
                        help="Max tokens generated per step (300 allows room for CoT)")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--model_cache_dir", default=None,
                        help="Directory to cache HuggingFace model weights")
    # --- layer interventions ---
    parser.add_argument("--ablate_layer", type=int, default=-1,
                        help="Bypass this transformer block (0-indexed; -1 = disabled)")
    parser.add_argument("--noise_scale", type=float, default=0.0,
                        help="Std dev of Gaussian noise injected into hidden states (0=disabled)")
    parser.add_argument("--noise_layer", type=int, default=-1,
                        help="Layer to inject noise at (-1 = all layers; only used if noise_scale > 0)")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        encoding="utf-8",
    )

    # Set up environment
    env = TowersOfHanoi(N=args.N, max_steps=args.max_steps)
    file_indx = get_starting_state(env, args.start)

    logging.info(
        f"LLM eval: model={args.model}  prompting={args.prompting}"
        f"  difficulty={file_indx}  episodes={args.episodes}"
    )

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, args.model_cache_dir, args.dtype)

    # Build intervention label suffix for file naming
    intervention_parts = []
    if args.ablate_layer >= 0:
        intervention_parts.append(f"ablateL{args.ablate_layer}")
    if args.noise_scale > 0.0:
        nl = "all" if args.noise_layer < 0 else str(args.noise_layer)
        intervention_parts.append(f"noiseS{args.noise_scale}_L{nl}")
    intervention_label = "_".join(intervention_parts)
    prompting_label = f"{args.prompting}_{intervention_label}" if intervention_label else args.prompting

    # Register layer intervention hooks (no-op if both args are at defaults)
    hook_handles = register_layer_hooks(
        model,
        ablate_layer=args.ablate_layer,
        noise_scale=args.noise_scale,
        noise_layer=args.noise_layer,
    )

    # Select prompt builder
    prompt_fn = PROMPT_BUILDERS[args.prompting]

    # Run evaluation
    logging.info("Starting evaluation...")
    try:
        agg = run_evaluation(
            env=env,
            model=model,
            tokenizer=tokenizer,
            prompt_fn=prompt_fn,
            start=args.start,
            episodes=args.episodes,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            history_length=args.history_length,
            n_disks=args.N,
            seed=args.seed,
        )
    finally:
        for h in hook_handles:
            h.remove()

    # Attach intervention metadata to JSON output
    if intervention_label:
        agg["intervention"] = {
            "ablate_layer": args.ablate_layer,
            "noise_scale": args.noise_scale,
            "noise_layer": args.noise_layer,
        }

    # Save results
    save_dir = os.path.join(
        PROJECT_ROOT, "stats", "Hanoi", args.timestamp, file_indx
    )
    save_results(agg, save_dir, args.model_label, prompting_label)
    logging.info("Done.")


if __name__ == "__main__":
    main()
