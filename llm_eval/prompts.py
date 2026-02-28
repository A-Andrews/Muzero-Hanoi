"""Prompt templates for LLM Tower of Hanoi evaluation.

Two prompting strategies:
  zero_shot  — describe state, ask for next move directly.
  cot        — same description, but elicit chain-of-thought reasoning before
               the move, to give the model a chance to plan explicitly.
"""

import itertools

# All 6 possible moves (0-indexed pegs): action index → (from_peg, to_peg)
MOVES = list(itertools.permutations([0, 1, 2], 2))
# [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]

PEG_NAMES = {0: "Peg 1 (left)", 1: "Peg 2 (middle)", 2: "Peg 3 (right)"}

RULES = """\
Rules:
  • Only the topmost disk on each peg can be moved.
  • A larger disk may never be placed on top of a smaller disk.
  • Exactly one disk is moved per turn."""

GOAL_TEXT = "Goal: Move all disks to Peg 3 (right)."

ANSWER_FORMAT = 'Give your answer on a single line in exactly this format: "Move from Peg X to Peg Y"'


def state_to_text(state: tuple, n_disks: int = 3) -> str:
    """Render a state tuple as a human-readable peg description.

    state[i] is the peg (0=left, 1=middle, 2=right) for disk i+1,
    where disk 1 is the smallest and disk n_disks is the largest.
    Smaller-indexed disks always sit on top of larger-indexed disks.

    Example for state (2, 2, 0) with n_disks=3:
        Peg 1 (left)  : [Disk 3 (largest)] — Disk 3 accessible
        Peg 2 (middle): [empty]
        Peg 3 (right) : [Disk 1 (smallest) → Disk 2] top→bottom — Disk 1 accessible
    """
    pegs: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for disk_idx in range(n_disks):
        pegs[state[disk_idx]].append(disk_idx + 1)  # 1-indexed disk number

    lines = []
    for p in range(3):
        disks = sorted(pegs[p])  # ascending = top-to-bottom (smallest on top)
        if disks:
            top = disks[0]
            if len(disks) == 1:
                size_tag = (
                    " (smallest)" if top == 1 else
                    f" (largest)" if top == n_disks else ""
                )
                lines.append(
                    f"  {PEG_NAMES[p]}: [Disk {top}{size_tag}]"
                    f" — Disk {top} is accessible"
                )
            else:
                parts = []
                for d in disks:
                    tag = (
                        " (smallest)" if d == 1 else
                        " (largest)" if d == n_disks else ""
                    )
                    parts.append(f"Disk {d}{tag}")
                stack = " → ".join(parts)
                lines.append(
                    f"  {PEG_NAMES[p]}: [{stack}] (top→bottom)"
                    f" — Disk {top} is accessible"
                )
        else:
            lines.append(f"  {PEG_NAMES[p]}: [empty]")

    return "\n".join(lines)


def _header(n_disks: int) -> str:
    return (
        f"Tower of Hanoi — {n_disks} disks, 3 pegs.\n"
        f"{GOAL_TEXT}\n"
        f"{RULES}"
    )


def build_zero_shot_prompt(
    state: tuple,
    n_disks: int = 3,
    history: list[str] | None = None,
) -> str:
    """Zero-shot prompt: show state, ask for next move.

    Args:
        state:   current env state tuple.
        n_disks: number of disks.
        history: optional list of move strings from previous steps,
                 e.g. ["Move from Peg 1 to Peg 3", ...].
    """
    state_text = state_to_text(state, n_disks)
    parts = [_header(n_disks), "", "Current state:", state_text]

    if history:
        parts += ["", "Previous moves (most recent last):"]
        for m in history[-5:]:          # show at most last 5
            parts.append(f"  {m}")

    parts += ["", ANSWER_FORMAT]
    return "\n".join(parts)


def build_cot_prompt(
    state: tuple,
    n_disks: int = 3,
    history: list[str] | None = None,
) -> str:
    """Chain-of-thought prompt: ask model to reason before giving the move.

    The CoT structure nudges the model to:
      1. Enumerate accessible disks.
      2. List legal moves.
      3. Reason which move brings the puzzle closest to the goal.
    """
    state_text = state_to_text(state, n_disks)
    parts = [_header(n_disks), "", "Current state:", state_text]

    if history:
        parts += ["", "Previous moves (most recent last):"]
        for m in history[-5:]:
            parts.append(f"  {m}")

    parts += [
        "",
        "Think step by step before answering:",
        "  1. Identify which disk is on top of each non-empty peg (these are the only moveable disks).",
        "  2. List all currently legal moves (moves that do not place a larger disk on a smaller one).",
        "  3. Among the legal moves, choose the one that best progresses toward the goal.",
        "  4. State your chosen move.",
        "",
        ANSWER_FORMAT,
    ]
    return "\n".join(parts)


# Map prompting strategy name → builder function
PROMPT_BUILDERS = {
    "zero_shot": build_zero_shot_prompt,
    "cot": build_cot_prompt,
}
