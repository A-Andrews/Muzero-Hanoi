"""
Statistical tests for the CCN two-page abstract.

Tests:
1. Difficulty effect: Close vs Moderate vs Far (within each LLM condition)
2. CoT effect: zero-shot vs CoT (within each model x difficulty)
3. Scaffolding effect: CoT vs CoT+h5+illfb (within each model x difficulty)
4. Cross-model: Llama CoT vs Qwen CoT (within each difficulty)
5. Solve rate: Fisher's exact test

Mann-Whitney U tests (sections 1-4) share a single Bonferroni correction family.
Fisher's exact tests (section 5) use a separate correction family.
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

# ── Config ──────────────────────────────────────────────────────────────────

TIMESTAMP = "1748875208"
BASE_DIR = Path(__file__).resolve().parent.parent / "stats" / "Hanoi" / TIMESTAMP

DIFFICULTIES = {
    "Close": "LS",
    "Moderate": "MS",
    "Far": "ES",
}

# Key conditions to test (must match filenames: LLM_{condition}_results.json)
CONDITIONS = [
    "llama3_8b_zero_shot",
    "llama3_8b_cot",
    "llama3_8b_cot_h5_illfb",
    "qwen25_7b_zero_shot",
    "qwen25_7b_cot",
    "qwen25_7b_cot_h5_illfb",
]


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_episodes(condition: str, diff_code: str) -> list[dict]:
    path = BASE_DIR / diff_code / f"LLM_{condition}_results.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path) as f:
        data = json.load(f)
    return data["episodes"]


def extract_errors(episodes: list[dict]) -> np.ndarray:
    return np.array([ep["error"] for ep in episodes], dtype=float)


def extract_solve(episodes: list[dict]) -> np.ndarray:
    return np.array([1.0 if ep["solved"] else 0.0 for ep in episodes])


def extract_illegal_rate(episodes: list[dict]) -> np.ndarray:
    return np.array([ep["illegal_rate"] for ep in episodes], dtype=float)


def bonferroni(p: float, n_tests: int) -> float:
    return min(p * n_tests, 1.0)


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


def effect_size_r(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation as effect size for Mann-Whitney U."""
    return 1 - (2 * u_stat) / (n1 * n2)


def print_mwu_result(tag: str, e_a: np.ndarray, e_b: np.ndarray,
                     label_a: str, label_b: str) -> tuple:
    """Print a Mann-Whitney U result inline and return the record for collection."""
    U, p = stats.mannwhitneyu(e_a, e_b, alternative="two-sided")
    r = effect_size_r(U, len(e_a), len(e_b))
    print(f"\n  {tag}:")
    print(f"    {label_a}: mean={e_a.mean():.2f}, median={np.median(e_a):.1f}, n={len(e_a)}")
    print(f"    {label_b}: mean={e_b.mean():.2f}, median={np.median(e_b):.1f}, n={len(e_b)}")
    print(f"    U={U:.0f}, p_raw={p:.2e}, r={r:.3f}")
    return (tag, U, p, r, len(e_a), len(e_b))


# ── Test runner ─────────────────────────────────────────────────────────────

def run_tests():
    # Load all data
    data = {}  # (condition, difficulty_name) -> episodes
    missing = []
    for cond in CONDITIONS:
        for diff_name, diff_code in DIFFICULTIES.items():
            try:
                data[(cond, diff_name)] = load_episodes(cond, diff_code)
            except FileNotFoundError as e:
                missing.append((cond, diff_name))
                print(f"  WARNING: missing {e}")

    if missing:
        print(f"\n  {len(missing)} condition(s) missing — tests involving them will be skipped.")

    all_pairwise = []  # collect (label, U, p_raw, r, n1, n2) for Bonferroni

    # ── 1. Difficulty effect (Kruskal-Wallis within each condition) ──────
    print("\n" + "=" * 80)
    print("1. DIFFICULTY EFFECT (Kruskal-Wallis on excess moves)")
    print("=" * 80)
    for cond in CONDITIONS:
        groups = []
        labels = []
        for diff_name in DIFFICULTIES:
            key = (cond, diff_name)
            if key in data:
                groups.append(extract_errors(data[key]))
                labels.append(diff_name)
        if len(groups) < 2:
            print(f"\n  {cond}: SKIPPED (fewer than 2 difficulties available)")
            continue
        H, p = stats.kruskal(*groups)
        print(f"\n  {cond}:")
        for lbl, g in zip(labels, groups):
            print(f"    {lbl:>10s}: mean={g.mean():.2f}, median={np.median(g):.1f}, n={len(g)}")
        print(f"    H={H:.2f}, p={p:.2e} {sig_stars(p)}")

        # Post-hoc pairwise
        for (l1, g1), (l2, g2) in combinations(zip(labels, groups), 2):
            tag = f"{cond} | {l1} vs {l2}"
            record = print_mwu_result(tag, g1, g2, l1, l2)
            all_pairwise.append(record)

    # ── 2. CoT effect (zero-shot vs CoT) ────────────────────────────────
    print("\n" + "=" * 80)
    print("2. CoT EFFECT (zero-shot vs CoT, Mann-Whitney U on excess moves)")
    print("=" * 80)
    for model in ["llama3_8b", "qwen25_7b"]:
        for diff_name in DIFFICULTIES:
            zs_key = (f"{model}_zero_shot", diff_name)
            cot_key = (f"{model}_cot", diff_name)
            if zs_key not in data or cot_key not in data:
                print(f"\n  {model} {diff_name}: SKIPPED (missing data)")
                continue
            e_zs = extract_errors(data[zs_key])
            e_cot = extract_errors(data[cot_key])
            tag = f"{model} {diff_name} | zero-shot vs CoT"
            record = print_mwu_result(tag, e_zs, e_cot, "zero-shot", "CoT")
            all_pairwise.append(record)

    # ── 3. Scaffolding effect (CoT vs CoT+h5+illfb) ─────────────────────
    print("\n" + "=" * 80)
    print("3. SCAFFOLDING EFFECT (CoT vs CoT+h5+illfb, Mann-Whitney U)")
    print("=" * 80)
    for model in ["llama3_8b", "qwen25_7b"]:
        for diff_name in DIFFICULTIES:
            cot_key = (f"{model}_cot", diff_name)
            fb_key = (f"{model}_cot_h5_illfb", diff_name)
            if cot_key not in data or fb_key not in data:
                print(f"\n  {model} {diff_name}: SKIPPED (missing data)")
                continue
            e_cot = extract_errors(data[cot_key])
            e_fb = extract_errors(data[fb_key])
            tag = f"{model} {diff_name} | CoT vs CoT+h5+illfb"
            record = print_mwu_result(tag, e_cot, e_fb, "CoT", "CoT+h5+illfb")
            all_pairwise.append(record)

    # ── 4. Cross-model (Llama CoT vs Qwen CoT) ──────────────────────────
    print("\n" + "=" * 80)
    print("4. CROSS-MODEL (Llama CoT vs Qwen CoT, Mann-Whitney U)")
    print("=" * 80)
    for diff_name in DIFFICULTIES:
        l_key = ("llama3_8b_cot", diff_name)
        q_key = ("qwen25_7b_cot", diff_name)
        if l_key not in data or q_key not in data:
            print(f"\n  {diff_name}: SKIPPED (missing data)")
            continue
        e_l = extract_errors(data[l_key])
        e_q = extract_errors(data[q_key])
        tag = f"{diff_name} | Llama CoT vs Qwen CoT"
        record = print_mwu_result(tag, e_l, e_q, "Llama CoT", "Qwen CoT")
        all_pairwise.append(record)

    # ── 5. Solve rate: Fisher's exact test ───────────────────────────────
    print("\n" + "=" * 80)
    print("5. SOLVE RATE (Fisher's exact test, separate Bonferroni family)")
    print("=" * 80)
    fisher_tests = [
        ("Llama zero-shot vs CoT", "llama3_8b_zero_shot", "llama3_8b_cot"),
        ("Llama CoT vs CoT+h5+illfb", "llama3_8b_cot", "llama3_8b_cot_h5_illfb"),
        ("Qwen zero-shot vs CoT", "qwen25_7b_zero_shot", "qwen25_7b_cot"),
        ("Qwen CoT vs CoT+h5+illfb", "qwen25_7b_cot", "qwen25_7b_cot_h5_illfb"),
        ("Llama CoT vs Qwen CoT", "llama3_8b_cot", "qwen25_7b_cot"),
    ]
    # Count actual tests (only those with available data)
    n_fisher = 0
    fisher_results = []
    for label, cond_a, cond_b in fisher_tests:
        for diff_name in DIFFICULTIES:
            key_a = (cond_a, diff_name)
            key_b = (cond_b, diff_name)
            if key_a in data and key_b in data:
                n_fisher += 1
    print(f"  (Bonferroni correction over {n_fisher} tests)\n")

    for label, cond_a, cond_b in fisher_tests:
        for diff_name in DIFFICULTIES:
            key_a = (cond_a, diff_name)
            key_b = (cond_b, diff_name)
            if key_a not in data or key_b not in data:
                print(f"  {label} | {diff_name}: SKIPPED (missing data)")
                continue
            s_a = extract_solve(data[key_a])
            s_b = extract_solve(data[key_b])
            table = np.array([
                [s_a.sum(), len(s_a) - s_a.sum()],
                [s_b.sum(), len(s_b) - s_b.sum()],
            ], dtype=int)
            _, p = stats.fisher_exact(table)
            p_corr = bonferroni(p, n_fisher)
            print(f"  {label} | {diff_name}:")
            print(f"    {cond_a}: {s_a.sum():.0f}/{len(s_a)} solved")
            print(f"    {cond_b}: {s_b.sum():.0f}/{len(s_b)} solved")
            print(f"    p={p:.2e}, p_corrected={p_corr:.2e} {sig_stars(p_corr)}")

    # ── Print all pairwise MWU results with Bonferroni ───────────────────
    n_tests = len(all_pairwise)
    print("\n" + "=" * 80)
    print(f"ALL MANN-WHITNEY U COMPARISONS (n={n_tests}, Bonferroni-corrected)")
    print("=" * 80)
    print(f"{'Test':<55s} {'U':>8s} {'p_raw':>10s} {'p_corr':>10s} {'r':>6s} {'sig':>5s}")
    print("-" * 100)
    for tag, U, p_raw, r, n1, n2 in all_pairwise:
        p_corr = bonferroni(p_raw, n_tests)
        print(f"  {tag:<53s} {U:8.0f} {p_raw:10.2e} {p_corr:10.2e} {r:6.3f} {sig_stars(p_corr):>5s}")

    # ── Summary for paper (looks up from collected results, no recomputation)
    print("\n" + "=" * 80)
    print("SUMMARY FOR PAPER")
    print("=" * 80)
    summary_keys = [
        "llama3_8b_cot | Close vs Far",
        "qwen25_7b_cot | Close vs Far",
        "qwen25_7b_cot | Close vs Moderate",
        "llama3_8b Moderate | zero-shot vs CoT",
        "llama3_8b Moderate | CoT vs CoT+h5+illfb",
        "qwen25_7b Moderate | CoT vs CoT+h5+illfb",
        "qwen25_7b Far | CoT vs CoT+h5+illfb",
        "Moderate | Llama CoT vs Qwen CoT",
    ]
    pairwise_lookup = {tag: (U, p_raw, r, n1, n2) for tag, U, p_raw, r, n1, n2 in all_pairwise}
    for key in summary_keys:
        if key in pairwise_lookup:
            U, p_raw, r, n1, n2 = pairwise_lookup[key]
            p_corr = bonferroni(p_raw, n_tests)
            print(f"  {key}: U={U:.0f}, p_corr={p_corr:.2e}, r={r:.3f} {sig_stars(p_corr)}")
        else:
            print(f"  {key}: NOT FOUND in pairwise results")


if __name__ == "__main__":
    run_tests()
