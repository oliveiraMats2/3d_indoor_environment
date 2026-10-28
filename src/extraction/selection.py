# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP - Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   IC - Institute of Computing            ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀   Computer Science Department              ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · IC · 2026                    ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝
"""Greedy frame selection by overlap and target adjustment to fit [min, max]."""
import numpy as np
from tqdm import tqdm

def selecionar(n, overlap, target, band=None, sharp=None):
    """Greedy: from the last kept frame, advance while overlap >= target; on the first drop below it,
    keep the best of (last sharp frame >= target, last frame >= target, current frame): inside band first,
    then sharp, then closest to target. No sharp mask = every frame sharp; no band = (target, target)."""

    lo, hi = band or (target, target)
    sharp = np.ones(n, bool) if sharp is None else np.asarray(sharp, bool)
    failed = getattr(overlap, "failed", set())  # pairs without an estimate (Sobreposicao.failed)
    kept, last, prev, prev_any = [0], 0, None, None
    progress = tqdm(range(1, n), desc=f"selection (target {target:.3f})", unit="frame", leave=False)
    for j in progress:

        while overlap(last, j) < target:
            if (last, j) in failed and prev_any is None and overlap(j - 1, j) == 0 and (j - 1, j) in failed:
                break  # j inside a textureless stretch (no estimate even vs j-1): skip it instead of keeping a blind 0

            def rank(c):
                o = overlap(last, c)
                return max(lo - o, o - hi, 0.0), not sharp[c], abs(o - target)

            new = min({c for c in (prev, prev_any, j) if c is not None}, key=rank)
            kept.append(new)
            last, prev, prev_any = new, None, None
            if new == j:
                break

        if last != j and (last, j) not in failed:
            prev_any = j
            prev = j if sharp[j] else prev

    return kept

def adjust(n, overlap, cfg, sharp=None):
    """Step the target by overlap_step until the count is in [min_frames, max_frames]."""
    target = cfg["overlap"]
    band = (cfg["overlap_min"], cfg["overlap_max"])
    kept = selecionar(n, overlap, target, band, sharp)

    for _ in range(cfg["max_adjust_iters"]):

        count = len(kept)
        if cfg["min_frames"] <= count <= cfg["max_frames"]:
            break

        step = -cfg["overlap_step"] if count > cfg["max_frames"] else cfg["overlap_step"]
        if not cfg["overlap_min"] <= target + step <= cfg["overlap_max"]:
            break

        target += step
        kept = selecionar(n, overlap, target, band, sharp)

    return kept, target

def subsample(kept, n_max):
    """Last resort: uniform subsample down to n_max frames."""
    last_index = len(kept) - 1
    positions = np.linspace(0, last_index, n_max)

    return [kept[int(i)] for i in positions]

def consecutive_overlap(kept, overlap):
    """Overlap estimate between each pair of consecutive selected frames."""

    return [overlap(a, b) for a, b in zip(kept[:-1], kept[1:])]
