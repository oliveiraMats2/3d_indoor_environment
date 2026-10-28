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

# PROMPT COMPLEMENT
# Stage: Stage 3, part 1 (Matching) - cross check (mutual nearest neighbour) after Lowe's ratio test.
# Goal: keep a Lowe match A->B only if the best reverse match B->A of its train descriptor points back to
#      the same query descriptor (same matcher name/norm as configured in the YAML block matching).
# In:  <run_dir>/matching/<det>/<A>__<B>/matches_lowe.npy (K, 3: queryIdx, trainIdx, distance) and
#      <run_dir>/detection/<det>/<frame>/descriptors.npy.
# Out: next to matches_lowe.npy: matches_ratio.npy (pure Lowe result, kept for the record) and
#      matches_cross.npy (Lowe + cross check); matches_lowe.npy is then overwritten with matches_cross so
#      every downstream reader (epipolar, matching metrics, figures, report) uses the cross-checked set with
#      aligned inlier masks. Counts n_lowe, n_cross, cross_keep_frac per pair (+ one "__all__" row per
#      detector) in <run_dir>/matching/cross_check.csv and <artifacts_dir>/cross_check.csv.
# YAML: filters.cross_check (bool, default false), read with .get; matching.{name, flann} pick the matcher.
# Rules: disabled -> no-op (matches_lowe.npy stays the pure Lowe result); re-running is idempotent (the
#      source is matches_ratio.npy when it already exists); missing inputs -> WARNING and return None.

# >>> BEGIN IMPLEMENTATION >>>
"""Cross check (mutual nearest neighbour) applied after Lowe's ratio test."""
import logging
from pathlib import Path

import numpy as np

from src import stage_io
from src.matching import matcher

log = logging.getLogger(__name__)

DEFAULTS = {"cross_check": False}
CSV_NAME = "cross_check.csv"


def enabled(filters_cfg: dict) -> bool:
    return bool({**DEFAULTS, **(filters_cfg or {})}.get("cross_check"))


def mutual(matches_lowe: np.ndarray, query: np.ndarray, train: np.ndarray, matching_cfg: dict) -> np.ndarray:
    """Rows of matches_lowe (K, 3) whose reverse best match train->query returns to the same query index."""

    matches = np.asarray(matches_lowe, dtype=np.float32).reshape(-1, 3)
    if len(matches) == 0 or len(query) == 0 or len(train) == 0:
        return matches[:0]
    reverse = matcher.run(matching_cfg, train).knnMatch(train, query, k=1)  # same name/norm, roles swapped
    best = np.full(len(train), -1, dtype=np.int64)
    for neighbors in reverse:
        if neighbors:
            best[neighbors[0].queryIdx] = neighbors[0].trainIdx
    keep = best[matches[:, 1].astype(np.int64)] == matches[:, 0].astype(np.int64)
    return matches[keep]


def _artifacts_dir(run_dir: Path) -> Path | None:
    """Default layout <ws>/results/<id> -> <ws>/artifacts/<id>; None when the layout is different."""

    if run_dir.parent.name == "results" and (run_dir.parent.parent / "artifacts").is_dir():
        return run_dir.parent.parent / "artifacts" / run_dir.name
    return None


def run(matching_cfg: dict, filters_cfg: dict, run_dir: str, artifacts_dir: str | None = None) -> Path | None:
    """Cross-check every saved pair; returns the CSV path (None when disabled or nothing to do)."""

    if not enabled(filters_cfg):
        return None
    root = Path(run_dir)
    matching_root = root / "matching"
    if not matching_root.is_dir():
        log.warning("cross_check: %s not found, skipped", matching_root)
        return None
    csv_paths = [matching_root / CSV_NAME]
    art = Path(artifacts_dir) if artifacts_dir else _artifacts_dir(root)
    if art is not None:
        csv_paths.append(art / CSV_NAME)

    for detector_dir in sorted(p for p in matching_root.iterdir() if p.is_dir()):
        detection = root / "detection" / detector_dir.name
        total_lowe = total_cross = 0
        for pair_dir in sorted(p for p in detector_dir.iterdir() if p.is_dir()):
            ratio_file, lowe_file = pair_dir / "matches_ratio.npy", pair_dir / "matches_lowe.npy"
            source = ratio_file if ratio_file.is_file() else lowe_file
            first, second = pair_dir.name.split("__", maxsplit=1)
            descriptors = [detection / name / "descriptors.npy" for name in (first, second)]
            if not source.is_file() or not all(p.is_file() for p in descriptors):
                log.warning("cross_check: %s/%s missing matches or descriptors, skipped", detector_dir.name, pair_dir.name)
                continue
            lowe = np.load(source)
            cross = mutual(lowe, np.load(descriptors[0]), np.load(descriptors[1]), matching_cfg)
            if source == lowe_file:
                np.save(ratio_file, lowe)
            np.save(pair_dir / "matches_cross.npy", cross)
            np.save(lowe_file, cross)  # downstream readers pick the cross-checked set
            total_lowe, total_cross = total_lowe + len(lowe), total_cross + len(cross)
            row = {"key": f"{detector_dir.name}/{pair_dir.name}", "detector": detector_dir.name,
                   "pair": pair_dir.name, "n_lowe": len(lowe), "n_cross": len(cross),
                   "cross_keep_frac": round(len(cross) / len(lowe), 4) if len(lowe) else None}
            for path in csv_paths:
                stage_io.upsert_csv(path, row, key="key")
        summary = {"key": f"{detector_dir.name}/__all__", "detector": detector_dir.name, "pair": "__all__",
                   "n_lowe": total_lowe, "n_cross": total_cross,
                   "cross_keep_frac": round(total_cross / total_lowe, 4) if total_lowe else None}
        for path in csv_paths:
            stage_io.upsert_csv(path, summary, key="key")
        stage_io.log_metrics(log, {f"cross_check_{detector_dir.name}_{k}": v for k, v in summary.items()
                                   if k in ("n_lowe", "n_cross", "cross_keep_frac")})
    return csv_paths[-1]
# <<< END IMPLEMENTATION <<<


if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    rng = np.random.default_rng(0)
    bf = {"name": "brute_force", "flann": {"checks": 50, "float_index": {"algorithm": 1, "trees": 5},
                                           "binary_index": {"algorithm": 6, "table_number": 6, "key_size": 12,
                                                            "multi_probe_level": 1}}}
    a = rng.random((50, 32)).astype(np.float32)
    b = a[::-1].copy() + 0.001  # b[49 - i] ~ a[i]
    b[0] = a[3] + 0.0005  # b[0] now points back to a[3]: the Lowe match (49, 0) is not mutual
    lowe = np.array([[i, 49 - i, 0.0] for i in range(50)], np.float32)
    kept = mutual(lowe, a, b, bf)
    assert len(kept) == 49 and 49 not in kept[:, 0], kept[:, 0]
    assert len(mutual(lowe, a, b, {**bf, "name": "flann"})) >= 45
    bits = rng.integers(0, 256, (40, 32), dtype=np.uint8)
    assert len(mutual(np.c_[np.arange(40), np.arange(40), np.zeros(40)], bits, bits.copy(), bf)) == 40
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        for name, d in (("f0", a), ("f1", b)):
            (run_dir / "detection" / "sift" / name).mkdir(parents=True)
            np.save(run_dir / "detection" / "sift" / name / "descriptors.npy", d)
        pair = run_dir / "matching" / "sift" / "f0__f1"
        pair.mkdir(parents=True)
        np.save(pair / "matches_lowe.npy", lowe)
        assert run(bf, {"lowe_ratio": 0.75}, str(run_dir)) is None  # disabled by default
        assert len(np.load(pair / "matches_lowe.npy")) == 50
        for _ in range(2):  # idempotent
            out = run(bf, {"cross_check": True}, str(run_dir), str(Path(tmp) / "art"))
            assert len(np.load(pair / "matches_ratio.npy")) == 50
            assert len(np.load(pair / "matches_lowe.npy")) == len(np.load(pair / "matches_cross.npy")) == 49
        rows = stage_io.read_csv(out)
        assert len(rows) == 2 and rows[-1]["pair"] == "__all__" and rows[-1]["n_cross"] == "49", rows
    print("cross_check self-test OK")
