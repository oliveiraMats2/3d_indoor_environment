"""room_fill stage on a synthetic kitchen: outline fit (scale, rotation, residual) and fill only where uncovered."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
import yaml

from src import stage_io
from src.constants import Config
from src.mesh import room_fill as rf

PATCH = (120.0, 170.0, 150.0, 200.0)  # floor rectangle (cm) covered by the selected mesh, inside the floor hole
WALL_HOLE = ((0, 120.0, 150.0), 40.0)  # (wall 0, u cm, z cm), radius cm


def fill_faces(run_dir: Path, truth: dict) -> tuple[np.ndarray, np.ndarray]:
    """Face centroids of room_fill.ply in the true outline frame (cm) and their unit normals (same frame)."""
    m = o3d.io.read_triangle_mesh(str(run_dir / "mesh/room_fill/room_fill.ply"))
    V, F = np.asarray(m.vertices), np.asarray(m.triangles)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]) @ truth["Rw"]   # floor frame
    n[:, :2] = n[:, :2] @ rf.rot2(truth["theta"])                                     # outline frame
    return rf.to_room_cm(V[F].mean(axis=1), truth), n / np.linalg.norm(n, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def walls_run(tmp_path_factory):
    """Random similarity + rigid motion, no labels (height band), floor/ceiling/walls with holes, a mesh patch."""
    tmp = tmp_path_factory.mktemp("room")
    run_dir, truth = rf.synthetic_run(tmp, np.random.default_rng(1), s_true=0.0377, theta_deg=-121.0,
                                      t_true=(-2.3, 4.1), wall_hole=WALL_HOLE, mesh_patch=PATCH)
    art = tmp / "art"
    out = rf.run(rf.default_cfg(surfaces=["floor", "ceiling", "walls"]), str(run_dir), str(art))
    return run_dir, art, truth, out, stage_io.read_csv(art / "room_fill.csv")[0]


def test_fit_recovers_similarity(walls_run):
    _, _, truth, out, row = walls_run
    assert out is not None and out.name == "room_scene.ply"
    assert abs(float(row["s_units_per_cm"]) / truth["s"] - 1) < 0.02, row
    assert abs(float(row["cm_per_unit"]) * truth["s"] - 1) < 0.02, row
    assert abs((float(row["rotation_deg"]) + 121.0 + 180) % 360 - 180) < 2.0, row
    assert row["mirrored"] == "False" and row["wall_points_from"] == "height_band", row
    assert float(row["fit_residual_median_cm"]) < 2.0 and float(row["fit_residual_p90_cm"]) < 5.0, row
    assert float(row["fit_inlier_frac"]) > 0.9, row
    assert abs(float(row["ceiling_height_cm"]) - 250.0) < 5.0 and row["ceiling_from"] == "p99", row
    assert abs(float(row["polygon_area_m2"]) - 11.26) < 0.01, row


def test_only_holes_are_filled(walls_run):
    run_dir, _, truth, _, row = walls_run
    c, n = fill_faces(run_dir, truth)
    floor, ceiling, walls = c[n[:, 2] > 0.99], c[n[:, 2] < -0.99], c[np.abs(n[:, 2]) < 0.01]
    assert len(floor) + len(ceiling) + len(walls) == len(c)                       # floor up, ceiling down, walls vertical
    assert np.abs(floor[:, 2]).max() < 1e-6 and np.abs(ceiling[:, 2] - truth["height_cm"]).max() < 5
    (hc, hr) = truth["hole"]
    # floor: fill only inside the hole, never on the mesh patch, and nearly all of the uncovered disc
    assert len(floor) and np.linalg.norm(floor[:, :2] - hc, axis=1).max() < hr + 8
    x0, y0, x1, y1 = PATCH
    on_patch = (floor[:, 0] > x0) & (floor[:, 0] < x1) & (floor[:, 1] > y0) & (floor[:, 1] < y1)
    assert not on_patch.any()
    expected = (np.pi * (hr - 5) ** 2 - ((x1 - x0) ** 2 + 4 * (x1 - x0) * 5 + np.pi * 25)) / 1e4
    assert abs(float(row["floor_filled_m2"]) - expected) < 0.15 * expected, row
    # ceiling: points only where y < ceiling_y_max -> fill only on the other side
    assert len(ceiling) and ceiling[:, 1].min() > truth["ceiling_y_max"] - 8
    assert 0.35 < float(row["ceiling_filled_frac"]) < 0.65, row
    # walls: only the disc hole on wall 0 (y = 0 edge of the outline), radius 40 cm around u=120, z=150
    (_, u, z), r = WALL_HOLE
    assert len(walls) and np.abs(walls[:, 1]).max() < 3
    assert np.hypot(walls[:, 0] - u, walls[:, 2] - z).max() < r + 8
    assert float(row["walls_filled_frac"]) < 0.02, row
    assert int(row["n_faces"]) == len(c) and int(row["n_faces"]) > 0


def test_normals_colours_and_outputs(walls_run):
    run_dir, _, truth, _, row = walls_run
    m = o3d.io.read_triangle_mesh(str(run_dir / "mesh/room_fill/room_fill.ply"))
    V, F = np.asarray(m.vertices), np.asarray(m.triangles)
    c, n = fill_faces(run_dir, truth)
    walls = np.abs(n[:, 2]) < 0.01
    assert np.allclose(n[walls][np.abs(c[walls, 1]) < 3, 1], 1.0, atol=0.01)       # wall y = 0 faces inside (+y)
    floor_v = np.abs(rf.to_room_cm(V, truth)[:, 2]) < 1e-6
    assert m.has_vertex_colors()
    assert np.allclose(np.asarray(m.vertex_colors)[floor_v][0], np.array([200, 180, 150]) / 255, atol=0.02)
    scene = o3d.io.read_triangle_mesh(str(run_dir / "mesh/room_fill/room_scene.ply"))
    assert len(scene.triangles) == len(F) + 2 * 40 * 40                            # mesh patch + fill
    for name in ("room_fill.ply", "room_scene.ply"):
        assert (run_dir / "exports/meshlab" / name).stat().st_size > 0
    T_room = np.load(run_dir / "mesh/room_fill/T_room.npy")
    poly = np.load(run_dir / "mesh/room_fill/footprint_cm.npy")
    corners = np.c_[poly, np.zeros(len(poly)), np.ones(len(poly))] @ T_room.T
    back = rf.to_room_cm(corners[:, :3], truth)[:, :2]
    assert rf.segment_distance(back, truth["poly"])[0].max() < 3.0                  # outline lands on the true one


def test_labels_reversed_and_mirrored_outline(tmp_path):
    run_dir, truth = rf.synthetic_run(tmp_path, np.random.default_rng(2), source="sparse", labels=True,
                                      ceiling_y_max=1e9, hole=((100.0, 100.0), 40.0))
    art = tmp_path / "art"
    footprints = {"reversed": rf.KITCHEN_CM[::-1], "mirrored": rf.KITCHEN_CM * [-1, 1]}
    for name, footprint in footprints.items():
        cfg = rf.default_cfg(footprint_cm=footprint.tolist(), source="sparse", ceiling_height_cm=250.0)
        assert rf.run(cfg, str(run_dir), str(art)) is not None
        row = stage_io.read_csv(art / "room_fill.csv")[0]
        assert row["wall_points_from"] == "labels" and row["ceiling_from"] == "config", row
        assert row["mirrored"] == str(name == "mirrored"), row
        assert abs(float(row["s_units_per_cm"]) / truth["s"] - 1) < 0.02, row
        assert float(row["fit_residual_median_cm"]) < 2.0, row
        assert float(row["ceiling_filled_m2"]) == 0.0 and float(row["floor_filled_m2"]) > 0.3, row
        assert row["mesh_input"] == "", row                                       # no mesh: cloud only


def test_missing_inputs_return_none(tmp_path):
    run_dir, _ = rf.synthetic_run(tmp_path, np.random.default_rng(3))
    art = tmp_path / "art"
    assert rf.run({"room_fill": {"enabled": False}}, str(run_dir), str(art)) is None
    assert rf.run(rf.default_cfg(footprint_cm=[]), str(run_dir), str(art)) is None
    assert rf.run(rf.default_cfg(footprint_cm=[[0, 0], [1, 1]]), str(run_dir), str(art)) is None
    shutil.rmtree(run_dir / "clean")
    assert rf.run(rf.default_cfg(), str(run_dir), str(art)) is None
    shutil.rmtree(run_dir / "planes")
    assert rf.run(rf.default_cfg(), str(run_dir), str(art)) is None
    assert not (art / "room_fill.csv").exists()


def test_configs_validate():
    root = Path(__file__).resolve().parents[1]
    assert Config().room_fill.enabled is False and Config().room_fill.footprint_cm == []
    for path in sorted((root / "configs").glob("*.yaml")):
        Config.model_validate(yaml.safe_load(path.read_text()) or {})
    kitchen = Config.model_validate(yaml.safe_load((root / "configs/kitchen_my_home_glomap.yaml").read_text()))
    assert kitchen.room_fill.enabled and kitchen.room_fill.footprint_cm[3] == [81.1, 424.5]
    assert kitchen.room_fill.surfaces == ["floor", "ceiling"] and kitchen.room_fill.ceiling_height_cm is None
    with pytest.raises(Exception):
        Config.model_validate({"room_fill": {"footprint_cm": [[0, 0, 0]]}})
