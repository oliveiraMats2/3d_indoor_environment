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
# Stage: liepa (Liepa 2003, "Filling Holes in Meshes", SGP'03), after close_holes, before mesh. Optional
#   post-processing, off by default: liepa.enabled.
# Goal: fill each hole of the chosen mesh with a patch that matches the surroundings, touching nothing else:
#   1. triangulation: O(n^3) minimum-weight triangulation of the boundary loop, weight = (largest dihedral
#      angle, area) compared lexicographically; chords that already are mesh edges are forbidden;
#   2. refinement: sigma(v) = mean length of the mesh edges at boundary vertex v; a patch triangle is split at
#      its centroid c when alpha*|c - v| > sigma(c) and > sigma(v) for its 3 vertices (alpha = density),
#      edges are relaxed (Delaunay swap) after every split pass, until no triangle splits;
#   3. fairing: new vertices solve L^k x = 0 (uniform umbrella; k=2 thin plate as in the paper, k=1
#      membrane, 0 = off) with every original vertex fixed; colours of new vertices: harmonic (k=1).
# In:  liepa.input poisson -> mesh/poisson/pre/mesh.ply | close_holes -> mesh/close_holes/pre_pre/mesh.ply
#      (missing -> WARNING, falls back to poisson/pre).
# Out: mesh/liepa/{mesh.ply, vertices.npy, faces.npy, colors.npy}, exports/meshlab/liepa_scene.ply,
#      artifacts liepa.csv (holes found/filled/skipped, new vertices/faces, boundary edges before/after, t_s).
# YAML: liepa: {enabled, input, max_hole_edges, density, fairing, max_refine_iters}
# Rules: degenerate/duplicated faces and faces on edges shared by > 2 faces are removed first (MeshLab
#   "Remove Faces"), so the border is made of closed loops; at a non-manifold boundary vertex the walk
#   follows the face fan. Loops longer than max_hole_edges (e.g. the open border of the scene) and walks that
#   cannot close are skipped, counted, never filled. A singular fairing system -> WARNING, refined
#   patch kept unfaired.
# >>> BEGIN IMPLEMENTATION >>>
"""Liepa (2003) hole filling: minimum-weight triangulation, density refinement and fairing of every hole."""
import logging, shutil, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src import stage_io

log = logging.getLogger(__name__)

COLUMNS = ("input", "faces_removed_non_manifold", "holes_found", "holes_filled", "holes_too_large", "holes_nonmanifold", "holes_failed",
           "largest_filled_edges", "new_vertices", "new_faces", "boundary_edges_before", "boundary_edges_after",
           "non_manifold_edges_after", "faired", "n_vertices", "n_faces", "t_s")

def edge_counts(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Half-edges (3F, 2) in face order, and per half-edge the number of faces sharing its undirected edge."""

    he = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    _, inv, cnt = np.unique(np.sort(he, axis=1), axis=0, return_inverse=True, return_counts=True)

    return he, cnt[inv.ravel()], cnt

def drop_non_manifold_faces(faces: np.ndarray) -> tuple[np.ndarray, int]:
    """Removes every face on an edge shared by more than 2 faces (MeshLab "Remove Faces" repair): the
    boundary then is made of closed loops, and the holes it opens are filled like the others."""

    _, count, _ = edge_counts(faces)
    bad = np.unique(np.flatnonzero(count > 2) // 3)

    return np.delete(faces, bad, axis=0), len(bad)

def boundary_loops(faces: np.ndarray, n_vertices: int) -> tuple[list[list[int]], int]:
    """Closed loops of boundary half-edges, walked in mesh orientation (edge-manifold input). At a
    non-manifold vertex (several boundary half-edges out) the next one is found by rotating through the
    face fan of the incoming half-edge. Walks that cannot close or revisit a vertex are dropped and counted."""

    he, count, _ = edge_counts(faces)
    b = he[count == 1]
    bad = np.bincount(b[:, 0], minlength=n_vertices) > 1
    nxt = dict(zip(b[:, 0].tolist(), b[:, 1].tolist()))
    near = bad[he[:, 0]] | bad[he[:, 1]]
    he_face = dict(zip(map(tuple, he[near].tolist()), (np.flatnonzero(near) // 3).tolist()))

    def leaving(u, v):
        """Boundary half-edge out of the bad vertex v in the fan of boundary half-edge u->v."""

        f = he_face[(u, v)]
        for _ in range(len(he_face)):
            t = faces[f].tolist()
            x = t[(t.index(v) + 1) % 3]  # half-edge v->x of face f
            if (x, v) not in he_face:
                return x
            f = he_face[(x, v)]

        return None

    visited, loops, dropped = set(), [], 0
    for start, first in b.tolist():
        if (start, first) in visited:
            continue
        loop, seen, u, v, ok = [start], {start}, start, first, True
        visited.add((start, first))
        while v != start:
            w = leaving(u, v) if bad[v] else nxt.get(v)
            if w is None or (v, w) in visited or v in seen:
                ok = False
                break
            visited.add((v, w))
            loop.append(v)
            seen.add(v)
            u, v = v, w
        if ok and len(loop) >= 3:
            loops.append(loop)
        else:
            dropped += 1

    return loops, dropped

def unit(v: np.ndarray) -> np.ndarray:

    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

def angle(n1: np.ndarray, n2: np.ndarray) -> np.ndarray:

    return np.arccos(np.clip(np.sum(n1 * n2, axis=-1), -1.0, 1.0))

def triangulate(Q: np.ndarray, edge_normal: np.ndarray, chord_exists: np.ndarray) -> list[tuple[int, int, int]] | None:
    """Liepa section 3: dynamic program over the sub-polygons i..k of the polygon Q (n, 3), in patch
    orientation. edge_normal[i]: unit normal of the mesh face on polygon edge (i, i+1 mod n). chord_exists[i, k]:
    the chord i-k already is a mesh edge (forbidden). Returns triangles as polygon indices, or None."""

    n = len(Q)
    A = np.full((n, n), np.inf)  # largest dihedral angle of the best triangulation of i..k
    S = np.zeros((n, n))         # its area
    O = np.full((n, n), -1)      # its apex
    N = np.zeros((n, n, 3))      # normal of the triangle on chord i-k
    idx = np.arange(n - 1)
    A[idx, idx + 1] = 0.0
    N[idx, idx + 1] = edge_normal[:-1]
    for d in range(2, n):
        for i in range(n - d):
            k = i + d
            closing = i == 0 and k == n - 1
            if chord_exists[i, k] and not closing:
                continue
            m = np.arange(i + 1, k)
            m = m[np.isfinite(A[i, m]) & np.isfinite(A[m, k])]
            if len(m) == 0:
                continue
            c = np.cross(Q[m] - Q[i], Q[k] - Q[i])
            nt = unit(c)
            dihedral = np.maximum(angle(nt, N[i, m]), angle(nt, N[m, k]))
            if closing:
                dihedral = np.maximum(dihedral, angle(nt, edge_normal[n - 1]))
            a = np.maximum(np.maximum(A[i, m], A[m, k]), dihedral)
            s = S[i, m] + S[m, k] + 0.5 * np.linalg.norm(c, axis=1)
            j = np.lexsort((s, a))[0]
            A[i, k], S[i, k], O[i, k], N[i, k] = a[j], s[j], m[j], nt[j]

    if O[0, n - 1] < 0:
        return None

    tris, stack = [], [(0, n - 1)]
    while stack:
        i, k = stack.pop()
        m = int(O[i, k])
        tris.append((i, m, k))
        stack += [(a, b) for a, b in ((i, m), (m, k)) if b - a > 1]

    return tris

class Patch:
    """Triangles of one hole patch with an edge -> faces map, for splits and edge swaps."""

    def __init__(self, tris, pos: dict, sigma: dict, has_mesh_edge):

        self.faces, self.edges, self.next_fid = {}, defaultdict(set), 0
        self.pos, self.sigma, self.has_mesh_edge = pos, sigma, has_mesh_edge
        for t in tris:
            self.add(t)

    @staticmethod
    def keys(t):

        a, b, c = t
        return [(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(c, a), max(c, a))]

    def add(self, t) -> int:

        fid = self.next_fid
        self.next_fid += 1
        self.faces[fid] = tuple(t)
        for e in self.keys(t):
            self.edges[e].add(fid)

        return fid

    def remove(self, fid: int) -> None:

        for e in self.keys(self.faces.pop(fid)):
            self.edges[e].discard(fid)
            if not self.edges[e]:
                del self.edges[e]

    def relax(self, e) -> bool:
        """Swaps the interior edge e when it is not locally Delaunay (opposite angles sum > pi)."""

        fs = self.edges.get(e)
        if not fs or len(fs) != 2:
            return False  # hole border: the other face belongs to the mesh
        f, g = fs
        tf = self.faces[f]
        r = next(r for r in range(3) if {tf[r], tf[(r + 1) % 3]} == set(e))
        x, y, c = tf[r], tf[(r + 1) % 3], tf[(r + 2) % 3]
        d = next(v for v in self.faces[g] if v not in e)
        if c == d or (min(c, d), max(c, d)) in self.edges or self.has_mesh_edge(c, d):
            return False
        P = self.pos
        opposite = sum(float(angle(unit(P[x] - P[o]), unit(P[y] - P[o]))) for o in (c, d))
        if opposite <= np.pi + 1e-9:
            return False
        self.remove(f)
        self.remove(g)
        self.add((x, d, c))
        self.add((d, y, c))

        return True

    def relax_all(self, max_rounds: int = 50) -> None:

        for _ in range(max_rounds):
            if not sum(self.relax(e) for e in list(self.edges) if e in self.edges):
                return

    def refine(self, alpha: float, next_vid: int, max_iters: int) -> int:
        """Liepa section 4; returns the next free vertex id."""

        P, sig = self.pos, self.sigma
        for _ in range(max_iters):
            splits = 0
            for fid in list(self.faces):
                if fid not in self.faces:
                    continue  # swapped away by a relax of this pass
                t = self.faces[fid]
                vc = (P[t[0]] + P[t[1]] + P[t[2]]) / 3.0
                sc = (sig[t[0]] + sig[t[1]] + sig[t[2]]) / 3.0
                dist = [alpha * np.linalg.norm(vc - P[v]) for v in t]
                if not all(dv > sc and dv > sig[v] for dv, v in zip(dist, t)):
                    continue
                v = next_vid
                next_vid += 1
                P[v], sig[v] = vc, sc
                self.remove(fid)
                a, b, c = t
                for nt in ((a, b, v), (b, c, v), (c, a, v)):
                    self.add(nt)
                for e in self.keys(t):
                    self.relax(e)
                splits += 1
            if not splits:
                break
            self.relax_all()

        return next_vid

def fair(V: np.ndarray, F: np.ndarray, U: np.ndarray, order: int, values: np.ndarray) -> np.ndarray:
    """Solves L^order x = 0 at the vertices U (uniform umbrella L = W - D), every other vertex fixed."""

    e = F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    e = np.unique(np.sort(e, axis=1), axis=0)
    n = len(V)
    W = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    L = W - sp.diags(np.asarray(W.sum(axis=1)).ravel())
    M = L[U]
    for _ in range(order - 1):
        M = M @ L
    M = M.tocsc()
    fixed = values.copy()
    fixed[U] = 0.0
    rhs = -(M @ fixed)
    solve = spla.factorized(M[:, U].tocsc())
    out = values.copy()
    out[U] = np.column_stack([solve(np.ascontiguousarray(rhs[:, j])) for j in range(values.shape[1])])
    if not np.isfinite(out[U]).all():
        raise np.linalg.LinAlgError("non-finite fairing solution")

    return out

def fill(V: np.ndarray, F: np.ndarray, C: np.ndarray, lcfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """All holes of (V, F, C) up to max_hole_edges; returns the new arrays and the CSV metrics."""

    nv = len(V)
    he, count, ucount = edge_counts(F)
    und = np.unique(np.sort(he, axis=1), axis=0)
    mesh_keys = np.sort(und[:, 0].astype(np.int64) * nv + und[:, 1])

    patch_adj = defaultdict(set)  # edges of the patches already built (two holes may share a vertex)

    def has_mesh_edge(a, b) -> bool:

        if b in patch_adj.get(a, ()):
            return True
        if a >= nv or b >= nv:
            return False
        key = min(a, b) * nv + max(a, b)
        i = np.searchsorted(mesh_keys, key)
        return i < len(mesh_keys) and mesh_keys[i] == key

    face_normal = unit(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]))
    boundary = count == 1
    he_face = dict(zip(map(tuple, he[boundary].tolist()), (np.flatnonzero(boundary) // 3).tolist()))
    lengths = np.linalg.norm(V[und[:, 0]] - V[und[:, 1]], axis=1)
    sigma_all = (np.bincount(und[:, 0], lengths, nv) + np.bincount(und[:, 1], lengths, nv)) / \
        np.maximum(np.bincount(und.ravel(), minlength=nv), 1)

    loops, dropped = boundary_loops(F, nv)
    stats = {"holes_found": len(loops) + dropped, "holes_nonmanifold": dropped, "holes_too_large": 0,
             "holes_failed": 0, "holes_filled": 0, "largest_filled_edges": 0,
             "boundary_edges_before": int((ucount == 1).sum())}
    new_pos, new_faces, next_vid = {}, [], nv
    for loop in loops:
        n = len(loop)
        if n > lcfg["max_hole_edges"]:
            stats["holes_too_large"] += 1
            continue
        poly = loop[::-1]  # patch orientation: every border half-edge reversed
        Q = V[poly]
        edge_normal = np.array([face_normal[he_face[(poly[(i + 1) % n], poly[i])]] for i in range(n)])
        keys = np.minimum.outer(poly, poly).astype(np.int64) * nv + np.maximum.outer(poly, poly)
        pos_in = np.clip(np.searchsorted(mesh_keys, keys), 0, len(mesh_keys) - 1)
        chord_exists = mesh_keys[pos_in] == keys
        where = {v: i for i, v in enumerate(poly)}
        for i, a in enumerate(poly):
            for b in patch_adj.get(a, ()):
                if b in where:
                    chord_exists[i, where[b]] = True
        tris = triangulate(Q, edge_normal, chord_exists)
        if tris is None:
            stats["holes_failed"] += 1
            log.warning("liepa: no valid triangulation for a hole of %d edges (every chord already a mesh edge)", n)
            continue
        pos = {v: V[v] for v in poly}
        sigma = {v: sigma_all[v] for v in poly}
        patch = Patch([(poly[a], poly[b], poly[c]) for a, b, c in tris], pos, sigma, has_mesh_edge)
        patch.relax_all()
        next_vid = patch.refine(lcfg["density"], next_vid, lcfg["max_refine_iters"])
        new_pos.update({v: p for v, p in pos.items() if v >= nv})
        new_faces += list(patch.faces.values())
        for a, b in patch.edges:
            patch_adj[a].add(b)
            patch_adj[b].add(a)
        stats["holes_filled"] += 1
        stats["largest_filled_edges"] = max(stats["largest_filled_edges"], n)

    U = np.arange(nv, next_vid)
    V2 = np.vstack([V, np.array([new_pos[v] for v in U]).reshape(-1, 3)])
    F2 = np.vstack([F, np.array(new_faces, dtype=F.dtype).reshape(-1, 3)])
    C2 = np.vstack([C, np.zeros((len(U), 3))])
    stats["faired"] = False
    if len(U):
        try:
            C2 = fair(V2, F2, U, 1, C2)
            if lcfg["fairing"]:
                V2 = fair(V2, F2, U, lcfg["fairing"], V2)
                stats["faired"] = True
        except (RuntimeError, np.linalg.LinAlgError) as exc:
            log.warning("liepa: fairing failed (%s); refined patches kept unfaired", exc)
    _, _, ucount2 = edge_counts(F2)
    stats.update(new_vertices=len(U), new_faces=len(new_faces), boundary_edges_after=int((ucount2 == 1).sum()),
                 non_manifold_edges_after=int((ucount2 > 2).sum()))

    return V2, F2, np.clip(C2, 0.0, 1.0), stats

def pick_input(cfg: dict, run_dir: Path) -> Path | None:
    """liepa.input close_holes -> close_holes/pre_pre (else poisson/pre, WARNING); poisson -> poisson/pre."""

    pre = run_dir / "mesh" / "poisson" / "pre" / "mesh.ply"
    if cfg["liepa"]["input"] == "close_holes":
        path = run_dir / "mesh" / "close_holes" / "pre_pre" / "mesh.ply"
        if path.exists():
            return path
        log.warning("liepa.input=close_holes but %s missing: falling back to %s", path, pre)

    return pre if pre.exists() else None

def run(cfg: dict, run_dir: str, artifacts_dir: str) -> dict | None:

    lcfg = cfg["liepa"]
    if not lcfg["enabled"]:
        log.info("liepa disabled")
        return None

    t0 = time.perf_counter()
    run_dir, artifacts_dir = Path(run_dir), Path(artifacts_dir)
    src = pick_input(cfg, run_dir)
    if src is None:
        log.warning("liepa: no input mesh under %s/mesh, stage skipped", run_dir)
        return None

    m = o3d.io.read_triangle_mesh(str(src))
    V, F = np.asarray(m.vertices, dtype=np.float64), np.asarray(m.triangles, dtype=np.int64)
    C = np.asarray(m.vertex_colors) if m.has_vertex_colors() else np.full((len(V), 3), 0.7)
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])]
    F = F[np.sort(np.unique(np.sort(F, axis=1), axis=0, return_index=True)[1])]  # drop duplicated faces
    F, removed = drop_non_manifold_faces(F)
    V2, F2, C2, stats = fill(V, F, C, lcfg)
    stats["faces_removed_non_manifold"] = removed

    out_dir = run_dir / "mesh" / "liepa"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    out = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V2), o3d.utility.Vector3iVector(F2.astype(np.int32)))
    out.vertex_colors = o3d.utility.Vector3dVector(C2)
    out.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out_dir / "mesh.ply"), out, write_ascii=False)
    np.save(out_dir / "vertices.npy", V2)
    np.save(out_dir / "faces.npy", F2.astype(np.int32))
    np.save(out_dir / "colors.npy", np.round(C2 * 255).astype(np.uint8))
    export = run_dir / "exports" / "meshlab" / "liepa_scene.ply"
    export.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_dir / "mesh.ply", export)

    row = {"input": str(src.relative_to(run_dir)), **stats, "n_vertices": len(V2), "n_faces": len(F2),
           "t_s": round(time.perf_counter() - t0, 3)}
    row = {k: row.get(k, "") for k in COLUMNS}
    stage_io.upsert_csv(artifacts_dir / "liepa.csv", row)
    stage_io.log_metrics(log, row)
    log.info("liepa: %d/%d holes filled (%d too large, %d non-manifold), +%d vertices -> %s", row["holes_filled"],
             row["holes_found"], row["holes_too_large"], row["holes_nonmanifold"], row["new_vertices"], export)

    return row
# <<< END IMPLEMENTATION <<<
if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=40)
    v, f = np.asarray(sphere.vertices), np.asarray(sphere.triangles)
    centroids = v[f].mean(axis=1)
    cap = centroids @ unit(np.array([1.0, 0.3, 0.2])) > 0.93                 # one large round hole
    small = np.zeros(len(f), bool)
    small[np.argmin(np.linalg.norm(centroids - unit(np.array([-1.0, 0, 0])), axis=1))] = True  # 3-edge hole
    f_holes = f[~cap & ~small]
    lcfg = {"enabled": True, "input": "poisson", "max_hole_edges": 200, "density": float(np.sqrt(2)),
            "fairing": 2, "max_refine_iters": 30}
    V2, F2, C2, st = fill(v, f_holes, np.tile([[0.8, 0.2, 0.1]], (len(v), 1)), lcfg)
    print(st)
    assert st["holes_found"] == 2 and st["holes_filled"] == 2 and st["boundary_edges_after"] == 0, st
    assert st["non_manifold_edges_after"] == 0 and st["faired"] and st["new_vertices"] > 10, st
    r = np.linalg.norm(V2[len(v):], axis=1)
    assert r.min() > 0.9 and r.max() < 1.02, (r.min(), r.max())            # fair patch follows the sphere
    assert np.allclose(C2[len(v):], [0.8, 0.2, 0.1], atol=1e-6)
    n2 = np.cross(V2[F2[:, 1]] - V2[F2[:, 0]], V2[F2[:, 2]] - V2[F2[:, 0]])
    assert (np.sum(n2 * V2[F2].mean(axis=1), axis=1) > 0).all()              # orientation kept (outward)
    # unfaired: flat patch inside the cap; hole cap larger than max_hole_edges: skipped
    V3, _, _, st3 = fill(v, f_holes, np.zeros((len(v), 3)), {**lcfg, "fairing": 0})
    assert not st3["faired"] and np.linalg.norm(V3[len(v):], axis=1).max() < 1.0
    _, _, _, st4 = fill(v, f_holes, np.zeros((len(v), 3)), {**lcfg, "max_hole_edges": 10})
    assert st4["holes_too_large"] == 1 and st4["holes_filled"] == 1 and st4["boundary_edges_after"] > 0
    with tempfile.TemporaryDirectory() as tmp:
        rd, ad = Path(tmp) / "results" / "t", Path(tmp) / "artifacts" / "t"
        (rd / "mesh/poisson/pre").mkdir(parents=True)
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f_holes))
        o3d.io.write_triangle_mesh(str(rd / "mesh/poisson/pre/mesh.ply"), mesh)
        row = run({"liepa": {**lcfg, "input": "close_holes"}}, str(rd), str(ad))
        assert row["holes_filled"] == 2 and (rd / "exports/meshlab/liepa_scene.ply").exists()
        assert len(o3d.io.read_triangle_mesh(str(rd / "exports/meshlab/liepa_scene.ply")).triangles) == row["n_faces"]
        assert stage_io.read_csv(ad / "liepa.csv")[0]["holes_filled"] == "2"
        assert run({"liepa": {**lcfg, "enabled": False}}, str(rd), str(ad)) is None
    print("OK")
