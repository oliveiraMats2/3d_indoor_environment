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
"""Pairwise frame overlap: ratio test + RANSAC homography + convex intersection."""
import cv2
import numpy as np

class Sobreposicao:
    def __init__(self, feats, wh, cfg):

        self.feats, self.cache, self.cfg = feats, {}, cfg
        self.failed = set()  # pairs with no geometric estimate (few keypoints/matches/inliers, degenerate H): 0.0 = unknown, not measured
        w, h = wh
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        self.rect = np.float32(corners)
        self.area = float(w * h)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def __call__(self, a, b):

        if (a, b) not in self.cache:
            ratio = self._compute(a, b)
            if ratio is None:
                self.failed.add((a, b))
            self.cache[(a, b)] = ratio or 0.0

        return self.cache[(a, b)]

    def _compute(self, a, b):

        c = self.cfg
        (pa, da), (pb, db) = self.feats[a], self.feats[b]
        if da is None or db is None or len(pa) < 20 or len(pb) < 20:
            return None

        matches = self.bf.knnMatch(db, da, k=2)
        pairs = [p for p in matches if len(p) == 2]
        good = [m for m, n in pairs if m.distance < c["lowe_ratio"] * n.distance]
        if len(good) < c["min_good_matches"]:
            return None

        src_points = [pb[m.queryIdx] for m in good]
        dst_points = [pa[m.trainIdx] for m in good]
        src = np.float32(src_points)
        dst = np.float32(dst_points)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, c["ransac_threshold_px"])
        if H is None or mask.sum() < c["min_inliers"]:
            return None

        rect_3d = self.rect.reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(rect_3d, H)
        quad = warped.astype(np.float32)
        if not cv2.isContourConvex(quad):
            return None

        quad_2d = quad.reshape(-1, 2)
        area, _ = cv2.intersectConvexConvex(self.rect, quad_2d)
        ratio = area / self.area
        clipped = min(ratio, 1.0)

        return float(clipped)
