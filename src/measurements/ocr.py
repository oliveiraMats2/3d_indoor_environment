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
# Stage: measurements (offline helper, not a main.py stage) - OCR of the tape-measure labels.
# Goal: read the iPhone Measure labels ("2,73 m") in the reference photos, so the real lengths versioned in
#      configs/measurements/<scene>.yaml can be re-derived from the photos.
# In:  <folder>/imagem_medidas_*.jpg (photos with the measuring-app overlay).
# Out: <out.json> = {image name: [{m, x, y, conf, rot_deg}, ...]} sorted by x (pixel centre of each label
#      in the unrotated photo); also printed per image.
# CLI: python -m src.measurements.ocr <folder> <out.json> [--pattern GLOB]
# Rules: rapidocr_onnxruntime is optional -> WARNING and return None when missing; the labels are rotated
#      along the measured line, so each photo is read at several rotations and every "d,dd m" match is
#      kept with its unrotated centre; matches closer than MERGE_PX are merged (highest confidence wins).

# >>> BEGIN IMPLEMENTATION >>>
"""OCR of the measuring-app labels in the reference photos -> lengths in metres per image."""
import argparse
import json
import logging
import re
import sys
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

PATTERN = re.compile(r"(\d+)[,.](\d{2})\s*m")  # "2,73 m" / "2.73m"
ANGLES = range(-45, 50, 5)                      # rotations tried (deg)
MERGE_PX = 40                                   # labels closer than this (both axes) are one label
GLOB = "imagem_medidas_*.jpg"

def engine():
    """RapidOCR instance, or None (WARNING) when rapidocr_onnxruntime is not installed."""

    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        log.warning("measurements.ocr: rapidocr_onnxruntime not installed "
                    "(uv pip install rapidocr-onnxruntime==1.4.4); OCR skipped")
        return None

    return RapidOCR()

def rotate(img: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """img rotated by angle (deg) on an enlarged white canvas, and the 2x3 affine used."""

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2

    return cv2.warpAffine(img, M, (nw, nh), borderValue=(255, 255, 255)), M

def read_image(ocr, img: np.ndarray) -> list[dict]:
    """Labels of one photo: [{m, x, y, conf, rot_deg}] sorted by x, duplicates merged."""

    found = []
    for angle in ANGLES:
        rot, M = rotate(img, angle)
        res, _ = ocr(rot)
        Minv = cv2.invertAffineTransform(M)
        for box, text, conf in res or []:
            m = PATTERN.search(text.replace(" ", ""))
            if not m:
                continue
            c = np.mean(np.asarray(box, float), axis=0)
            x, y = Minv @ np.array([c[0], c[1], 1.0])
            found.append((float(f"{m.group(1)}.{m.group(2)}"), float(x), float(y), float(conf), angle))
    merged = []
    for v, x, y, conf, a in sorted(found, key=lambda f: -f[3]):
        if all(abs(x - mx) > MERGE_PX or abs(y - my) > MERGE_PX for _, mx, my, _, _ in merged):
            merged.append((v, x, y, conf, a))

    return [{"m": v, "x": round(x), "y": round(y), "conf": round(c, 3), "rot_deg": a}
            for v, x, y, c, a in sorted(merged, key=lambda f: f[1])]

def read_folder(folder: str, out: str | None = None, pattern: str = GLOB, ocr=None) -> dict | None:
    """{image name: labels} for every photo of folder matching pattern; written to out (JSON) when given.
    None when there is no OCR engine or no photo."""

    ocr = ocr or engine()
    if ocr is None:
        return None
    paths = sorted(Path(folder).glob(pattern))
    if not paths:
        log.warning("measurements.ocr: no %s in %s", pattern, folder)
        return None

    result = {}
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            log.warning("measurements.ocr: cannot read %s", p)
            continue
        result[p.name] = read_image(ocr, img)
        log.info("%s: %s", p.name, [l["m"] for l in result[p.name]])
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=1, ensure_ascii=False))
        log.info("measurements.ocr -> %s", out)

    return result

def main(argv=None) -> int:

    parser = argparse.ArgumentParser(description="OCR of the measuring-app labels (\"2,73 m\") in the photos.")
    parser.add_argument("folder", help="folder with the photos")
    parser.add_argument("out", help="output JSON {image: [{m, x, y, conf, rot_deg}]}")
    parser.add_argument("--pattern", default=GLOB, help=f"photo glob (default {GLOB})")
    args = parser.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    return 0 if read_folder(args.folder, args.out, args.pattern) is not None else 1

# <<< END IMPLEMENTATION <<<

if __name__ == "__main__" and len(sys.argv) > 1:
    sys.exit(main())

if __name__ == "__main__":
    import tempfile

    # Fake engine: called once per angle of ANGLES (in order); reports "2,73 m" (and noise) only at +10 and
    # +15 deg, centred where that rotation maps the point (300, 200) -> the unrotated centre must come back
    # and the two hits must merge into the more confident one.
    target = np.array([300.0, 200.0, 1.0])
    def make_fake():
        angles = iter(list(ANGLES) * 2)
        def fake(img):
            angle = next(angles)
            if angle not in (10, 15):
                return [], None
            _, M = rotate(np.zeros((400, 600, 3), np.uint8), angle)
            cx, cy = M @ target
            box = [[cx - 30, cy - 10], [cx + 30, cy - 10], [cx + 30, cy + 10], [cx - 30, cy + 10]]
            return [(box, "2,73 m", 0.9 if angle == 10 else 0.8), (box, "abc", 0.99)], None
        return fake

    labels = read_image(make_fake(), np.full((400, 600, 3), 255, np.uint8))
    assert len(labels) == 1, labels
    assert labels[0]["m"] == 2.73 and labels[0]["rot_deg"] == 10, labels
    assert abs(labels[0]["x"] - 300) <= 1 and abs(labels[0]["y"] - 200) <= 1, labels
    assert PATTERN.search("4.29m") and not PATTERN.search("273 cm")

    with tempfile.TemporaryDirectory() as tmp:
        cv2.imwrite(f"{tmp}/imagem_medidas_1.jpg", np.full((400, 600, 3), 255, np.uint8))
        res = read_folder(tmp, f"{tmp}/out/ocr.json", ocr=make_fake())
        assert json.loads(Path(f"{tmp}/out/ocr.json").read_text()) == res
        assert res["imagem_medidas_1.jpg"][0]["m"] == 2.73
        assert read_folder(f"{tmp}/out", ocr=make_fake()) is None  # no photo -> None

    print("OK")
