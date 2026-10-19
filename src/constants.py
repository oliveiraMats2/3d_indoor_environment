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

"""YAML schema: every block of the config with its default value. Unknown keys fail validation."""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # every model inherits it: unknown keys fail

class Run(Strict):
    id: Optional[str] = None; output_root: str = "results"; artifacts_root: str = "artifacts"
    seed: int = 42; log_level: str = "INFO"; overwrite: bool = False; cache: bool = True

class Extraction(Strict):
    rerun: bool = False; video: str = "3d_image_video.MOV"; output_dir: str = "frames"; min_frames: int = 100; max_frames: int = 300
    overlap: float = 0.70; overlap_min: float = 0.60; overlap_max: float = 0.80; overlap_step: float = 0.025
    max_adjust_iters: int = 12; stride: int = 2; analysis_width: int = 640; nfeatures: int = 1000
    lowe_ratio: float = 0.75; min_good_matches: int = 15; ransac_threshold_px: float = 3.0; min_inliers: int = 12
    sharp_percentile: float = 25; max_width: int = 1920; image_format: Literal["png", "jpg"] = "png"; jpeg_quality: int = 95

class Blur(Strict): laplacian_var_min: float = 100.0; relative_to_median: float = 0.5
class Exposure(Strict): mean_min: float = 40; mean_max: float = 215; clipped_max_frac: float = 0.05
class Contrast(Strict): std_min: float = 20.0
class Texture(Strict): min_keypoints: int = 300; fast_threshold: int = 20

class FrameQuality(Strict):
    resize_width: int = 640
    blur: Blur = Blur(); exposure: Exposure = Exposure(); contrast: Contrast = Contrast(); texture: Texture = Texture()

class Detection(Strict):
    methods: list[Literal["sift", "orb", "akaze"]] = ["sift", "orb", "akaze"]
    nfeatures: int = 5000; reps: int = 30; warmup: int = 2

class Report(Strict): max_figures: int = 5
class Filters(Strict): lowe_ratio: float = 0.75; min_matches: int = 10
class FloatIndex(Strict): algorithm: int = 1; trees: int = 5
class BinaryIndex(Strict): algorithm: int = 6; table_number: int = 6; key_size: int = 12; multi_probe_level: int = 1
class Flann(Strict): checks: int = 50; float_index: FloatIndex = FloatIndex(); binary_index: BinaryIndex = BinaryIndex()

class Matching(Strict):
    enabled: bool = False
    name: Literal["brute_force", "flann"] = "brute_force"
    strategy: Literal["sequential", "all_pairs"] = "sequential"
    cross_check: Literal[False] = False  # knnMatch(k=2) for Lowe excludes crossCheck
    reps: int = 30
    warmup: int = 2
    flann: Flann = Flann()

class Ransac(Strict): method: str = "USAC_MAGSAC"; threshold_px: float = 1.0; confidence: float = 0.999; max_iters: int = 5000
class Intrinsics(Strict): fx: Optional[float] = None; fy: Optional[float] = None; cx: Optional[float] = None; cy: Optional[float] = None
class Geometry(Strict): enabled: bool = False; ransac: Ransac = Ransac(); intrinsics: Intrinsics = Intrinsics()
class MatchesViz(Strict): max_lines: int = 100; dpi: int = 150; format: str = "png"
class EpilinesViz(Strict): n_lines: int = 20; dpi: int = 150
class Visualize(Strict): matches: MatchesViz = MatchesViz(); epilines: EpilinesViz = EpilinesViz()
class Measure(Strict): name: str; value_cm: float
class Evaluation(Strict): enabled: bool = False; measured_cm: list[Measure] = []; ablation_runs: list[str] = []

class Config(Strict):
    run: Run = Run(); extraction: Extraction = Extraction(); frame_quality: FrameQuality = FrameQuality()
    detection: Detection = Detection(); report: Report = Report(); filters: Filters = Filters()
    matching: Matching = Matching(); geometry: Geometry = Geometry(); visualize: Visualize = Visualize()
    evaluation: Evaluation = Evaluation()
