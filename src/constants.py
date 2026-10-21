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
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # every model inherits it: unknown keys fail

class Run(Strict):
    id: Optional[str] = None; output_root: Optional[str] = None; artifacts_root: Optional[str] = None  # None = <ws>/results, <ws>/artifacts
    seed: int = 42; log_level: str = "INFO"; overwrite: bool = False; cache: bool = True

class ExtractionBlur(Strict):
    enabled: bool = True; coef_min: float = 0.5; motion_ratio_max: float = 4.0; keep_for_sfm_only: bool = False
    window: int = 9; discard_width: int = 640; discard_jpeg_quality: int = 90

class Extraction(Strict):
    rerun: bool = False; video: str = "video_preview.MOV"; output_dir: Optional[str] = None; tag: Optional[str] = None  # None = derived
    selection: Literal["all", "overlap"] = "all"  # all = every stride-th frame minus blur drops; overlap = greedy overlap selection
    min_frames: int = 100; max_frames: Optional[int] = None  # None = no cap
    overlap: float = 0.70; overlap_min: float = 0.60; overlap_max: float = 0.80; overlap_step: float = 0.025
    max_adjust_iters: int = 12; stride: int = 1; analysis_width: int = 640; nfeatures: int = 1000
    lowe_ratio: float = 0.75; min_good_matches: int = 15; ransac_threshold_px: float = 3.0; min_inliers: int = 12
    sharp_percentile: float = 25; max_side: int = 1920; image_format: Literal["png", "jpg"] = "png"; jpeg_quality: int = 95
    blur: ExtractionBlur = ExtractionBlur()

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
    max_frames: Optional[int] = 100  # uniform subsample of the quality-kept frames for the 2D stages; None = all

class Report(Strict): max_figures: int = 5; max_rows: int = 20
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
class Measure(Strict): name: Literal["room_width", "room_length", "ceiling_height"]; value_cm: float
class Evaluation(Strict):
    enabled: bool = False; measured_cm: list[Measure] = []; ablation_runs: list[str] = []
    loop_closed: Optional[bool] = None; hard_regions: bool = True; texture_radius_px: float = 30.0
    texture_split_quantile: float = 0.5; inlier_px: float = 2.0

class Sfm(Strict):
    enabled: bool = True; rerun: bool = False; colmap_bin: str = "colmap"; camera_model: str = "OPENCV"
    single_camera: bool = True; intrinsics: Literal["estimate", "fixed"] = "estimate"
    fixed_dist: list[float] = [0.0, 0.0, 0.0, 0.0]; matcher: Literal["sequential", "exhaustive"] = "sequential"
    sequential_overlap: int = 10; loop_detection: bool = False; vocab_tree: Optional[str] = None
    mapper: Literal["incremental", "global"] = "incremental"; use_gpu: Union[bool, Literal["auto"]] = "auto"; gpu_index: str = "0"
    ba_use_gpu: bool = False; max_image_size: int = 1920; max_num_features: int = 8192
    min_num_matches: int = 15; multiple_models: bool = True
    ba_global_max_num_iterations: Optional[int] = None; ba_global_max_refinements: Optional[int] = None
    project_quality: Literal["low", "medium", "high", "extreme"] = "high"
    link_images: Literal["auto", "hardlink", "symlink", "copy"] = "auto"  # how exports/colmap/images/ holds the frames
    backend: Literal["auto", "cli", "pycolmap"] = "auto"  # auto = COLMAP CLI when sfm.colmap_bin is found, else pycolmap

class Click(Strict): image: str; x: float; y: float
class ManualPair(Strict):
    length_cm: float
    p: Annotated[list[Click], Field(min_length=2, max_length=2)]
    q: Annotated[list[Click], Field(min_length=2, max_length=2)]

class Scale(Strict):
    enabled: bool = True; method: Literal["aruco", "manual"] = "aruco"; aruco_dict: str = "DICT_4X4_50"
    marker_cm: Optional[float] = None; min_parallax_deg: float = 1.0; manual_pairs: list[ManualPair] = []
    provisional_extent_cm: float = 500

class Cleanup(Strict):
    enabled: bool = True; voxel_cm: float = 1.0; nb_neighbors: int = 20; std_ratio: float = 2.0
    crop_percentiles: list[float] = [1, 99]; device: str = "auto"  # auto | CPU:0 | CUDA:0

class Planes(Strict):
    enabled: bool = True; source: Literal["sparse", "patch_match", "openmvs"] = "sparse"
    distance_cm: float = 2.0; ransac_n: int = 3; iterations: int = 2000; max_walls: int = 4
    min_inlier_frac: float = 0.02; angle_tol_deg: float = 15.0; max_planes: int = 10

class Floorplan(Strict):
    enabled: bool = True; cell_cm: float = 5.0; dpi: int = 150; fig_inches: float = 8.0
    extent_percentiles: list[float] = [1, 99]

class DensePatchMatch(Strict):
    num_iterations: int = 5; window_radius: int = 5; num_samples: int = 15; num_src_images: int = 20
    geom_consistency: bool = True; cache_size: int = 32
class DenseFusion(Strict): min_num_pixels: int = 5; max_reproj_error: float = 2.0; max_depth_error: float = 0.01; cache_size: int = 32
class OpenMVS(Strict): bin_dir: Optional[str] = None; resolution_level: int = 1; number_views: int = 5; max_threads: int = 0

class Dense(Strict):
    enabled: bool = False; rerun: bool = False
    methods: list[Literal["patch_match", "openmvs"]] = ["patch_match", "openmvs"]
    max_image_size: int = 1600; gpu_index: str = "0"; ref_step: int = 1
    patch_match: DensePatchMatch = DensePatchMatch(); fusion: DenseFusion = DenseFusion(); openmvs: OpenMVS = OpenMVS()

class DenseCompare(Strict): enabled: bool = True

class Poisson(Strict):
    enabled: bool = True; source: Literal["dense", "sparse"] = "dense"
    variants: list[Literal["pre", "raw"]] = ["pre", "raw"]; depth: int = 10; density_quantile: float = 0.05
    normal_knn: int = 30; threads: int = 8

class CloseHoles(Strict):
    enabled: bool = True; variants: list[Literal["pre", "raw"]] = ["pre", "raw"]; max_hole_size: int = 30
    min_component_frac: float = 0.01; self_intersection: bool = True

class Mesh(Strict):
    enabled: bool = True; input: Literal["close_holes", "poisson"] = "close_holes"
    frustum_cm: float = 20; frustum_every: int = 5; trajectory_cm: float = 1.0

class Export(Strict):
    enabled: bool = True; normals_k: int = 10; normals_smooth_iter: int = 0
    poisson_depth: int = 10; poisson_samples_per_node: float = 1.5; poisson_point_weight: float = 4.0
    isolated_diameter_frac: float = 0.1

class Viewer(Strict):
    enabled: bool = True; cloud: Literal["sparse", "patch_match", "openmvs", "mesh"] = "sparse"
    seconds: int = 45; fps: int = 30; offscreen: Union[bool, Literal["auto"]] = "auto"; timeout_s: float = 1800
    width: int = 1280; height: int = 720
    frustum_cm: float = 20.0; point_size: float = 2.0; fov_deg: float = 60.0
    orbit_radius_factor: float = 1.5; orbit_elevation_deg: float = 30.0; background: list[float] = [1.0, 1.0, 1.0]

class Videos(Strict): enabled: bool = True; topdown_fps: float = 10; kept_vs_discarded: bool = True; width: int = 960; codec: str = "auto"

class Config(Strict):
    run: Run = Run(); extraction: Extraction = Extraction(); frame_quality: FrameQuality = FrameQuality()
    detection: Detection = Detection(); report: Report = Report(); filters: Filters = Filters()
    matching: Matching = Matching(); geometry: Geometry = Geometry(); visualize: Visualize = Visualize()
    evaluation: Evaluation = Evaluation(); sfm: Sfm = Sfm(); scale: Scale = Scale(); cleanup: Cleanup = Cleanup()
    planes: Planes = Planes(); floorplan: Floorplan = Floorplan(); dense: Dense = Dense()
    dense_compare: DenseCompare = DenseCompare(); poisson: Poisson = Poisson(); close_holes: CloseHoles = CloseHoles()
    mesh: Mesh = Mesh(); export: Export = Export()
    viewer: Viewer = Viewer(); videos: Videos = Videos()
