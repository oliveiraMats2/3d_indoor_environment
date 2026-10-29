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
class Filters(Strict): lowe_ratio: float = 0.75; min_matches: int = 10; cross_check: bool = False  # src/matching/cross_check.py
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
class MatchesViz(Strict): enabled: bool = False; samples: int = 5; max_lines: int = 100; dpi: int = 150; format: str = "png"
class EpilinesViz(Strict): n_lines: int = 20; dpi: int = 150
class Visualize(Strict): matches: MatchesViz = MatchesViz(); epilines: EpilinesViz = EpilinesViz()
class Measure(Strict): name: str; value_cm: Optional[float] = None  # room_width | room_length | ceiling_height | <wall name>
class Evaluation(Strict):
    enabled: bool = False; measured_cm: list[Measure] = []; ablation_runs: list[str] = []
    loop_closed: Optional[bool] = None; hard_regions: bool = True; texture_radius_px: float = 30.0
    texture_split_quantile: float = 0.5; inlier_px: float = 2.0
    measurements_file: Optional[str] = None  # tape measurements yaml (paredes_cm, pe_direito_cm), e.g. configs/measurements/
    metric_scale: Literal["auto", "scale", "leave_one_out"] = "auto"  # leave_one_out = wall scale from the other walls
    loop_auto: dict = {}  # src/evaluation/loop_drift.py LOOP_AUTO overrides (automatic loop-closure check)
    hard_cases: dict = {}  # src/evaluation/hard_regions.py overrides (glare, dark, depth boundary, visibility)

class InstantSfm(Strict):  # src/sfm/instantsfm.py
    num_iteration_bundle_adjustment: int = 3; ba_max_num_iterations: int = 100
    skip_view_graph_calibration: bool = False; skip_retriangulation: bool = True; skip_pruning: bool = True
    max_frames: Optional[int] = None  # None = every frame (scales to thousands)
    min_track_length: int = 3  # shorter tracks dropped before positioning/BA; raise it when the GPU runs out of memory
    ba: bool = False  # extra pycolmap BA after the retriangulation (InstantSfM already ran its GPU BA)
    min_points_per_image: int = 50  # images seeing fewer 3D points are deregistered

class Mast3rSfm(Strict):  # src/sfm/mast3r_sfm.py
    repo_dir: str = "~/mast3r"; weights: str = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    retrieval_url: str = "https://download.europe.naverlabs.com/ComputerVision/MASt3R"; weights_dir: str = "~/.cache/mast3r"
    image_size: int = 512; max_frames: Optional[int] = 200  # uniform subsample; memory and pairs grow with it
    scene_graph: Literal["retrieval", "swin"] = "retrieval"; retrieval_na: int = 20; retrieval_k: int = 10; window: int = 10
    lr1: float = 0.07; niter1: int = 300; lr2: float = 0.01; niter2: int = 300
    optim_level: Literal["coarse", "refine", "refine+depth"] = "refine+depth"; matching_conf_thr: float = 0.0
    shared_intrinsics: bool = True; subsample: int = 8; ba: bool = True; refine_intrinsics: bool = True
    matcher: Literal["exhaustive", "sfm"] = "exhaustive"  # SIFT pairs of the subsampled frames; sfm = sfm.matcher
    min_points_per_image: int = 50  # images seeing fewer 3D points are deregistered

class Vggt(Strict):  # src/sfm/vggt.py
    weights: str = "facebook/VGGT-1B"; image_size: int = 518; max_frames: Optional[int] = 80  # one forward pass: VRAM-bound
    dtype: Literal["auto", "bf16", "fp16", "fp32"] = "auto"; ba: bool = True; refine_intrinsics: bool = True
    matcher: Literal["exhaustive", "sfm"] = "exhaustive"  # SIFT pairs of the subsampled frames; sfm = sfm.matcher
    min_points_per_image: int = 50  # images seeing fewer 3D points are deregistered

class Sfm(Strict):
    technique: Literal["colmap", "glomap", "instantsfm", "mast3r_sfm", "vggt"] = "colmap"  # glomap = colmap with mapper global
    enabled: bool = True; rerun: bool = False; colmap_bin: str = "colmap"; camera_model: str = "OPENCV"
    single_camera: bool = True; intrinsics: Literal["estimate", "fixed"] = "estimate"
    camera_groups: bool = False  # colmap/glomap: one camera per frame-name group (digits stripped: img_k*, img_o*) = frames merged from several videos
    fixed_dist: list[float] = [0.0, 0.0, 0.0, 0.0]; matcher: Literal["sequential", "exhaustive"] = "sequential"
    sequential_overlap: int = 10; loop_detection: bool = False; vocab_tree: Optional[str] = None
    mapper: Literal["incremental", "global"] = "incremental"; use_gpu: Union[bool, Literal["auto"]] = "auto"; gpu_index: str = "0"
    ba_use_gpu: bool = False; max_image_size: int = 1920; max_num_features: int = 8192
    min_num_matches: int = 15; multiple_models: bool = True
    ba_global_max_num_iterations: Optional[int] = None; ba_global_max_refinements: Optional[int] = None
    max_camera_spread: Optional[float] = 5.0  # cameras farther than this x p95 of the centre spread are dropped; None = off
    ba_global_frames_ratio: Optional[float] = None  # None = COLMAP 1.1; 1.3 = fewer global BAs (incremental only)
    project_quality: Literal["low", "medium", "high", "extreme"] = "high"
    link_images: Literal["auto", "hardlink", "symlink", "copy"] = "auto"  # how exports/colmap/images/ holds the frames
    backend: Literal["auto", "cli", "pycolmap"] = "auto"  # auto = COLMAP CLI when sfm.colmap_bin is found, else pycolmap
    container: Optional[str] = None  # docker image with COLMAP + Ceres/cuDSS (docker/colmap-cudss); None = host colmap_bin
    container_steps: list[str] = ["mapper"]  # feature | match | mapper | export | all: steps run in the container
    container_ba_use_gpu: Optional[bool] = True  # GPU BA when the mapper runs in the container; None = ba_use_gpu
    instantsfm: InstantSfm = InstantSfm(); mast3r_sfm: Mast3rSfm = Mast3rSfm(); vggt: Vggt = Vggt()

class Click(Strict): image: str; x: float; y: float
class ManualPair(Strict):
    length_cm: float
    p: Annotated[list[Click], Field(min_length=2, max_length=2)]
    q: Annotated[list[Click], Field(min_length=2, max_length=2)]

class ScaleOutline(Strict):  # src/sfm/scale_outline.py: tape-measured room outline fitted to the sparse walls
    footprint_cm: Optional[list[Annotated[list[float], Field(min_length=2, max_length=2)]]] = None
    measurements: Optional[str] = None  # yaml with contorno_cm / footprint_cm (configs/measurements/)
    fallback: bool = True  # aruco/manual left the scale unknown -> fit the outline
    wall_band: Optional[list[float]] = None; fit_rotations: Optional[int] = None  # None = room_fill values
    voxel_cm: float = 2.0; nb_neighbors: int = 20; std_ratio: float = 2.0; crop_percentiles: list[float] = [1.0, 99.0]
    max_residual_cm: float = 10.0; min_inlier_frac: float = 0.5; max_p90_cm: float = 30.0; min_support: float = 0.5

class Scale(Strict):
    enabled: bool = True; method: Literal["aruco", "manual", "outline"] = "aruco"; aruco_dict: str = "DICT_4X4_50"
    marker_cm: Optional[float] = None; min_parallax_deg: float = 1.0; manual_pairs: list[ManualPair] = []
    provisional_extent_cm: float = 500
    outline: ScaleOutline = ScaleOutline()

class Cleanup(Strict):
    enabled: bool = True; voxel_cm: float = 1.0; nb_neighbors: int = 20; std_ratio: float = 2.0
    crop_percentiles: list[float] = [1, 99]; device: str = "auto"  # auto | CPU:0 | CUDA:0

class Planes(Strict):
    enabled: bool = True; source: str = "sparse"  # sparse | <dense method>
    distance_cm: float = 2.0; ransac_n: int = 3; iterations: int = 2000; max_walls: int = 4
    min_inlier_frac: float = 0.02; angle_tol_deg: float = 15.0; max_planes: int = 10
    wall_distance_cm: float = 4.0; wall_rounds: int = 40; wall_band: list[float] = [0.4, 0.95]  # vertical walls, top view
    wall_min_inlier_frac: float = 0.005; wall_min_rel: float = 0.08; wall_min_height_frac: float = 0.3
    wall_max_behind_frac: float = 1.5; merge_angle_deg: float = 10.0; merge_offset_frac: float = 0.04
    segment_gap_frac: float = 0.08; neighbour_min_deg: float = 30.0; corner_gap_frac: float = 0.15

class Floorplan(Strict):
    enabled: bool = True; cell_cm: float = 5.0; dpi: int = 150; fig_inches: float = 8.0
    extent_percentiles: list[float] = [1, 99]
    min_wall_directions: int = 2; min_aspect: float = 0.15; wall_angle_tol_deg: float = 20.0
    fallback: list[Literal["room_fill", "percentiles"]] = ["room_fill", "percentiles"]  # when the wall rect degenerates
    fallback_percentiles: list[float] = [5, 95]

class DensePatchMatch(Strict):
    num_iterations: int = 5; window_radius: int = 5; num_samples: int = 15; num_src_images: int = 20
    geom_consistency: bool = True; cache_size: int = 32
class DenseFusion(Strict): min_num_pixels: int = 5; max_reproj_error: float = 2.0; max_depth_error: float = 0.01; cache_size: int = 32
class OpenMVS(Strict):
    bin_dir: Optional[str] = None; container: Optional[str] = None  # docker image (docker/openmvs) over bin_dir
    resolution_level: int = 1; number_views: int = 5; max_threads: int = 0; cuda_device: int = -1; remove_dmaps: bool = True

class PlaneSweep(Strict):  # src/dense/plane_sweep.py (torch, GPU)
    max_size: int = 800; num_depths: int = 192; num_sources: int = 6; best_k: int = 3; window_radius: int = 4
    min_zncc: float = 0.4; min_std: float = 0.01; peak_ratio: float = 0.95; chunk_mb: int = 2048

class Sgm(Strict):  # src/dense/sgm.py (torch GPU | opencv CPU)
    backend: Literal["torch", "opencv"] = "torch"; max_size: int = 800; num_pairs: int = 2; p1: float = 10; p2: float = 120
    census_window: int = 7; paths: int = 8; lr_max_diff: float = 1; min_angle_deg: float = 3.0; max_angle_deg: float = 20.0
    max_disparities: int = 256; min_valid_frac: float = 0.5; depth_rel_tol: float = 0.02; speckle_size: int = 100
    normal_radius: int = 6

class CasMvsNet(Strict):  # src/dense/casmvsnet.py (official alibaba/cascade-stereo weights)
    max_size: int = 1152; num_views: int = 5; num_depth: int = 192; conf_threshold: float = 0.9
    ndepths: list[int] = [48, 32, 8]; depth_interval_ratio: list[float] = [4, 2, 1]; cr_base_chs: list[int] = [8, 8, 8]

class MvsNet(Strict):  # src/dense/mvsnet.py (official YoYo000/MVSNet TF weights, converted)
    weights: Literal["blendedmvs", "dtu"] = "blendedmvs"; max_size: int = 1600; num_views: int = 5; num_depth: int = 192
    prob_threshold: float = 0.8; depth_chunk: int = 16

class PatchmatchNet(Strict):  # src/dense/patchmatchnet.py (official FangjinhuaWang/PatchmatchNet weights)
    max_size: int = 1600; num_views: int = 5; conf_threshold: float = 0.5
    patchmatch_interval_scale: list[float] = [0.005, 0.0125, 0.025]; patchmatch_range: list[int] = [6, 4, 2]
    patchmatch_iteration: list[int] = [1, 2, 2]; patchmatch_num_sample: list[int] = [8, 8, 16]
    propagate_neighbors: list[int] = [0, 8, 16]; evaluate_neighbors: list[int] = [9, 9, 9]
    checkpoint: Literal["params", "module"] = "params"

class VggtOmega(Strict):  # src/dense/vggt_omega.py (neural multi-view reconstruction)
    model: Literal["vggt_1b", "vggt_omega"] = "vggt_1b"  # vggt_omega needs access to the gated facebook/VGGT-Omega
    checkpoint: Optional[str] = None; max_frames: Optional[int] = 80; conf_percentile: float = 50.0
    image_size: Optional[int] = None; use_sfm_poses: bool = True; edge_rtol: float = 0.03; ransac_thresh: float = 0.05
    voxel_size: Optional[float] = None; dtype: Literal["auto", "bf16", "fp16", "fp32"] = "auto"

class MapAnything(Strict):  # src/dense/mapanything.py (neural reconstruction with SfM intrinsics/poses as input)
    weights: str = "facebook/map-anything"; max_frames: Optional[int] = 80; use_sfm_poses: bool = True
    use_intrinsics: bool = True; conf_percentile: float = 50; voxel: Union[float, Literal["auto"], None] = "auto"

DENSE_METHODS = Literal["patch_match", "openmvs", "plane_sweep", "sgm", "casmvsnet", "mvsnet", "patchmatchnet",
                        "vggt_omega", "mapanything"]

class Dense(Strict):
    enabled: bool = False; rerun: bool = False
    methods: list[DENSE_METHODS] = ["patch_match", "openmvs"]
    max_image_size: int = 1600; gpu_index: str = "0"; ref_step: int = 1
    patch_match: DensePatchMatch = DensePatchMatch(); fusion: DenseFusion = DenseFusion(); openmvs: OpenMVS = OpenMVS()
    plane_sweep: PlaneSweep = PlaneSweep(); sgm: Sgm = Sgm(); casmvsnet: CasMvsNet = CasMvsNet(); mvsnet: MvsNet = MvsNet()
    patchmatchnet: PatchmatchNet = PatchmatchNet(); vggt_omega: VggtOmega = VggtOmega(); mapanything: MapAnything = MapAnything()

class DenseCompare(Strict): enabled: bool = True; device: str = "auto"  # nearest neighbours: auto | CPU:0 | CUDA:0

class GaussianSplatting(Strict):
    enabled: bool = False; rerun: bool = False
    repo_dir: str = "~/gaussian-splatting"; python: str = "~/gaussian-splatting/.venv/bin/python"  # Inria clone and its venv
    iterations: int = 30000; eval: bool = True  # eval = hold out every 8th image (llffhold) for PSNR/SSIM/LPIPS
    resolution: int = -1  # Inria -r: -1 = auto (at most 1600 px wide), 1|2|4|8 = downscale factor, else target width
    max_image_size: int = 1600  # image_undistorter --max_image_size
    data_device: Literal["cpu", "cuda"] = "cpu"  # cpu: thousands of frames do not fit in 16 GB VRAM
    sh_degree: int = 3; gpu_index: str = "0"; timeout_s: Optional[float] = None  # None = no timeout per command

class Poisson(Strict):
    enabled: bool = True; source: Literal["dense", "sparse"] = "dense"
    variants: list[Literal["pre", "raw"]] = ["pre", "raw"]; depth: int = 10; density_quantile: float = 0.05
    normal_knn: int = 30; threads: int = 8
    method: Union[str, list[Optional[str]], None] = None  # None = first dense method with output; names -> mesh/poisson_<m>/
    device: str = "auto"  # nearest-camera normal orientation on CUDA when available

class CloseHoles(Strict):
    enabled: bool = True; variants: list[Literal["pre", "raw"]] = ["pre", "raw"]; max_hole_size: int = 30
    min_component_frac: float = 0.01; self_intersection: bool = True

class Liepa(Strict):
    enabled: bool = False; input: Literal["poisson", "close_holes"] = "poisson"; max_hole_edges: int = 200
    density: float = 2 ** 0.5; fairing: Literal[0, 1, 2] = 2; max_refine_iters: int = 30

class RoomFill(Strict):  # src/mesh/room_fill.py
    enabled: bool = False
    footprint_cm: list[Annotated[list[float], Field(min_length=2, max_length=2)]] = []  # [] -> stage skipped
    ceiling_height_cm: Optional[float] = None  # None = estimated from the reconstruction
    surfaces: list[Literal["floor", "ceiling", "walls"]] = ["floor", "ceiling"]
    source: str = "patch_match"  # sparse | <dense method>
    device: str = "CPU:0"
    grid_cm: float = 5.0; max_gap_cm: float = 5.0; fit_rotations: int = 72
    wall_band: Annotated[list[float], Field(min_length=2, max_length=2)] = [0.2, 0.8]  # height fractions of the fit points

class Mesh(Strict):
    enabled: bool = True; input: Literal["close_holes", "poisson", "liepa"] = "close_holes"
    frustum_cm: float = 20; frustum_every: int = 5; trajectory_cm: float = 1.0

class Export(Strict):
    enabled: bool = True; normals_k: int = 10; normals_smooth_iter: int = 0
    poisson_depth: int = 10; poisson_samples_per_node: float = 1.5; poisson_point_weight: float = 4.0
    isolated_diameter_frac: float = 0.1

class Viewer(Strict):
    enabled: bool = True; cloud: str = "sparse"  # sparse | mesh | <dense method>
    clouds: Optional[list[str]] = None  # one navigation video per cloud; None = [cloud]
    figures: bool = True  # figures/sparse_poses.png and sparse_poses_top.png
    seconds: int = 45; fps: int = 30; offscreen: Union[bool, Literal["auto"]] = "auto"; timeout_s: float = 1800
    width: int = 1280; height: int = 720
    frustum_cm: float = 20.0; point_size: float = 2.0; fov_deg: float = 60.0
    orbit_radius_factor: float = 1.5; orbit_elevation_deg: float = 30.0; background: list[float] = [1.0, 1.0, 1.0]

class Videos(Strict): enabled: bool = True; topdown_fps: float = 10; kept_vs_discarded: bool = True; width: int = 960; codec: str = "auto"

class PhotoVsRender(Strict):  # src/evaluation/photo_vs_render.py (T2 4.B.3)
    enabled: bool = True; n_views: int = 6; source: Literal["mesh", "dense", "sparse"] = "mesh"; max_size: int = 960
    point_px: int = 4; gs: bool = True

class Lidar(Strict):  # src/evaluation/lidar.py: real LiDAR map = ground truth
    path: Optional[str] = None  # None = no LiDAR, stage skipped
    unit: Literal["m", "cm", "mm"] = "m"; voxel_cm: float = 1.0; taus_cm: list[float] = [2, 5, 10]
    icp_voxels_cm: list[float] = [8, 4, 2]; methods: Union[Literal["auto"], list[str]] = "auto"; up: str = "auto"
    crop_margin_cm: float = 10.0; mesh: bool = True; poisson_depth: int = 10; measurements: Optional[str] = None
    min_fitness: float = 0.5

class Config(Strict):
    flow: Optional[Literal["cpu", "gpu"]] = "cpu"  # null = cpu without preset; src/flows.py presets: cpu = COLMAP incremental + ablations, gpu = GLOMAP + cuDSS
    run: Run = Run(); extraction: Extraction = Extraction(); frame_quality: FrameQuality = FrameQuality()
    detection: Detection = Detection(); report: Report = Report(); filters: Filters = Filters()
    matching: Matching = Matching(); geometry: Geometry = Geometry(); visualize: Visualize = Visualize()
    evaluation: Evaluation = Evaluation(); sfm: Sfm = Sfm(); scale: Scale = Scale(); cleanup: Cleanup = Cleanup()
    planes: Planes = Planes(); floorplan: Floorplan = Floorplan(); dense: Dense = Dense()
    dense_compare: DenseCompare = DenseCompare(); gaussian_splatting: GaussianSplatting = GaussianSplatting()
    poisson: Poisson = Poisson(); close_holes: CloseHoles = CloseHoles()
    liepa: Liepa = Liepa(); room_fill: RoomFill = RoomFill(); mesh: Mesh = Mesh(); export: Export = Export()
    viewer: Viewer = Viewer(); videos: Videos = Videos()
    photo_vs_render: PhotoVsRender = PhotoVsRender(); lidar: Lidar = Lidar()
