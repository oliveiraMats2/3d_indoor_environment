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

# ==============================================================================
# PROMPT COMPLEMENT
# This block is a prompt complement for the AI assistant that will implement
# this file. Read it fully before writing any code. It is self-contained: do not
# look for other documents.
#
# Stage: Stage 5, Evaluation          Owner: unassigned          Due: TBD
# Project: 3D reconstruction of an indoor environment (variant A) from frames
#          extracted from a phone video.
# Context: final metrics of the reconstruction, required for every group, plus
# the indoor-environment (variant A) metrics. Each result is one row in a CSV
# under artifacts/<run_id>/ and the partial values are saved as .npy.
#
# Goal of this file (single responsibility):
#   Metric error of every dimension measured by hand at collection time.
#   With L the real measure and L_hat the measure in the model, report
#   |L_hat - L| and the relative error e = |L_hat - L| / L.
#   Variant A: report the errors of the room dimensions in cm.
#
# What to implement:
#   - For each entry in measured_cm, read its L_hat from the scaled model.
#   - Compute abs_error_cm = |L_hat - L| and rel_error = abs_error_cm / L.
#
# Inputs / outputs:
#   Inputs: scaled model measures from results/<run_id>/.
#   Outputs under results/<run_id>/evaluation/: metric_errors.npy (K, 2):
#   abs_error_cm, rel_error. One row per dimension in
#   artifacts/<run_id>/metric_error.csv: name, value_cm, model_cm,
#   abs_error_cm, rel_error.
#
# Defaults (read from the YAML block evaluation.measured_cm, never hardcoded):
#   measured_cm: a list of {name, value_cm}
#
# Rules:
#   - Write code ONLY between the BEGIN and END markers below. Do not create
#     other regions or other files for this task.
#   - Configuration comes from the YAML loaded by main.py. No argparse.
#   - Save every partial result as .npy under results/<run_id>/<stage>/.
#   - Keep this block after implementing; update it if the behavior changes.
#   - Expose a function run(cfg: dict, run_dir: str, ...) that main.py can call
#     when the YAML block of this stage has enabled: true (default false).
#   - Pipeline already available: frames are extracted and quality-filtered, and
#     SIFT, ORB and AKAZE keypoints/descriptors are saved per frame under
#     results/<run_id>/detection/<detector>/<frame>/ as keypoints.npy (N, 6:
#     x, y, size, angle, response, octave) and descriptors.npy (N, D).
# ==============================================================================

# >>> BEGIN IMPLEMENTATION >>>

# <<< END IMPLEMENTATION <<<
