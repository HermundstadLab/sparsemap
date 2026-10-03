# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: sparsemap
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## IN_DATA
# - `dat_traj`
#
# ## OUT_DATA
# - `dat_map_{job}`

# %% [markdown]
# ## VERSION

# %% [markdown]
# ## PACKAGE

# %%
# %load_ext autoreload
# %autoreload 2

from src.utils import *
from src.metadata import *
from src.sparsemap import *

warnings.filterwarnings("ignore", category=RuntimeWarning)

# %%
# FULL BATCH
seed_list = np.arange(n_seeds_traj)
rotation_id_list = np.arange(6)

# get job lists
job_list_rand = [(0,seed,-1) for seed in seed_list]

job_list_all_mice = []
for mouse_id in mouse_ids:
    job_list_mouse = [(1,mouse_id,rot_id) for rot_id in rotation_id_list]
    job_list_all_mice.extend(job_list_mouse)

# pack
job_batch_wallmap = {i+1: job for i, job in enumerate(job_list_rand + job_list_all_mice)}

# print
len(job_batch_wallmap)

# %%
job_batch_wallmap

# %%
np.arange(n_seeds_traj+1, len(job_batch_wallmap)+1)

# %%
# load
pi_level_list = np.arange(80, 91, 1)
out_dir = project_dir / "results" / "data_wallmap"
score_tensor_rw = np.array([pickle.load(open(out_dir / f"dat_map_{x}", "rb"))[0] for x in [job_batch_wallmap[x] for x in range(1,n_seeds_traj+1)]])
score_tensor_unrot = np.array([pickle.load(open(out_dir / f"dat_map_{x}", "rb"))[0] for x in [job_batch_wallmap[x] for x in [11,17,23,29,35,41,47]]])
score_tensor_rot = np.array([pickle.load(open(out_dir / f"dat_map_{x}", "rb"))[0] for x in [job_batch_wallmap[x] for x in [12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52]]])

# prep
score_tensor_list = [score_tensor_rw, score_tensor_rot, score_tensor_unrot]
label_list = ['optimized maps, random walk', 'optimized maps, rotated mouse', 'optimized maps, unrotated mouse']
color_list = ['k', 'b', 'r']

# plot
plt.figure(figsize=(6, 4), dpi=200)
plt.title('wall-only maps')
for score_tensor, label, color in zip(score_tensor_list, label_list, color_list):
    # labels = [label] + ['']*(score_tensor.shape[0]-1)
    score_ci, score_mean, _ = get_confidence_interval(score_tensor, axis=0)
    plt.plot(pi_level_list, score_mean, label=label, color=color)
    for pi, ci0, ci1 in zip(pi_level_list, score_mean-score_ci, score_mean+score_ci):
        plt.plot([pi, pi], [ci0, ci1], color=color, lw=1)
    plt.scatter(pi_level_list, score_mean-score_ci, color=color, s=50, marker='_', lw=1)
    plt.scatter(pi_level_list, score_mean+score_ci, color=color, s=50, marker='_', lw=1)
plt.xlabel('PI fidelity')
plt.ylabel('performance')
plt.legend()
plt.grid()
