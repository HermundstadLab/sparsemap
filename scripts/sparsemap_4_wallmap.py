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

# %% [markdown]
# ## BASH PARAMETERS

# %%
job_id = 17#int(sys.argv[1])

# %%
in_dir = project_dir / "results"
out_dir = project_dir / "results" / "data_wallmap"
#
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# %%
# load tile trajectories
dat_traj_dict = pickle.load(open(in_dir / "dat_traj_dict", "rb"))

# %% [markdown]
# ## SPECIFY ONE JOB BATCH
# 1. Select a subset of `dat_traj_dict` with a `pi_level` as a batch of jobs to run on cluster
# 2. the key for reading `dat_traj_dict` is `(type_id, id_1, id_2)` to output the value `(s_traj, a_traj, occ_map, s_top, xy_traj)`
# 3. for loading a mouse trajectory, use:
#     - `type_id = 1`
#     - `id_1` = 3 or 4 or ... (mouse_id)
#     - `id_2` = 0 or 1 ... or 5 (rotation_id)
# 4. for loading a random trajectory, use:
#     - `type_id = 0`
#     - `id_1` = 0 or 1 or ... (seed)
#     - `id_2` = -1 (-1 to specify n/a)
# 5. load a job as a nested tuple `((type_id, id_1, id_2), pi_level)`

# %%
# WALL-ONLY MAP BATCH: 6 traj x 2 mice + 5 random traj (17 jobs, 16K walltime)
seed_list = np.arange(5)
rotation_id_list = np.arange(6)

# get job lists
job_list_rand = [(0,seed,-1) for seed in seed_list]
job_list_mouse_3 = [(1,3,rot_id) for rot_id in rotation_id_list]
job_list_mouse_4 = [(1,4,rot_id) for rot_id in rotation_id_list]

# pack
job_batch = {i+1: job for i, job in enumerate(job_list_rand + job_list_mouse_3 + job_list_mouse_4)}
len(job_batch)

# %% [markdown]
# ## LOCAL PARAMETERS

# %%
pi_level_list = np.arange(80, 91, 1)


# %% [markdown]
# ## PREP
# ### setting up belief distribution propagation

# %%
def get_T_tensor(sas_dict, ringsize, n_tiles):
    """permutation matrix for (s,a) --> (s',a) given tile s and heading h"""
    T = np.zeros([n_tiles, ringsize, n_tiles, ringsize])  # p,q --> p',q'
    for s in range(n_tiles):
        for a in range(ringsize):
            s1 = sas_dict.get((s, a), -1)
            if s1 == -1:
                # T[s, a, :, a] = 1 / n_tiles
                T[s, a, :, a] = 0 # for wallmap
            else:
                T[s, a, s1, a] = 1.0
    return T


# %%
# prep: maze dictionaries
hex_0_grid, hex_1_grid, s_z_dict, s_hex_dict, hex_a_dict, xy_s_dict, a_dict, sas_dict, ssa_dict, s_nbr_dict = load_maze_dicts(maze)

# prep: map
e_sh_arr = get_e_sh_arr(n_tiles, ringsize)
eid_wall, sh_wall, n_edges_wall = get_eid_wall(sas_dict, ringsize, n_tiles, e_sh_arr)
sh_dz_arr = get_sh_dz_arr(s_z_dict, sas_dict, sh_wall)
pq_mask_amb_dict = get_pq_mask_dict_for_ambiguous_edges(sh_dz_arr)

# prep: pi & tile transition tensor
pi_all = get_p_vonmises_for_enumPI()
T = get_T_tensor(sas_dict, ringsize, n_tiles)
T_dict = get_T_tensor_dict(T, ringsize)

# %% [markdown]
# ## LOAD JOB

# %%
# check if the saved file exists
dat_path = out_dir / f"dat_map_{job_batch[job_id]}"
dat_exists = os.path.exists(dat_path)

# %%
key = job_batch[job_id]
s_traj, a_traj, occ_map, s_top, xy_traj = dat_traj_dict[key]
# pi = pi_all[pi_level]

# %% [markdown]
# ## RUN: eval random maps

# %%
def eval_multiple_maps(pi_list, s_traj, a_traj, s_top, map=eid_wall):
    '''
    NOTE: 
    '''
    def run_mp(pi_level):
        # load
        pi = pi_all[pi_level]
        
        # iter
        score, score_map, _ = eval_one_map(map, s_traj, a_traj, s_top, T_dict, pi, e_sh_arr, pq_mask_amb_dict, n_tiles, ringsize, n_edges, sh_dz_arr)
        return score, score_map

    # mp
    with multiprocess.Pool() as p:
        score_list, score_map_list = zip(*p.map(run_mp, pi_list))
    score_list = np.array(score_list)
    return score_list, score_map_list


# %%
## RUN (4m)
if not dat_exists:
    # # get random map
    # print(f'eval on job={job_batch[job_id]}')
    # eid_removed = sample_one_randmap(n_edges, map_seed)
    
    # eval all maps on full trajectory
    print(f'evaluating all maps on full trajectory...')
    # map_iter = get_map_list_from_eid_removed(eid_removed, n_edges)
    score_list, score_map_list = eval_multiple_maps(pi_level_list, s_traj, a_traj, s_top, eid_wall)

# %% [markdown]
# ## PICKLE

# %%
if not dat_exists:
# if True:
    dat_map = score_list, score_map_list
    pickle.dump(dat_map, open(dat_path, "wb"))
    
else:
    score_iter_full, score_map_iter_full, eid_removed = pickle.load(open(dat_path, "rb"))

# %% [markdown]
# ## TEST

# %%
# plot
plt.figure(figsize=(6,4), dpi=200)
plt.title(f"job: {job_batch[job_id]}")
plt.plot(pi_level_list, score_list)
plt.xlabel('PI fidelity')
plt.ylabel('localization score')
plt.ylim([-.1,1.1])
plt.grid()

if not os.path.exists(out_dir / 'plot'):
    os.makedirs(out_dir / 'plot')
plt.savefig(out_dir / 'plot' / f"map_score_{job_batch[job_id]}.png")

# %% [markdown]
# ## RERUN

# %%
np.where(~np.array([os.path.exists(out_dir / f"dat_map_{y}") for x,y in job_batch.items()]))[0]+1
