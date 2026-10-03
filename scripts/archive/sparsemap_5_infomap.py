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
out_dir = project_dir / "results" / 'dat_infomap'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# %% [markdown]
# ## LOCAL PARAMETERS

# %%
method = 'nearest_neighbor_limit' # 'score_map', 'nearest_neighbor_limit'

# %% [markdown]
# ## LOAD

# %%
# load tile trajectories
dat_traj_dict = pickle.load(open(project_dir / "results" / "dat_traj_dict", "rb"))

# %% [markdown]
# ## SPECIFY JOB BATCHES

# %%
# PILOT OPTIMAL MAP BATCH: 10 pi_level, 6 traj for one mouse x 2 + 5 random trajectories (170 jobs, 16K walltime)
seed_list = np.arange(5)
rotation_id_list = np.arange(6)
pi_level_list = np.arange(0, 91, 10)

# get job lists
job_list_rand = [((0,seed,-1), pi_lev) for pi_lev in pi_level_list for seed in seed_list]
job_list_mouse_3 = [((1,3,rot_id), pi_lev) for pi_lev in pi_level_list for rot_id in rotation_id_list]
job_list_mouse_4 = [((1,4,rot_id), pi_lev) for pi_lev in pi_level_list for rot_id in rotation_id_list]

# pack
job_batch_optmap_pilot = {i+1: job for i, job in enumerate(job_list_rand + job_list_mouse_3 + job_list_mouse_4)}
len(job_batch_optmap_pilot)

# %%
# PILOT RANDOM MAP BATCH: 10 pi_level, 2 mice x 5 random maps + 5 random trajectories x 5 random maps (350 jobs)
traj_seed_list = np.arange(5)
map_seed_list = np.arange(5)
pi_level_list = np.arange(0, 91, 10)

# get job lists
job_list_rand = [((0,traj_seed,-1), pi_lev, map_seed) for pi_lev in pi_level_list for traj_seed in traj_seed_list for map_seed in map_seed_list]
job_list_mouse_3 = [((1,3,0), pi_lev, map_seed) for pi_lev in pi_level_list for map_seed in map_seed_list]
job_list_mouse_4 = [((1,4,0), pi_lev, map_seed) for pi_lev in pi_level_list for map_seed in map_seed_list]

# pack
job_batch_randmap_pilot = {i+1: job for i, job in enumerate(job_list_rand + job_list_mouse_3 + job_list_mouse_4)}
len(job_batch_randmap_pilot)

# %%
# FULL OPTIMAL MAP BATCH
seed_list = np.arange(n_seeds_traj)
rotation_id_list = np.arange(6)
pi_level_list = np.arange(0, 91, 10)

# get job lists
job_list_rand = [((0,seed,-1), pi_lev) for pi_lev in pi_level_list for seed in seed_list]

job_list_all_mice = []
for mouse_id in mouse_ids:
    job_list_mouse = [((1,mouse_id,rot_id), pi_lev) for pi_lev in pi_level_list for rot_id in rotation_id_list]
    job_list_all_mice.extend(job_list_mouse)

# pack
job_batch_optmap_full = {i+1: job for i, job in enumerate(job_list_rand + job_list_all_mice)}

# print
len(job_batch_optmap_full)

# %%
# FULL RANDOM MAP BATCH
traj_seed_list = np.arange(n_seeds_traj)
map_seed_list = np.arange(n_seeds_map)
pi_level_list = np.arange(0, 91, 10)

# get job lists
job_list_rand = [((0,traj_seed,-1), pi_lev, map_seed) for pi_lev in pi_level_list for traj_seed in traj_seed_list for map_seed in map_seed_list]
#
job_list_all_mice = []
for mouse_id in mouse_ids:
    job_list_mouse = [((1,mouse_id,0), pi_lev, map_seed) for pi_lev in pi_level_list for map_seed in map_seed_list]
    job_list_all_mice.extend(job_list_mouse)

# pack
job_batch_randmap_full = {i+1: job for i, job in enumerate(job_list_rand + job_list_all_mice)}
len(job_batch_randmap_full)

# %% [markdown]
# ## JOB BATCH TO RUN

# %%
job_batch_optmap = job_batch_optmap_full
job_batch_randmap = job_batch_randmap_full

# %% [markdown]
# ## PREP
# ### setting up belief distribution propagation

# %%
# prep: maze dictionaries
hex_0_grid, hex_1_grid, s_z_dict, s_hex_dict, hex_a_dict, xy_s_dict, a_dict, sas_dict, ssa_dict, s_nbr_dict = load_maze_dicts(maze)
xy_tile_all = np.array([(hex_0_grid[y,x], hex_1_grid[y,x]) for (x,y), s in xy_s_dict.items()])

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
# ## RUN SINGLE

# %%
load_dir = project_dir / "results" / "data_optmap"
job_id = 93 # 105,46; 99,41; 93,36; 87,31
mapsize = 67  # 13,14; 39,41; 68,71

# load job
job = job_batch_optmap[job_id]

# %%
# load_dir = project_dir / "results" / "data_randmap"
# job_id = 286 # 286, 176, 181
# mapsize = 67

# # load job
# job = job_batch_randmap[job_id]

# %%
# load one job
s_traj, a_traj, occ_map, s_top, xy_traj = dat_traj_dict[job[0]]
map_iter, mapsize_iter, score_map_iter, score_iter = load_maps_from_one_job(load_dir, job, n_edges)

# load one map so that score≈0.7
map, score_map, score, n_edges_map = load_one_map(mapsize, map_iter, mapsize_iter, score_iter, score_map_iter, e_sh_arr, sh_dz_arr)

# get diffusion radius
sigma_tile = get_sigma_tile(score_map, xy_tile_all, method=method)

# get tile locality, ambiguity, identifiability
L_mat, A_mat, I_mat_local = get_locality_ambiguity_identifiability(map, sigma_tile, xy_tile_all, sh_dz_arr, e_sh_arr)

# get infomap
infomap, info_mean = get_infomap(L_mat, A_mat, I_mat_local, s_top, n_edges_map, mapsize, score_map, method=method)

# print
info_mean

# %% [markdown]
# ## TEST SINGLE

# %%
# # prep
hex_0_all, hex_1_all = np.array(list(s_hex_dict.values())).T
hex_0_all, hex_1_all = np.array(list(s_hex_dict.values())).T

# plot
plt.figure(figsize=(6.3,5.2), dpi=200)

# map
plt.scatter(hex_0_all, hex_1_all, marker='h', c='lightgray', s=900, edgecolor='none', alpha=.15)

plt.scatter(hex_0_all, hex_1_all, c=infomap, cmap='PuRd', marker='o', s=300*n_edges_map, vmin=0, vmax=1, edgecolor='none')
# plt.colorbar()

for (x,y), s in xy_s_dict.items():
    plt.text(hex_0_grid[y,x], hex_1_grid[y,x], str(s), color='dimgray', fontsize=6, ha='center', va='center')

# setting
plt.axis('off')
plt.axis('equal')
plt.tight_layout()


# %% [markdown]
# ## RUN ALL

# %%
def run_mp(para):
    # load
    job_id, label = para
    if label == "optmap":
        load_dir = project_dir / "results" / "data_optmap"
        job = job_batch_optmap[job_id]
    elif label == "randmap":
        load_dir = project_dir / "results" / "data_randmap"
        job = job_batch_randmap[job_id]
    
    # load one job
    s_traj, a_traj, occ_map, s_top, xy_traj = dat_traj_dict[job[0]]
    map_iter, mapsize_iter, score_map_iter, score_iter = load_maps_from_one_job(load_dir, job, n_edges)
    
    # iter
    infomap_list = []
    info_mean_list = []
    for mapsize in mapsize_iter:
        # load one map so that score≈0.7
        map, score_map, score, n_edges_map = load_one_map(mapsize, map_iter, mapsize_iter, score_iter, score_map_iter, e_sh_arr, sh_dz_arr)
        
        # get diffusion radius
        sigma_tile = get_sigma_tile(score_map, xy_tile_all, method=method)

        # get tile locality, ambiguity, identifiability
        L_mat, A_mat, I_mat_local = get_locality_ambiguity_identifiability(map, sigma_tile, xy_tile_all, sh_dz_arr, e_sh_arr)

        # get infomap
        infomap, info_mean = get_infomap(L_mat, A_mat, I_mat_local, s_top, n_edges_map, mapsize, score_map, method=method)
        
        # append
        infomap_list.append(infomap)
        info_mean_list.append(info_mean)
    return mapsize_iter, infomap_list, info_mean_list


# %% [markdown]
# ### optmap

# %%
# run (4m30s for 170jobs; 14m for 520 jobs)
if method == 'score_map':
    if not os.path.exists(out_dir / 'dat_infomap_optmap'):
        param = [(job_id, 'optmap') for job_id in list(job_batch_optmap)]
        with multiprocess.Pool() as pool:
            mapsize_iter_all, infomap_all, info_mean_all = zip(*pool.map(run_mp, param))
            
        # pack
        mapsize_iter = mapsize_iter_all[0]
        infomap_all = np.array(infomap_all)
        info_mean_all = np.array(info_mean_all)
        dat_infomap_optmap = mapsize_iter, infomap_all, info_mean_all

# %% [markdown]
# ### randmap

# %%
# run (9m for 350jobs; 44m for 1700 jobs)
if method == 'score_map':
    if not os.path.exists(out_dir / 'dat_infomap_randmap'):
        param = [(job_id, 'randmap') for job_id in list(job_batch_randmap)]
        with multiprocess.Pool() as pool:
            mapsize_iter_all, infomap_all, info_mean_all = zip(*pool.map(run_mp, param))

        # pack
        mapsize_iter = mapsize_iter_all[0]
        infomap_all = np.array(infomap_all)
        info_mean_all = np.array(info_mean_all)
        dat_infomap_randmap = mapsize_iter, infomap_all, info_mean_all

# %% [markdown]
# ### optmap, nearest neighbor limit

# %%
# run (4m30s for 170jobs)
if method == 'nearest_neighbor_limit':
    if not os.path.exists(out_dir / 'dat_infomap_optmap_nn'):
        param = [(job_id, 'optmap') for job_id in list(job_batch_optmap)]
        with multiprocess.Pool() as pool:
            mapsize_iter_all, infomap_all, info_mean_all = zip(*pool.map(run_mp, param))
            
        # pack
        mapsize_iter = mapsize_iter_all[0]
        infomap_all = np.array(infomap_all)
        info_mean_all = np.array(info_mean_all)
        dat_infomap_optmap_nn = mapsize_iter, infomap_all, info_mean_all

# %% [markdown]
# ### randmap, nearest neighbor limit

# %%
# run (9m for 350jobs)
if method == 'nearest_neighbor_limit':
    if not os.path.exists(out_dir / 'dat_infomap_randmap_nn'):
        param = [(job_id, 'randmap') for job_id in list(job_batch_randmap)]
        with multiprocess.Pool() as pool:
            mapsize_iter_all, infomap_all, info_mean_all = zip(*pool.map(run_mp, param))

        # pack
        mapsize_iter = mapsize_iter_all[0]
        infomap_all = np.array(infomap_all)
        info_mean_all = np.array(info_mean_all)
        dat_infomap_randmap_nn = mapsize_iter, infomap_all, info_mean_all

# %% [markdown]
# ## PICKLE

# %%
if method == 'score_map':
    if not os.path.exists(out_dir / 'dat_infomap_optmap'):
        pickle.dump(dat_infomap_optmap, open(out_dir / 'dat_infomap_optmap', "wb"))
    else:
        dat_infomap_optmap = pickle.load(open(out_dir / 'dat_infomap_optmap', "rb"))

# %%
if method == 'score_map':
    if not os.path.exists(out_dir / 'dat_infomap_randmap'):
        pickle.dump(dat_infomap_randmap, open(out_dir / 'dat_infomap_randmap', "wb"))
    else:
        dat_infomap_randmap = pickle.load(open(out_dir / 'dat_infomap_randmap', "rb"))

# %%
if method == 'nearest_neighbor_limit':
    if not os.path.exists(out_dir / 'dat_infomap_optmap_nn'):
        pickle.dump(dat_infomap_optmap_nn, open(out_dir / 'dat_infomap_optmap_nn', "wb"))
    else:
        dat_infomap_optmap_nn = pickle.load(open(out_dir / 'dat_infomap_optmap_nn', "rb"))

# %%
if method == 'nearest_neighbor_limit':
    if not os.path.exists(out_dir / 'dat_infomap_randmap_nn'):
        pickle.dump(dat_infomap_randmap_nn, open(out_dir / 'dat_infomap_randmap_nn', "wb"))
    else:
        dat_infomap_randmap_nn = pickle.load(open(out_dir / 'dat_infomap_randmap_nn', "rb"))
