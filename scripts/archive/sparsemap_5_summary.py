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
job_id = 72#int(sys.argv[1])

# %%
in_dir = project_dir / "results"
# out_dir = project_dir / "results" / "data_map"
#
# if not os.path.exists(out_dir):
#     os.makedirs(out_dir)

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
# FULL PILOT BATCH: 10 pi_level, 6 traj for one mouse x 2 + 5 random trajectories (170 jobs, 16K walltime)
seed_list = np.arange(5)
rotation_id_list = np.arange(6)
pi_level_list = np.arange(0, 91, 10)

# get job lists
job_list_rand = [((0,seed,-1), pi_lev) for pi_lev in pi_level_list for seed in seed_list]
job_list_mouse_3 = [((1,3,rot_id), pi_lev) for pi_lev in pi_level_list for rot_id in rotation_id_list]
job_list_mouse_4 = [((1,4,rot_id), pi_lev) for pi_lev in pi_level_list for rot_id in rotation_id_list]

# pack
job_batch_optmap = {i+1: job for i, job in enumerate(job_list_rand + job_list_mouse_3 + job_list_mouse_4)}
len(job_batch_optmap)

# %%
# RANDOM MAP BATCH: 10 pi_level, 2 mice x 5 random maps + 5 random trajectories x 5 random maps (350 jobs)
traj_seed_list = np.arange(5)
map_seed_list = np.arange(5)
pi_level_list = np.arange(0, 91, 10)

# get job lists
job_list_rand = [((0,traj_seed,-1), pi_lev, map_seed) for pi_lev in pi_level_list for traj_seed in traj_seed_list for map_seed in map_seed_list]
job_list_mouse_3 = [((1,3,0), pi_lev, map_seed) for pi_lev in pi_level_list for map_seed in map_seed_list]
job_list_mouse_4 = [((1,4,0), pi_lev, map_seed) for pi_lev in pi_level_list for map_seed in map_seed_list]

# pack
job_batch_randmap = {i+1: job for i, job in enumerate(job_list_rand + job_list_mouse_3 + job_list_mouse_4)}
len(job_batch_randmap)

# %%
# WALL-ONLY MAP BATCH: 6 traj x 2 mice + 5 random traj (17 jobs, 16K walltime)
seed_list = np.arange(5)
rotation_id_list = np.arange(6)

# get job lists
job_list_rand = [(0,seed,-1) for seed in seed_list]
job_list_mouse_3 = [(1,3,rot_id) for rot_id in rotation_id_list]
job_list_mouse_4 = [(1,4,rot_id) for rot_id in rotation_id_list]

# pack
job_batch_wallmap = {i+1: job for i, job in enumerate(job_list_rand + job_list_mouse_3 + job_list_mouse_4)}
len(job_batch_wallmap)

# %% [markdown]
# ## LOCAL PARAMETERS

# %%
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

# %% [markdown]
# ## PREP
# ### setting up belief distribution propagation

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
# # check if the saved file exists
# dat_path = out_dir / f"dat_map_{job_batch[job_id]}"
# dat_exists = os.path.exists(dat_path)

# %%
# key, pi_level = job_batch[job_id]
# s_traj, a_traj, occ_map, s_top, xy_traj = dat_traj_dict[key]
# pi = pi_all[pi_level]

# %% [markdown]
# ## RUN: map optimization

# %%
# def run_map_optimizer(s_traj, a_traj, occ_map, s_top, pi, eid_removed, eid_remain, schedule):
#     '''
#     NOTE: deletion-based optimizer
#     '''
#     # mp
#     def run_mp(param):
#         # load
#         map, chunk_id, chunksize = param
        
#         # load chunk of trajectory
#         s_traj_r, a_traj_r = get_one_traj_chunk(s_traj, a_traj, chunk_id, chunksize)
        
#         # eval
#         score, score_map, pq_traj = eval_one_map(map, s_traj_r, a_traj_r, s_top, T_dict, pi, e_sh_arr, pq_mask_amb_dict, n_tiles, ringsize, n_edges, sh_dz_arr)
        
#         # # get p_map
#         # p_map = pq_traj.sum(-1).mean(0)
#         return score, score_map
    
#     # eval full map
#     score_max, _ = run_mp((eid_remain, None, None))
#     score_iter = [score_max]
    
#     # get initial candidate edges (from zeros of occ_map of full traj)
#     s_unoccupied = np.where(occ_map == 0)[0]
#     init_eid_cand = get_eid_from_s(s_unoccupied, e_sh_arr)
#     if len(s_unoccupied) > 0:
#         print(f"{len(init_eid_cand)} unoccupied edges found from full trajectory...")

#     # iter
#     for iter in np.arange(len(schedule)):
#         # load schedule
#         n_chunks, chunk_id, chunksize = schedule[iter]
        
#         # find candidate edges to remove
#         # if iter==0:
#         #     _, _, p_map = run_mp((eid_remain, chunk_id, chunksize))
#         for subiter in range(n_chunks):
#             s_traj_r, _ = get_one_traj_chunk(s_traj, a_traj, chunk_id, chunksize)
#             eid_cand = get_eid_cand_for_one_iter(eid_remain, init_eid_cand, s_traj_r, e_sh_arr, n_tiles)
#             if len(eid_cand) > 0:
#                 eid_cand_exists = True
#                 break
#             else:
#                 eid_cand_exists = False
#                 print(f"chunk {chunk_id} has no candidate edges to remove, trying next chunk...")
#                 chunk_id = (chunk_id + 1) % n_chunks # override until eid_cand is non-empty
#                 _, _ = run_mp((eid_remain, chunk_id, chunksize))
#         if not eid_cand_exists:
#             print(f"no candidate edges found in any chunk, using all remaining edges...")
#             eid_cand = eid_remain # if no candidate edges found, use all remaining edges
        
#         # eval candidates on one traj chunk
#         eid_remain_cand = [eid_remain[eid_remain!=x] for x in eid_cand]
#         param = [(x, chunk_id, chunksize) for x in eid_remain_cand]
#         with multiprocess.Pool() as p:
#             score_cand, _ = zip(*p.map(run_mp, param))
        
#         # find top
#         idx_top = np.argmax(score_cand)
#         eid_top = eid_cand[idx_top] # top to be removed
#         score_top = score_cand[idx_top]
#         eid_remain = eid_remain_cand[idx_top]
#         # p_map = p_map_cand[idx_top]

#         # append
#         eid_removed.append(eid_top)
#         score_iter.append(score_top)
        
#         # print
#         print(f"iter {iter}: eval {len(eid_cand)} edges, remove edge {eid_top}, score = {score_top:.6f}, chunk_id = {chunk_id}/{n_chunks}")
#     return score_iter, eid_removed

# def eval_multiple_maps(map_list, s_traj, a_traj, s_top, pi):
#     '''
#     NOTE: 
#     '''
#     def run_mp(map):
#         score, score_map, _ = eval_one_map(map, s_traj, a_traj, s_top, T_dict, pi, e_sh_arr, pq_mask_amb_dict, n_tiles, ringsize, n_edges, sh_dz_arr)
#         return score, score_map
    
#     # mp
#     with multiprocess.Pool() as p:
#         score_list, score_map_list = zip(*p.map(run_mp, map_list))
#     score_list = np.array(score_list)
#     return score_list, score_map_list

# %%
# ## RUN (100m)
# if not dat_exists:
# # if True:
#     # get optimization schedule
#     schedule = get_optimization_schedule(n_parallel, n_edges, s_traj)
#     schedule = {x:schedule[x] for x in range(60)} # TEST RUN
    
#     # initialization
#     eid_removed, eid_remain = initialize_optimization(n_edges)

#     # run optimization
#     score_iter, eid_removed = run_map_optimizer(s_traj, a_traj, occ_map, s_top, pi, eid_removed, eid_remain, schedule)

#     # eval all maps on full trajectory
#     print('evaluating all maps on full trajectory...')
#     map_iter = get_map_list_from_eid_removed(eid_removed, n_edges)
#     score_iter_full, score_map_iter_full = eval_multiple_maps(map_iter, s_traj, a_traj, s_top, pi)

# %% [markdown]
# ## PICKLE

# %%
# if not dat_exists:
# # if True:
#     dat_map = score_iter_full, score_map_iter_full, eid_removed
#     pickle.dump(dat_map, open(dat_path, "wb"))
    
# else:
#     score_iter_full, score_map_iter_full, eid_removed = pickle.load(open(dat_path, "rb"))

# %% [markdown]
# # TEST batch

# %% [markdown]
# ### scores, random map on random walk

# %%
# prep
job_selected = [x for x,((y,z,w),u,v) in job_batch_randmap.items() if y==0]
out_dir = project_dir / "results" / "data_randmap"
score_tensor = np.array([pickle.load(open(out_dir / f"dat_map_{job_batch_randmap[job_id]}", "rb"))[0] for job_id in job_selected]).reshape(10,5,5,-1)
# score_tensor_rw_rm = np.transpose(score_tensor, (1, 0, 2, 3)).reshape(10, 25, -1)[:,:,::-1]
score_tensor_rw_rm = score_tensor.reshape(10, 25, -1)[:,:,::-1]

# plot
plt.figure(figsize=(8,6), dpi=200)
plt.title('random maps for random walk')
for i,x in enumerate(score_tensor_rw_rm):
    labels = [f'PI={i*10}'] + ['']*24
    plt.plot(x.T, label=labels, color=colors[i])
for y,c in zip([.5,.6,.7], ['limegreen', 'gold', 'tomato']): 
    plt.axhline(y, color=c, linestyle='--')
plt.xlabel('map size (# edges)')
plt.ylabel('performance')
plt.legend()
plt.grid()

# %% [markdown]
# ### scores, random map on unrotated mouse

# %%
# prep
job_selected = [x for x,((y,z,w),u,v) in job_batch_randmap.items() if y==1]
out_dir = project_dir / "results" / "data_randmap"
score_tensor = np.array([pickle.load(open(out_dir / f"dat_map_{job_batch_randmap[job_id]}", "rb"))[0] for job_id in job_selected]).reshape(2,10,5,-1)
score_tensor_unrot_rm = np.transpose(score_tensor, (1, 0, 2, 3)).reshape(10, 10, -1)[:,:,::-1]

# plot
plt.figure(figsize=(8,6), dpi=200)
plt.title('random maps for unrotated mouse')
for i,x in enumerate(score_tensor_unrot_rm):
    labels = [f'PI={i*10}'] + ['']*9
    plt.plot(x.T, label=labels, color=colors[i])
for y,c in zip([.5,.6,.7], ['limegreen', 'gold', 'tomato']): 
    plt.axhline(y, color=c, linestyle='--')
plt.xlabel('map size (# edges)')
plt.ylabel('performance')
plt.legend()
plt.grid()

# %% [markdown]
# ### scores, random walk

# %%
# prep
job_selected = np.arange(1, 51)
out_dir = project_dir / "results" / "data_optmap"
score_tensor = np.array([pickle.load(open(out_dir / f"dat_map_{job_batch_optmap[job_id]}", "rb"))[0] for job_id in job_selected]).reshape(10,5,-1)
score_tensor_rw = score_tensor[:,:,::-1]

# plot
plt.figure(figsize=(8,6), dpi=200)
plt.title('optimized maps for random walk')
for i,x in enumerate(score_tensor_rw):
    labels = [f'PI={i*10}'] + ['']*4
    plt.plot(x.T, label=labels, color=colors[i])
for y,c in zip([.5,.6,.7], ['limegreen', 'gold', 'tomato']): 
    plt.axhline(y, color=c, linestyle='--')
plt.xlabel('map size (# edges)')
plt.ylabel('performance')
plt.legend()
plt.grid()

# %% [markdown]
# ### scores, mouse unrotated

# %%
# prep
job_selected = [x for x,((y,z,w),u) in job_batch_optmap.items() if w==0]
out_dir = project_dir / "results" / "data_optmap"
score_tensor = np.array([pickle.load(open(out_dir / f"dat_map_{job_batch_optmap[job_id]}", "rb"))[0] for job_id in job_selected]).reshape(2,10,-1)
score_tensor_unrot = np.transpose(score_tensor, (1,0,2))[:,:,::-1]

# plot
plt.figure(figsize=(8,6), dpi=200)
plt.title('optimized maps for unrotated mouse')
for i,x in enumerate(score_tensor_unrot):
    labels = [f'PI={i*10}'] + ['']*1
    plt.plot(x.T, label=labels, color=colors[i])
for y,c in zip([.5,.6,.7], ['limegreen', 'gold', 'tomato']): 
    plt.axhline(y, color=c, linestyle='--')
plt.xlabel('map size (# edges)')
plt.ylabel('performance')
plt.legend()
plt.grid()

# %% [markdown]
# ### scores, mouse rotated

# %%
# prep
job_selected = [x for x,((y,z,w),u) in job_batch_optmap.items() if w>0]
out_dir = project_dir / "results" / "data_optmap"
score_tensor = np.array([pickle.load(open(out_dir / f"dat_map_{job_batch_optmap[job_id]}", "rb"))[0] for job_id in job_selected]).reshape(2,10,5,-1)
score_tensor_rot = np.transpose(score_tensor, (1,0,2,3)).reshape(10,10,-1)[:,:,::-1]

# plot
plt.figure(figsize=(8,6), dpi=200)
plt.title('optimized maps for rotated mouse')
for i,x in enumerate(score_tensor_rot):
    labels = [f'PI={i*10}'] + ['']*9
    plt.plot(x.T, label=labels, color=colors[i])
for y,c in zip([.5,.6,.7], ['limegreen', 'gold', 'tomato']): 
    plt.axhline(y, color=c, linestyle='--')
plt.xlabel('map size (# edges)')
plt.ylabel('performance')
plt.legend()
plt.grid()

# %% [markdown]
# ### tradeoff curves

# %%
score_tar = .6

# %%
# prep: tradeoff curves
score_tensor_list = [score_tensor_unrot_rm, score_tensor_rw, score_tensor_rot, score_tensor_unrot]
# score_tensor_list = [score_tensor_rw_rm, score_tensor_rw, score_tensor_rot, score_tensor_unrot]
label_list = ['random maps, unrotated mouse', 'optimized maps, random walk', 'optimized maps, rotated mouse', 'optimized maps, unrotated mouse']
color_list = ['gray', 'k', 'b', 'r']

# prep: heatmap
score_arr = score_tensor_unrot.mean(1)

# plot
plt.figure(figsize=(8,5), dpi=200)
plt.title(f'target score= {score_tar}\nheatmap: average score for unrotated mouse')
plt.imshow(score_arr.T, aspect='auto', cmap='YlGnBu_r', origin='lower', vmin=0, vmax=1, interpolation='nearest', extent=[pi_level_list[0]-5, pi_level_list[-1]+5, 0, score_arr.shape[1]])
plt.colorbar()

for i, (score_tensor, label, color) in enumerate(zip(score_tensor_list, label_list, color_list)):
    mapsize_arr = np.argmin(np.abs(score_tensor-score_tar), axis=-1)
    mapsize_ci, mapsize_mean, _ = get_confidence_interval(mapsize_arr, axis=1)
    #
    for pi, ci0, ci1 in zip(pi_level_list, mapsize_mean-mapsize_ci, mapsize_mean+mapsize_ci):
        plt.plot([pi, pi], [ci0, ci1], color=color, lw=1)
    plt.scatter(pi_level_list, mapsize_mean-mapsize_ci, color=color, s=50, label=f'{label}', marker='_', lw=1)
    plt.scatter(pi_level_list, mapsize_mean+mapsize_ci, color=color, s=50, marker='_', lw=1)

# setting
plt.xlabel('PI fidelity')
plt.ylabel('map size (# edges)')
plt.legend()

# zoom 0
plt.ylim([0, 500])

# # zoom 1
# plt.xlim([15, 95])
# plt.ylim([0, 180])

# # zoom 2
# plt.xlim([75, 95])
# plt.ylim([0, 45])

# %% [markdown]
# ### normalized tradeoff curves

# %%
score_tar = .6

# %%
# prep: tradeoff curves
score_tensor_list = [score_tensor_rw, score_tensor_rot, score_tensor_unrot]
label_list = ['optimized maps, random walk', 'optimized maps, rotated mouse', 'optimized maps, unrotated mouse']
color_list = ['k', 'b', 'r']

# prep: heatmap
score_arr = score_tensor_unrot.mean(1)

# baseline
mapsize_arr_rw = np.argmin(np.abs(score_tensor_rw-score_tar), axis=-1)
mapsize_ci_rw, mapsize_mean_rw, _ = get_confidence_interval(mapsize_arr_rw, axis=1)

# plot
plt.figure(figsize=(10,4), dpi=200)
plt.suptitle(f'target score= {score_tar}; baseline: optimized maps for random walk')

plt.subplot(121)
for i, (score_tensor, label, color) in enumerate(zip(score_tensor_list, label_list, color_list)):
    mapsize_arr = np.argmin(np.abs(score_tensor-score_tar), axis=-1)
    mapsize_ci, mapsize_mean, _ = get_confidence_interval(mapsize_arr, axis=1)
    
    # override
    mapsize_mean -= mapsize_mean_rw
    
    # iter
    plt.plot(pi_level_list, mapsize_mean, color=color, lw=1, label=label)
    for pi, ci0, ci1 in zip(pi_level_list, mapsize_mean-mapsize_ci, mapsize_mean+mapsize_ci):
        plt.plot([pi, pi], [ci0, ci1], color=color, lw=1)
    plt.scatter(pi_level_list, mapsize_mean-mapsize_ci, color=color, s=50, marker='_', lw=1)
    plt.scatter(pi_level_list, mapsize_mean+mapsize_ci, color=color, s=50, marker='_', lw=1)
plt.axhline(0, color='gray', linestyle='--', lw=1)
plt.xlabel('PI fidelity')
plt.ylabel('map size - baseline (# edges)')
plt.legend()
plt.xlim([5, 95])
plt.ylim([-25, 5])
plt.grid(alpha=.2)

plt.subplot(122)
for i, (score_tensor, label, color) in enumerate(zip(score_tensor_list, label_list, color_list)):
    mapsize_arr = np.argmin(np.abs(score_tensor-score_tar), axis=-1)
    mapsize_ci, mapsize_mean, _ = get_confidence_interval(mapsize_arr, axis=1)
    
    # z-score
    zscore = (mapsize_mean - mapsize_mean_rw) / mapsize_mean_rw
    
    # uncertainty propagation
    # z_ci = np.sqrt((mapsize_ci / mapsize_mean_rw)**2 + (mapsize_mean * mapsize_ci_rw / mapsize_mean_rw**2)**2)
    
    z_ci = (1+zscore) * np.sqrt((mapsize_ci / mapsize_mean)**2 + (mapsize_ci_rw / mapsize_mean_rw)**2)
    
    # iter
    plt.plot(pi_level_list, zscore, color=color, lw=1, label=label)
    for pi, ci0, ci1 in zip(pi_level_list, zscore-z_ci, zscore+z_ci):
        plt.plot([pi, pi], [ci0, ci1], color=color, lw=1)
    plt.scatter(pi_level_list, zscore-z_ci, color=color, s=50, marker='_', lw=1)
    plt.scatter(pi_level_list, zscore+z_ci, color=color, s=50, marker='_', lw=1)
plt.axhline(0, color='gray', linestyle='--', lw=1)
plt.xlabel('PI fidelity')
plt.ylabel('zscore (% increase in map size)')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.legend()
plt.xlim([5, 95])
plt.ylim([-.25, .05])
plt.grid(alpha=.2)

# setting
plt.tight_layout()

# %% [markdown]
# ### compression rate (baseline: random map, unrotated)

# %%
# prep: tradeoff curves
score_tensor_list = [score_tensor_rw, score_tensor_rot, score_tensor_unrot]
label_list = ['optimized maps, random walk', 'optimized maps, rotated mouse', 'optimized maps, unrotated mouse']
color_list = ['k', 'b', 'r']

# baseline
mapsize_arr = np.argmin(np.abs(score_tensor_unrot_rm-score_tar), axis=-1)
mapsize_ci_base, mapsize_mean_base, _ = get_confidence_interval(mapsize_arr, axis=1)

# iter
cr_list, cr_ci_list = [], []
for score_tensor in score_tensor_list:
    # get mapsize for target score
    mapsize_arr = np.argmin(np.abs(score_tensor-score_tar), axis=-1)
    mapsize_ci, mapsize_mean, _ = get_confidence_interval(mapsize_arr, axis=1)
    
    # compute compression rate
    cr = mapsize_mean_base / mapsize_mean
    
    # uncertainty propagation
    cr_ci = cr * np.sqrt((mapsize_ci / mapsize_mean)**2 + (mapsize_ci_base / mapsize_mean_base)**2)
    
    # append
    cr_list.append(cr)
    cr_ci_list.append(cr_ci)
    
# plot
plt.figure(figsize=(7,5), dpi=200)
plt.title(f'target score= {score_tar}\nbaseline: random maps, unrotated mouse')
for i, (cr, cr_ci, label, color) in enumerate(zip(cr_list, cr_ci_list, label_list, color_list)):
    plt.plot(pi_level_list, cr, color=color, lw=1, label=label)
    for pi, ci0, ci1 in zip(pi_level_list, cr-cr_ci, cr+cr_ci):
        plt.plot([pi, pi], [ci0, ci1], color=color, lw=1)
    plt.scatter(pi_level_list, cr-cr_ci, color=color, s=50, marker='_', lw=1)
    plt.scatter(pi_level_list, cr+cr_ci, color=color, s=50, marker='_', lw=1)

# setting
plt.xlabel('PI fidelity')
plt.ylabel('compression rate (optimized / baseline)')
plt.legend()
plt.grid(alpha=.2)

# %% [markdown]
# ### compression rate (baseline: optimized map, random walk)

# %%
# prep: tradeoff curves
score_tensor_list = [score_tensor_rot, score_tensor_unrot]
label_list = ['optimized maps, rotated mouse', 'optimized maps, unrotated mouse']
color_list = ['b', 'r']

# baseline
mapsize_arr = np.argmin(np.abs(score_tensor_rw-score_tar), axis=-1)
mapsize_ci_base, mapsize_mean_base, _ = get_confidence_interval(mapsize_arr, axis=1)

# iter
cr_list, cr_ci_list = [], []
for score_tensor in score_tensor_list:
    # get mapsize for target score
    mapsize_arr = np.argmin(np.abs(score_tensor-score_tar), axis=-1)
    mapsize_ci, mapsize_mean, _ = get_confidence_interval(mapsize_arr, axis=1)
    
    # compute compression rate
    cr = mapsize_mean_base / mapsize_mean
    
    # uncertainty propagation
    cr_ci = cr * (mapsize_ci / mapsize_mean + mapsize_ci_base / mapsize_mean_base)
    
    # append
    cr_list.append(cr)
    cr_ci_list.append(cr_ci)
    
# plot
plt.figure(figsize=(7,5), dpi=200)
plt.title('baseline: random maps, unrotated mouse')
for i, (cr, cr_ci, label, color) in enumerate(zip(cr_list, cr_ci_list, label_list, color_list)):
    plt.plot(pi_level_list, cr, color=color, lw=1, label=label)
    for pi, ci0, ci1 in zip(pi_level_list, cr-cr_ci, cr+cr_ci):
        plt.plot([pi, pi], [ci0, ci1], color=color, lw=1)
    plt.scatter(pi_level_list, cr-cr_ci, color=color, s=50, marker='_', lw=1)
    plt.scatter(pi_level_list, cr+cr_ci, color=color, s=50, marker='_', lw=1)

# setting
plt.xlabel('PI fidelity')
plt.ylabel('compression rate (optimized / baseline)')
plt.legend()
plt.grid(alpha=.2)

# %% [markdown]
# ### wall-only map

# %%
# load
pi_level_list = np.arange(80, 91, 1)
out_dir = project_dir / "results" / "data_wallmap"
score_tensor_rw = np.array([pickle.load(open(out_dir / f"dat_map_{x}", "rb"))[0] for x in [job_batch_wallmap[x] for x in [1,2,3,4,5]]])
score_tensor_rot = np.array([pickle.load(open(out_dir / f"dat_map_{x}", "rb"))[0] for x in [job_batch_wallmap[x] for x in [7,8,9,10,11,13,14,15,16,17]]])
score_tensor_unrot = np.array([pickle.load(open(out_dir / f"dat_map_{x}", "rb"))[0] for x in [job_batch_wallmap[x] for x in [6,12]]])

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
