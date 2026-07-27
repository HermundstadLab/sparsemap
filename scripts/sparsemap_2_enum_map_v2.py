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
job_id = 6#int(sys.argv[1])

# %%
in_dir = project_dir / "results"
out_dir = project_dir / "results" / "data_map"
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
# # TEST PILOT BATCH: 1 pi_level, 6 traj for one mouse + 5 random trajectories (11K)
# pi_level = 80
# job_batch = {
#     1: ((0,0,-1), pi_level), 
#     2: ((0,1,-1), pi_level), 
#     3: ((0,2,-1), pi_level), 
#     4: ((0,3,-1), pi_level), 
#     5: ((0,4,-1), pi_level),
#     6: ((1,3,0), pi_level), 
#     7: ((1,3,1), pi_level), 
#     8: ((1,3,2), pi_level), 
#     9: ((1,3,3), pi_level), 
#     10: ((1,3,4), pi_level), 
#     11: ((1,3,5), pi_level),
#     12: ((1,4,0), pi_level),
#     13: ((1,4,1), pi_level),
#     14: ((1,4,2), pi_level),
#     15: ((1,4,3), pi_level),
#     16: ((1,4,4), pi_level),
#     17: ((1,4,5), pi_level),
# }
# len(job_batch)

# %%
# FULL PILOT BATCH: 10 pi_level, 6 traj for one mouse + 5 random trajectories (70K)
mouse_id = 3
seed_list = np.arange(5)
rotation_id_list = np.arange(6)
pi_level_list = np.arange(0, 91, 10)

# get job lists
job_list_rand = [((0,seed,-1), pi_lev) for pi_lev in pi_level_list for seed in seed_list]
job_list_mouse = [((1,mouse_id,rot_id), pi_lev) for pi_lev in pi_level_list for rot_id in rotation_id_list]

# pack
job_batch = {i+1: job for i, job in enumerate(job_list_rand + job_list_mouse)}
len(job_batch)

# %% [markdown]
# ## LOCAL PARAMETERS

# %%
debug = False
n_parallel = 25

# %% [markdown]
# ## PREP
# ### setting up belief distribution propagation

# %%
# prep: maze dictionaries
hex_0_grid, hex_1_grid, s_z_dict, s_hex_dict, hex_a_dict, xy_s_dict, a_dict, sas_dict, ssa_dict, s_nbr_dict = load_maze_dicts(maze)

# prep: map
sh_dz_arr = get_sh_dz_arr(s_z_dict, sas_dict)
e_sh_arr = get_e_sh_arr(n_tiles, ringsize)
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
key, pi_level = job_batch[job_id]
s_traj, a_traj, occ_map, s_top, xy_traj = dat_traj_dict[key]
pi = pi_all[pi_level]


# %% [markdown]
# ## RUN: map optimization

# %% [markdown]
# ### DEV: improving optimizer
# 1. issue: short traj chunk might not cover some s_top, which could result an immature deletion.
# 2. sol: 
#     - set lower priority to delete unoccupied edges (not even evaluating them?)
#     - delete a highly occupied edge when tie (softmax deletion? try hard deletion first)

# %%
def run_map_optimizer(s_traj, a_traj, s_top, pi, schedule):
    '''
    NOTE: 
    '''
    # mp
    def run_mp(param):
        # load
        map, chunk_id, chunksize = param
        
        # load chunk of trajectory
        if chunk_id is None:
            s_traj_r, a_traj_r = s_traj, a_traj
        else:
            s_traj_r = s_traj[chunk_id*chunksize:(chunk_id+1)*chunksize]
            a_traj_r = a_traj[chunk_id*chunksize:(chunk_id+1)*chunksize]
        
        # eval
        score, score_map = eval_one_map(map, s_traj_r, a_traj_r, s_top, T_dict, pi, e_sh_arr, pq_mask_amb_dict, n_tiles, ringsize, n_edges, sh_dz_arr)
        return score, score_map
    
    # eval full map
    score_max, _ = run_mp((np.arange(n_edges), None, None))

    # initialize
    eid_removed = []
    eid_remain = np.arange(n_edges)
    # eid_remain = np.argsort(occ_map[e_sh_arr[:,0]]) # all edges sorted from least to most occupied (by tile occupancy)
    score_iter = [score_max]

    # iter
    np.random.seed(42)
    for iter, (n_chunks, chunk_id, chunksize) in schedule.items():
        # randomize to avoid selecting same tiles when there is a tie
        np.random.shuffle(eid_remain)
              
        # eval candidates on one traj chunk
        eid_remain_cand = [eid_remain[eid_remain!=x] for x in eid_remain]
        param = [(x, chunk_id, chunksize) for x in eid_remain_cand]
        with multiprocess.Pool() as p:
            score_cand, _ = zip(*p.map(run_mp, param))
        
        # find top
        idx_top = np.argmax(score_cand)
        eid_top = eid_remain[idx_top] # top to be removed
        score_top = score_cand[idx_top]
        eid_remain = eid_remain_cand[idx_top]

        # append
        eid_removed.append(eid_top)
        score_iter.append(score_top)
        
        # print
        print(f"iter {iter}: remove edge {eid_top}, score = {score_top:.6f}, n_chunks = {n_chunks}, chunk_id = {chunk_id}")
    return score_iter, eid_removed

def eval_multiple_maps(map_list, s_traj, a_traj, s_top, pi):
    '''
    NOTE: 
    '''
    def run_mp(map):
        score, score_map = eval_one_map(map, s_traj, a_traj, s_top, T_dict, pi, e_sh_arr, pq_mask_amb_dict, n_tiles, ringsize, n_edges, sh_dz_arr)
        return score, score_map
    
    # mp
    with multiprocess.Pool() as p:
        score_list, score_map_list = zip(*p.map(run_mp, map_list))
    score_list = np.array(score_list)
    return score_list, score_map_list


# %%
## RUN (100m)
# if not dat_exists:
if True:
    # get optimization schedule
    schedule = get_optimization_schedule(n_parallel, n_edges, s_traj)
    # schedule = {x:schedule[x] for x in range(5)} # TEST RUN

    # run optimization
    score_iter, eid_removed = run_map_optimizer(s_traj, a_traj, s_top, pi, schedule)

    # eval all maps on full trajectory
    print('evaluating all maps on full trajectory...')
    map_iter = get_map_list_from_eid_removed(eid_removed, n_edges)
    score_iter_full, score_map_iter_full = eval_multiple_maps(map_iter, s_traj, a_traj, s_top, pi)

# %% [markdown]
# ## PICKLE

# %%
if not dat_exists:
    dat_map = score_iter_full, score_map_iter_full, eid_removed
    pickle.dump(dat_map, open(dat_path, "wb"))
    
else:
    score_iter_full, score_map_iter_full, eid_removed = pickle.load(open(dat_path, "rb"))

# %% [markdown]
# ## TEST

# %%
n_plot = 500

# plot
map_iter = get_map_list_from_eid_removed(eid_removed, n_edges)
mapsize_iter = [len(x) for x in map_iter]
plt.figure(figsize=(6,4), dpi=200)
plt.title(f"job: {job_batch[job_id]}")
plt.plot(mapsize_iter[::-1][:n_plot], score_iter_full[::-1][:n_plot])
plt.xlabel('map size (# edges)')
plt.ylabel('localization score')
plt.ylim([-.1,1.1])
plt.grid()

if not os.path.exists(out_dir / 'plot'):
    os.makedirs(out_dir / 'plot')
plt.savefig(out_dir / 'plot' / f"map_score_{job_batch[job_id]}.png")

# %% [markdown]
# # TEST batch

# %% [markdown]
# ### test pilot (two mice)

# %%
job_selected = [1,2,3,4,5] + [7,8,9,10,11,13,14,15,16,17] + [6,12]
n_plot = 200

# plot
mapsize_iter = np.arange(n_edges, -1, -1)
colors = ['tab:gray']*5 + ['tab:blue']*10 + ['red']*2
plt.figure(figsize=(8,5), dpi=200)
plt.title('gray: randwalk, blue: rotated mouse, red: unrotated mouse')
for i, job_id in enumerate(job_selected):
    dat_path = out_dir / f"dat_map_{job_batch[job_id]}"
    score_iter_full, score_map_iter_full, eid_removed = pickle.load(open(dat_path, "rb"))
    plt.plot(mapsize_iter[::-1][:n_plot], score_iter_full[::-1][:n_plot], label=f'{job_batch[job_id]}', lw=1, color=colors[i])
plt.xlabel('map size (# edges)')
plt.ylabel('localization score')
plt.ylim([-.1,1.1])
plt.grid()
plt.legend(fontsize=9)

# if not os.path.exists(out_dir / 'plot'):
    # os.makedirs(out_dir / 'plot')
# plt.savefig(out_dir / 'plot' / f"map_score_{job_batch[job_id]}.png")

# %% [markdown]
# ### full pilot (mouse 3)

# %%
job_id = 1
score_iter_full, score_map_iter_full, eid_removed = pickle.load(open(out_dir / f"dat_map_{job_batch[job_id]}", "rb"))

# %%
job_selected = np.arange(1, 51)
score_tensor = np.array([pickle.load(open(out_dir / f"dat_map_{job_batch[job_id]}", "rb"))[0] for job_id in job_selected]).reshape(10,5,-1)
score_arr = score_tensor.mean(1)[:,::-1]
mapsize_50 = [np.argmin(np.abs(x-.5)) for x in score_arr]
mapsize_60 = [np.argmin(np.abs(x-.6)) for x in score_arr]
mapsize_70 = [np.argmin(np.abs(x-.7)) for x in score_arr]

# %%
plt.figure(figsize=(8,5), dpi=200)
plt.title('optimized maps for random walk; color: performance')
plt.imshow(score_arr.T, aspect='auto', cmap='PuRd', origin='lower', vmin=0, vmax=1, interpolation='nearest')
plt.colorbar()
plt.scatter(np.arange(10), mapsize_50, color='limegreen', s=10, label='performance=50%')
plt.scatter(np.arange(10), mapsize_60, color='gold', s=10, label='performance=60%')
plt.scatter(np.arange(10), mapsize_70, color='tomato', s=10, label='performance=70%')
plt.ylim([0, 400])
plt.xlabel('PI fidelity')
plt.ylabel('map size (# edges)')
plt.legend()

# %%
plt.plot(score_arr[9,:], label=f'PI={i*10}')
plt.ylim(.995,1.001)

# %%
plt.figure(figsize=(8,6), dpi=200)
plt.title('optimized maps for random walk')
for i,x in enumerate(score_arr[:,:]): plt.plot(x, label=f'PI={i*10}')
for y,c in zip([.5,.6,.7], ['limegreen', 'gold', 'tomato']): 
    plt.axhline(y, color=c, linestyle='--')
plt.xlabel('map size (# edges)')
plt.ylabel('performance')
plt.legend()
plt.grid()

# %%
plt.plot(score_tensor[9,:,::-1][:,:].T)

# %%
np.arange(50).reshape(10,5)

# %%
score_iter_arr.reshape(5,10,-1)

# %%
plt.imshow(score_iter_arr)

# %%
n_plot = 200

# plot
mapsize_iter = np.arange(n_edges, -1, -1)
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
plt.figure(figsize=(8,5), dpi=200)
plt.title('pilot')
for i, job_id in enumerate(job_selected):
    dat_path = out_dir / f"dat_map_{job_batch[job_id]}"
    score_iter_full, score_map_iter_full, eid_removed = pickle.load(open(dat_path, "rb"))
    plt.plot(mapsize_iter[::-1][:n_plot], score_iter_full[::-1][:n_plot], label=f'{job_batch[job_id]}', lw=1, color=colors[i])
plt.xlabel('map size (# edges)')
plt.ylabel('localization score')
plt.ylim([-.1,1.1])
plt.grid()
plt.legend(fontsize=9)

# if not os.path.exists(out_dir / 'plot'):
    # os.makedirs(out_dir / 'plot')
# plt.savefig(out_dir / 'plot' / f"map_score_{job_batch[job_id]}.png")
