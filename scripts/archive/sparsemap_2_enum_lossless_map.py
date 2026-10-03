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
# - `dat_minimap_traj`
#
# ## OUT_DATA
# - `idx_masked_map_{job_id}`

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
# ## JOBS

# %%
# job_dict = get_job_dict_for_lossless_map_enum()
# len(job_dict)

# %%
# job_dict

# %% [markdown]
# ## BASH PARAMETERS

# %%
# 38 enum_maps = 2 trajectories x 19 PIs
job_id = 5#int(sys.argv[1])

# %%
mouse_id = 3

# %%
# tag = f"mouse_{mouse_id}"
in_dir = project_dir / "results"
out_dir = in_dir / "data_maps"
#
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# %% [markdown]
# ## LOCAL PARAMETERS

# %%
n_cand_per_iter = 20
n_failures_to_stop = 10
n_maps = 100

# %% [markdown]
# ## LOAD

# %%
# load df
# df_raw = pickle.load(open(in_dir / "df_raw", "rb"))
# df_equal = pickle.load(open(in_dir / "df_equal", "rb"))

# load tile trajectories
dat_traj_dict = pickle.load(open(in_dir / "dat_traj_dict", "rb"))

# %% [markdown]
# ## RUN SINGLE: eval on one (traj, pi, map)

# %%
# prep: maze dictionaries
hex_0_grid, hex_1_grid, s_z_dict, s_hex_dict, hex_a_dict, xy_s_dict, a_dict, sas_dict, ssa_dict, s_nbr_dict = load_maze_dicts(maze)

# prep: map
sh_dz_arr = get_sh_dz_arr(s_z_dict, sas_dict)
e_sh_arr = get_e_sh_arr(n_tiles, ringsize)

# prep: pi & tile transition tensor
pi_all = get_p_vonmises_for_enumPI()
T = get_T_tensor(sas_dict, ringsize, n_tiles)
T_dict = get_T_tensor_dict(T, ringsize)

# %%
# load traj
s_traj, a_traj, occ_map, s_top, _ = dat_traj_dict[(1, 3, 0)]
# s_traj = np.array([37,50,63,76,89,102,115,127,128,129,130,131])
# a_traj = np.array([-1,1,1,1,1,1,1,1,0,0,0,0])

# load pi
idx_pi = 90
pi = pi_all[idx_pi]
sap_arr = get_sap_arr(pi_all[100], sas_dict, ringsize, n_tiles) # pq_prop_v1

# load map
np.random.seed(42)
map = np.random.permutation(n_edges)[:30]

# prep: pq_mask from map
pq_mask_amb_dict = get_pq_mask_dict_for_ambiguous_edges(sh_dz_arr)
pq_mask_unknown_dict = get_pq_mask_dict_for_unknown_edges(map, e_sh_arr, n_tiles, ringsize, n_edges)
pq_mask_dict = get_pq_mask_dict(pq_mask_unknown_dict, pq_mask_amb_dict)

# %%
# pq_prop
pq_traj = pq_prop_for_n_steps(s_traj, a_traj, T_dict, pi, pq_mask_dict, n_tiles, ringsize, sh_dz_arr)

# pq_prop_v1
# p_traj, q_traj = pq_prop_for_n_steps_v1(s_traj, a_traj, sap_arr, pi, pq_mask_dict, n_tiles, ringsize, sh_dz_arr)

# %% [markdown]
# ## TEST SINGLE

# %% [markdown]
# ### pq_traj

# %%
plt.figure(figsize=(10,4), dpi=200)
plt.subplot(1,2,1)
plt.imshow(pq_traj.sum(-1)[:100], aspect='auto')
plt.subplot(1,2,2)
plt.imshow(pq_traj.sum(1)[:100], aspect='auto')

# %%
# prep
s_count = [Counter(s_traj)[x] for x in range(n_tiles)]
hex_0_all, hex_1_all = np.array(list(s_hex_dict.values())).T
hex_0_all, hex_1_all = np.array(list(s_hex_dict.values())).T

# plot
plt.figure(figsize=(8,6), dpi=200)
i = 0
for t in range(0,12):
    i+=1
    plt.subplot(3,4,i)
    # plt.title(f'random walk ({seed}); top half occupied: {len(s_top)} tiles')
    # map
    s = s_traj[t]
    plt.scatter(hex_0_all, hex_1_all, c=pq_traj[t].sum(-1), cmap='PuRd', marker='h', s=100, vmin=0, edgecolor='none')
    plt.scatter([hex_0_all[s]], [hex_1_all[s]], s=50, edgecolor='k', facecolor='none')
    # for (x,y), s in xy_s_dict.items():
    #     plt.text(hex_0_grid[y,x], hex_1_grid[y,x], str(s), color='dimgray', fontsize=6, ha='center', va='center')

    # setting
    plt.axis('off')
    plt.axis('equal')
plt.tight_layout()

# %% [markdown]
# ### score map

# %%
# load
score_map, score = get_score_map(s_traj, pq_traj.sum(-1), s_top)

# prep
s_count = [Counter(s_traj)[x] for x in range(n_tiles)]
hex_0_all, hex_1_all = np.array(list(s_hex_dict.values())).T
hex_0_all, hex_1_all = np.array(list(s_hex_dict.values())).T

# plot
plt.figure(figsize=(10,5.5), dpi=200)

plt.subplot(121)
plt.title(f'score = {score:.6f}')
plt.scatter(hex_0_all, hex_1_all, c=score_map, cmap='PuRd', marker='h', s=800, vmin=0, vmax=1, edgecolor='none')
# plot s_top in circular markers
plt.scatter(hex_0_all[s_top], hex_1_all[s_top], s=300, edgecolor='k', facecolor='none', marker='o', linewidth=1.5)
for (x,y), s in xy_s_dict.items():
    plt.text(hex_0_grid[y,x], hex_1_grid[y,x], str(s), color='dimgray', fontsize=6, ha='center', va='center')
# setting
plt.axis('off')
plt.axis('equal')
plt.tight_layout()

plt.subplot(122)
plt.title('score traj')
plt.plot(pq_traj.sum(-1).max(1))

plt.tight_layout()

# %% [markdown]
# ## RUN: finding multiple lossless maps

# %%
# # load job
# type_id, seed, idx_pi = job_dict[job_id]
# s_traj, a_traj, occ_map, s_top = dat_traj_dict[type_id][seed]
# pi = pi_all[idx_pi]

# %% [markdown]
# ### step 1) find maximum score with full map

# %%
# MP functions
sap_arr = get_sap_arr(pi, sas_dict, ringsize, n_tiles)
def get_score_for_one_map(idx_masked):
    # load
    ring_arr_masked = get_ring_arr_masked(idx_masked, ring_arr, edge_id_arr)
    
    # eval
    p_traj = eval_one_map(ring_arr_masked, ring_arr, sap_arr, s_traj, a_traj)
    score_map, score_traj = get_score_map(s_traj, p_traj)
    score = score_map[s_top].mean()
    return score, score_map.mean()

# run
score_max, _ = get_score_for_one_map([])
score_max = np.floor(score_max * 1000) / 1000
print(f'score_max = {score_max}')


# %% [markdown]
# ### step 2) find multiple lossless (zero cost) maps

# %%
def find_one_lossless_map(seed):
    # load
    n_segments = len(edge_id_arr)
    
    # initialize
    idx_masked = []
    idx_remain = list(range(n_segments))
    
    # iter
    np.random.seed(seed)
    n_failures = 0
    for iter in range(n_segments):
        if len(idx_remain) >= n_cand_per_iter:
            idx_cand = np.random.choice(idx_remain, n_cand_per_iter)
        else:
            idx_cand = idx_remain
            
        # iter until finding zero cost
        is_zero_cost = False
        idx_masked_cand = [idx_masked + [x] for x in idx_cand]
        for jter, idx_ in enumerate(idx_masked_cand):
            score_, _ = get_score_for_one_map(idx_)
            if score_ >= score_max:
                idx_select = idx_[-1]
                # print(f'iter: {iter} hit at', jter, score_)
                is_zero_cost = True
                break
        
        # append
        if is_zero_cost:
            idx_masked.append(idx_select)
            idx_remain.remove(idx_select)
        else:
            n_failures += 1
            # print('no hit, n_failures', n_failures)
            if n_failures >= n_failures_to_stop:
                print(f'seed {seed} is stopping due to n_failures ={n_failures_to_stop}')
                break
        
        # print
        # print(f"iter {iter}: remove segm {idx_select}, score = {score_}")
        
        if len(idx_remain)==0: break
    return idx_masked


# %%
# run (1h)
with multiprocess.Pool() as p:
    idx_masked_all = p.map(find_one_lossless_map, range(n_maps))


# %% [markdown]
# ### step 3) sort maps with linear regression

# %%
def sort_maps_with_linear_regression(idx_masked_all):
    '''note: score is idential for all maps to numeric precision'''
    # get score_map
    n_removal_all = np.array([len(x) for x in idx_masked_all])
    with multiprocess.Pool() as p:
        score_all, score_map_all = zip(*p.map(get_score_for_one_map, idx_masked_all))
    
    # linear regression
    X = n_removal_all.reshape(-1, 1)
    y = np.array(score_map_all)
    y_pred = LinearRegression().fit(X, y).predict(X)
    
    # sort by residual
    idx_sort = np.argsort(y - y_pred)[::-1]
    idx_masked_sort = np.array(idx_masked_all, dtype=object)[idx_sort]
    return idx_masked_sort


# %%
idx_masked_sort = sort_maps_with_linear_regression(idx_masked_all)

# %% [markdown]
# ## PICKLE

# %%
pickle.dump(idx_masked_sort, open(out_dir / f"lossless_maps_{job_id}", "wb"))

# %% [markdown]
# ## TEST

# %%
idx_masked_sort = pickle.load(open(out_dir / f"lossless_maps_{5}", "rb"))
np.array([len(x) for x in idx_masked_sort]).mean()

# %%
rank = 0
edge_id_masked = idx_masked_sort[rank]

# prep: eval
ring_arr_masked = get_ring_arr_masked(edge_id_masked, ring_arr, edge_id_arr)
sap_arr = get_sap_arr(pi, sas_dict, ringsize, n_tiles)
p_traj = eval_one_map(ring_arr_masked, ring_arr, sap_arr, s_traj, a_traj)
score_map, score_traj = get_score_map(s_traj, p_traj)
score_top = score_map[s_top].mean()
hex_0_all, hex_1_all = np.array(list(s_hex_dict.values())).T

# prep: map
z_all = list(s_z_dict.values())
hex_0_all, hex_1_all = np.array(list(s_hex_dict.values())).T
a0_all, a1_all = np.array(list(hex_a_dict)).T
hex_a0_all = hex_0_all[:,None] + a0_all[None]/2
hex_a1_all = hex_1_all[:,None] + a1_all[None]/2
ring_hex_0_all = hex_0_all[:,None] + a0_all[None]/4
ring_hex_1_all = hex_1_all[:,None] + a1_all[None]/4

# plot
plt.figure(figsize=(11,5), dpi=200)
plt.suptitle(f"map {job_id} ({rank}) on traj {(type_id, seed)},   PI:{idx_pi}/100,   sensory:{930 - len(edge_id_masked)}/930,   performance = {score_top:.3f}")


## plot map
plt.subplot(121)
plt.scatter(ring_hex_0_all, ring_hex_1_all, c=ring_arr_masked, cmap='coolwarm', marker='o', s=20, zorder=1, vmin=-4, vmax=4)
plt.scatter(hex_0_all, hex_1_all, c=z_all, cmap='Greys', marker='h', s=600, zorder=0, alpha=.5)
for (x,y), s in xy_s_dict.items():
    plt.text(hex_0_grid[y,x], hex_1_grid[y,x], str(s), color='dimgray', fontsize=6, ha='center', va='center')
plt.axis('off')
plt.axis('equal')

# plot performance
plt.subplot(122)
plt.scatter(hex_0_all, hex_1_all, c=score_map, cmap='PuRd', marker='h', s=600, vmin=0)
plt.colorbar()
for (x,y), s in xy_s_dict.items():
    plt.text(hex_0_grid[y,x], hex_1_grid[y,x], str(s), color='dimgray', fontsize=6, ha='center', va='center')

# setting
plt.axis('off')
plt.axis('equal')
plt.tight_layout()
