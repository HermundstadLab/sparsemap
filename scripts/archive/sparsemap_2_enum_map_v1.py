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
pq_mask_amb_dict = get_pq_mask_dict_for_ambiguous_edges(sh_dz_arr)

# prep: pi & tile transition tensor
pi_all = get_p_vonmises_for_enumPI()
T = get_T_tensor(sas_dict, ringsize, n_tiles)
T_dict = get_T_tensor_dict(T, ringsize)

# %%
# load traj
s_traj, a_traj, occ_map, s_top, _ = dat_traj_dict[(1, 3, 0)]
# s_traj, a_traj, occ_map, s_top, _ = dat_traj_dict[(0, 3, -1)]
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
# pq_mask_dict = get_pq_mask_dict_for_one_map(map, e_sh_arr, pq_mask_amb_dict, n_tiles, ringsize, n_edges)
s_pq_mask_dict = get_s_pq_mask_dict_for_one_map(map, e_sh_arr, pq_mask_amb_dict, n_tiles, ringsize, n_edges, sh_dz_arr)

# %%
# pq_prop
pq_traj = pq_prop_for_n_steps(s_traj, a_traj, T_dict, pi, s_pq_mask_dict, n_tiles, ringsize, sh_dz_arr)
score_map, score = get_score_map(s_traj, pq_traj.sum(-1), s_top)

# pq_prop_v1
# p_traj, q_traj = pq_prop_for_n_steps_v1(s_traj, a_traj, sap_arr, pi, pq_mask_dict, n_tiles, ringsize, sh_dz_arr)

# print
score

# %%
count, _ = np.histogram(s_traj, bins=np.arange(n_tiles+1))


# %%
(count/np.sum(count))[s_top].sum()

# %%
set0 = set(s_top)

# %%
t = 10000
count_1 = np.histogram(s_traj[t:t+len(s_traj)//10], bins=np.arange(n_tiles+1))[0]
set1 = set(np.argsort(count_1/np.sum(count_1))[::-1][:77])
len(set0.intersection(set1))

# %%
p0 = np.histogram(s_traj, bins=np.arange(n_tiles+1), density=True)[0]

# %%
t = 10000
p1 = np.histogram(s_traj[t:t+len(s_traj)//10], bins=np.arange(n_tiles+1), density=True)[0]

# %%
plt.plot(p0)
plt.plot(p1)

# %%
count

# %%
plt.plot(Counter(s_traj))

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

# %% [markdown]
# ## todo: just enum lossy map to simplify coding

# %%
# # load job
# type_id, seed, idx_pi = job_dict[job_id]
# s_traj, a_traj, occ_map, s_top = dat_traj_dict[type_id][seed]
# pi = pi_all[idx_pi]

# %% [markdown]
# ### step 1) find maximum score with full map

# %%
# # MP functions
# sap_arr = get_sap_arr(pi, sas_dict, ringsize, n_tiles)
# def get_score_for_one_map(idx_masked):
#     # load
#     ring_arr_masked = get_ring_arr_masked(idx_masked, ring_arr, edge_id_arr)
    
#     # eval
#     p_traj = eval_one_map(ring_arr_masked, ring_arr, sap_arr, s_traj, a_traj)
#     score_map, score_traj = get_score_map(s_traj, p_traj)
#     score = score_map[s_top].mean()
#     return score, score_map.mean()

# # run
# score_max, _ = get_score_for_one_map([])
# score_max = np.floor(score_max * 1000) / 1000
# print(f'score_max = {score_max}')

# %% [markdown]
# ### step 2) find multiple lossless (zero cost) maps

# %%
# def find_one_lossless_map(seed):
#     # load
#     n_segments = len(edge_id_arr)
    
#     # initialize
#     idx_masked = []
#     idx_remain = list(range(n_segments))
    
#     # iter
#     np.random.seed(seed)
#     n_failures = 0
#     for iter in range(n_segments):
#         if len(idx_remain) >= n_cand_per_iter:
#             idx_cand = np.random.choice(idx_remain, n_cand_per_iter)
#         else:
#             idx_cand = idx_remain
            
#         # iter until finding zero cost
#         is_zero_cost = False
#         idx_masked_cand = [idx_masked + [x] for x in idx_cand]
#         for jter, idx_ in enumerate(idx_masked_cand):
#             score_, _ = get_score_for_one_map(idx_)
#             if score_ >= score_max:
#                 idx_select = idx_[-1]
#                 # print(f'iter: {iter} hit at', jter, score_)
#                 is_zero_cost = True
#                 break
        
#         # append
#         if is_zero_cost:
#             idx_masked.append(idx_select)
#             idx_remain.remove(idx_select)
#         else:
#             n_failures += 1
#             # print('no hit, n_failures', n_failures)
#             if n_failures >= n_failures_to_stop:
#                 print(f'seed {seed} is stopping due to n_failures ={n_failures_to_stop}')
#                 break
        
#         # print
#         # print(f"iter {iter}: remove segm {idx_select}, score = {score_}")
        
#         if len(idx_remain)==0: break
#     return idx_masked

# %%
# # run (1h)
# with multiprocess.Pool() as p:
#     idx_masked_all = p.map(find_one_lossless_map, range(n_maps))

# %% [markdown]
# ### step 3) sort maps with linear regression

# %%
# def sort_maps_with_linear_regression(idx_masked_all):
#     '''note: score is idential for all maps to numeric precision'''
#     # get score_map
#     n_removal_all = np.array([len(x) for x in idx_masked_all])
#     with multiprocess.Pool() as p:
#         score_all, score_map_all = zip(*p.map(get_score_for_one_map, idx_masked_all))
    
#     # linear regression
#     X = n_removal_all.reshape(-1, 1)
#     y = np.array(score_map_all)
#     y_pred = LinearRegression().fit(X, y).predict(X)
    
#     # sort by residual
#     idx_sort = np.argsort(y - y_pred)[::-1]
#     idx_masked_sort = np.array(idx_masked_all, dtype=object)[idx_sort]
#     return idx_masked_sort

# %%
# idx_masked_sort = sort_maps_with_linear_regression(idx_masked_all)

# %% [markdown]
# ## PICKLE

# %%
# pickle.dump(idx_masked_sort, open(out_dir / f"lossless_maps_{job_id}", "wb"))

# %% [markdown]
# ## RUN: finding multiple lossless maps

# %%
# job param
type_id, id_1, id_2 = 1, 3, 2
pi_level = 90

# mp
s_traj, a_traj, occ_map, s_top, _ = dat_traj_dict[(type_id, id_1, id_2)]
pi = pi_all[pi_level]
def eval_one_map(map):
    # get pq_mask
    s_pq_mask_dict = get_s_pq_mask_dict_for_one_map(map, e_sh_arr, pq_mask_amb_dict, n_tiles, ringsize, n_edges, sh_dz_arr)
    
    # eval
    pq_traj = pq_prop_for_n_steps(s_traj, a_traj, T_dict, pi, s_pq_mask_dict, n_tiles, ringsize, sh_dz_arr)
    score_map, score = get_score_map(s_traj, pq_traj.sum(-1), s_top)
    score = np.around(score, 6)
    return score, score_map


# %%
# # load traj
# s_traj, a_traj, occ_map, s_top, _ = dat_traj_dict[(1, 3, 2)]
# # s_traj, a_traj, occ_map, s_top, _ = dat_traj_dict[(0, 3, -1)]
# # s_traj = np.array([37,50,63,76,89,102,115,127,128,129,130,131])
# # a_traj = np.array([-1,1,1,1,1,1,1,1,0,0,0,0])

# # load pi
# idx_pi = 90
# pi = pi_all[idx_pi]
# sap_arr = get_sap_arr(pi_all[100], sas_dict, ringsize, n_tiles) # pq_prop_v1

# # load map
# np.random.seed(42)
# map = np.random.permutation(n_edges)[:30]

# # prep: pq_mask from map
# pq_mask_unknown_dict = get_pq_mask_dict_for_unknown_edges(map, e_sh_arr, n_tiles, ringsize, n_edges)
# pq_mask_dict = get_pq_mask_dict(pq_mask_unknown_dict, pq_mask_amb_dict)

# %%
# load map
np.random.seed(42)
map = np.random.permutation(n_edges)[:80]

# %%
score, score_map = eval_one_map(map)
score

# %%
plt.plot([10,20,30,50,100], [81,83,90,131,280])
plt.ylim(0,None)

# %%
pq0 = np.zeros([n_tiles, ringsize])
pq0[s_traj[0], 0] = 1
pq_mask = s_pq_mask_dict[77]

# %%
# s_pq_mask_dict = dict()
# for s in range(n_tiles):
#     s_pq_mask = 1
#     for h in range(ringsize):
#         dz = sh_dz_arr[s, h]
#         s_pq_mask = s_pq_mask * pq_mask_dict[(dz, h)]
#     s_pq_mask_dict[s] = s_pq_mask

# %%
# %%timeit
pq1 = pq0 * pq_mask
# pq1 / pq1.sum()

# %%
# %%timeit
collapse_pq_one_step(pq0, s_traj[0], s_pq_mask_dict, ringsize, sh_dz_arr)

# %%
# %%timeit
collapse_pq_one_step(pq0, s_traj[0], s_pq_mask_dict, ringsize, sh_dz_arr)

# %%
# get score_max
score_max, score_map_max = eval_one_map(np.arange(n_edges))

# initialize
eid_removed = []
eid_remain = np.argsort(occ_map[e_sh_arr[:,0]]) # all edges sorted from least to most occupied (by tile occupancy)
score_iter, score_map_iter = [score_max], [score_map_max]

# iter
for iter in range(n_edges):
    # if len(eid_remain) >= n_cand_per_iter:
    #     eid_cand = np.random.choice(eid_remain, n_cand_per_iter)
    # else:
    #     eid_cand = eid_remain
    if score_iter[-1] == score_max:
        n_cand_per_iter = 100
    else:
        n_cand_per_iter = 100
    eid_cand = eid_remain[:n_cand_per_iter]
    
    # eval candidates
    eid_removed_cand = [eid_removed + [x] for x in eid_cand]
    eid_remain_cand = [eid_remain[eid_remain!=x] for x in eid_cand]
    with multiprocess.Pool() as p:
        score_cand, score_map_cand = zip(*p.map(eval_one_map, eid_remain_cand))
    
    # append top 
    idx_top = np.argmax(score_cand)
    eid_select = eid_cand[idx_top]
    eid_remain = eid_remain_cand[idx_top]
    # eid_select = eid_remain[np.random.choice(np.where(np.array(score_cand) == max(score_cand))[0])]
    # append
    eid_removed.append(eid_select)
    score_iter.append(score_cand[idx_top])
    score_map_iter.append(score_map_cand[idx_top])
    # eid_remain = eid_remain[eid_remain != eid_select]
    
    # print
    print(f"iter {iter}: remove edge {eid_select}, score = {score_iter[-1]}, n_cand_per_iter = {n_cand_per_iter}")
    
    if len(eid_remain)==0: break

# %%
dat_test = score_iter, score_map_iter, eid_removed
pickle.dump(dat_test, open(out_dir / "dat_test_2", "wb"))

# %%
score_iter, score_map_iter, eid_removed = pickle.load(open(out_dir / "dat_test", "rb"))

# %%
np.unique(np.around(score_iter, 6))

# %%
plt.plot(score_iter[::-1][100:1000])
plt.ylim(.9,1.1)

# %% [markdown]
# ### step 1) find best lossy maps

# %%
# MP functions
sap_arr = get_sap_arr(pi, sas_dict, ringsize, n_tiles)
def get_score_for_one_map(idx_masked):
    # load
    ring_arr_masked = get_ring_arr_masked(idx_masked, ring_arr, segm_idx_arr)
    
    # eval
    p_traj = eval_one_map(ring_arr_masked, ring_arr, sap_arr, s_traj, a_traj)
    score_map, score_traj = get_score_map(s_traj, p_traj)
    score = score_map[s_top].mean()
    return score

def find_one_lossy_map(seed=42):
    # load
    n_segments = len(segm_idx_arr)
    
    # initialize
    idx_masked = idx_masked_init.copy()
    idx_remain = [x for x in range(n_segments) if x not in idx_masked]
    
    # iter
    np.random.seed(seed)
    for iter in range(n_segments):
        if len(idx_remain) >= n_cand_per_iter:
            idx_cand = np.random.choice(idx_remain, n_cand_per_iter)
        else:
            idx_cand = idx_remain
        
        # eval candidates
        idx_masked_cand = [idx_masked + [x] for x in idx_cand]
        with multiprocess.Pool() as p:
            score_cand = p.map(get_score_for_one_map, idx_masked_cand)
        
        # append top 
        # idx_select = idx_cand[np.argmax(score_cand)]
        idx_select = idx_cand[np.random.choice(np.where(np.array(score_cand) == max(score_cand))[0])]
        idx_masked.append(idx_select)
        idx_remain.remove(idx_select)
        
        # print
        print(f"iter {iter}: remove segm {idx_select}, score = {max(score_cand)}")
        
        if len(idx_remain)==0: break
    return idx_masked


# %%
# run (20m)
idx_masked = find_one_lossy_map()

# %% [markdown]
# ### step 2) eval lossy maps

# %%
# eval (15s)
idx_masked_eval = [idx_masked[:x] for x in range(len(idx_masked)+1)]
with multiprocess.Pool() as p:
    score_eval = p.map(get_score_for_one_map, idx_masked_eval)
sensory_eval = [len(segm_idx_arr) - len(x) for x in idx_masked_eval]

# %% [markdown]
# ## PICKLE

# %%
dat_lossy_maps = idx_masked, sensory_eval, score_eval
pickle.dump(dat_lossy_maps, open(out_dir / f"dat_lossy_maps_{job_id}", "wb"))

# %% [markdown]
# ## TEST

# %%
# plot
plt.figure(figsize=(6,4), dpi=200)
plt.title(f"map {job_id_lossless} ({init_map_rank}) on traj {(type_id, seed)},   PI:{idx_pi}/100")

plt.plot(sensory_eval, score_eval, color='r')
#
plt.axhline(.8, color='dimgray', linestyle='--', lw=1)
plt.axhline(.2, color='dimgray', linestyle='--', lw=1)
plt.xlim([0,600])
plt.ylim([0,1])
# plt.legend()
plt.xlabel("sensory level")
plt.ylabel("performance")
