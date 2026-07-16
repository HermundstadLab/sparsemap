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
# - `maze` (height data)
#
# ## OUT_DATA
# - `dat_minimap_traj`

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

# %% [markdown]
# ## BASH PARAMETERS

# %%
in_dir = project_dir / "data"
out_dir = project_dir / "results"
#
if not os.path.exists(in_dir):
    os.makedirs(in_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# %% [markdown]
# ## LOAD

# %%
# load df
# df_raw = pickle.load(open(in_dir / "df_raw", "rb"))
# df_equal = pickle.load(open(in_dir / "df_equal", "rb"))
df_tile_mean = pickle.load(open(in_dir / "df_tile_mean", "rb"))
df_tile = pickle.load(open(in_dir / "df_tile", "rb"))
xy_pixel_all = df_tile[['x', 'y']].values
tile_id_all = df_tile.tile_id.values

# %%
# load maze dictionaries
hex_0_grid, hex_1_grid, s_z_dict, s_hex_dict, hex_a_dict, xy_s_dict, a_dict, sas_dict, ssa_dict, s_nbr_dict = load_maze_dicts(maze)

# %% [markdown]
# ## RUN: get `s_traj` and `a_traj`

# %% [markdown]
# ### get rotated trajectories for multiple mice

# %%
# param
mouse_id_all = [3, 4]
session_selected = [2]

# mp
theta_list = np.arange(0,360, 60) / 180 * np.pi
def run_mp(mouse_id):
    print(f'Processing mouse {mouse_id}')
    dat_traj_list = get_rotated_dat_traj_for_one_mouse(mouse_id, theta_list, session_selected, df_tile_mean, xy_pixel_all, tile_id_all, ssa_dict, n_tiles, in_dir, df_tile, d_tile)
    print('')
    return dat_traj_list

with multiprocess.Pool() as pool:
    dat_traj_list_all = pool.map(run_mp, mouse_id_all)

# pack to dat_traj_dict
# key: (type, mouse_id, rotation_id)
# value: s_traj, a_traj, occ_map, s_top, xy_traj
type_id = 1
rotation_id_list = np.arange(len(theta_list))
dat_traj_dict = dict()
for mouse_id, dat_traj_list in zip(mouse_id_all, dat_traj_list_all):
    dat_traj_dict_1 = dict(zip(product([type_id], [mouse_id], rotation_id_list), dat_traj_list))
    dat_traj_dict.update(dat_traj_dict_1)

# %% [markdown]
# ### get random tile walks

# %%
# generate random walk data to match mouse data on session 2 (5s)
# prep
selected_mouse_id = mouse_id_all[0]
init_s = dat_traj_dict[(1,selected_mouse_id,0)][0][0]
L_traj = int(np.mean([len(x[0]) for x in dat_traj_dict.values()]))

# iter
seed_list = np.arange(5)
dat_traj_list = []
for seed in seed_list:
    s_traj_rand, a_traj_rand = get_s_traj_from_random_walk(init_s, L_traj, sas_dict, seed=seed)
    occ_map_rand, s_top_rand = get_occ_map_from_s_traj(s_traj_rand, n_tiles)
    # append
    dat_traj = s_traj_rand, a_traj_rand, occ_map_rand, s_top_rand, None
    dat_traj_list.append(dat_traj)
    
# pack to dat_traj_dict
# key: (type, seed, -1)
# value: s_traj, a_traj, occ_map, s_top, xy_traj
type_id = 0
rotation_id_list = np.arange(len(theta_list))
for seed, dat_traj in zip(seed_list, dat_traj_list):
    dat_traj_dict_1 = dict(zip(product([type_id], [seed], [-1]), [dat_traj]))
    dat_traj_dict.update(dat_traj_dict_1)

# %%
dat_traj_dict.keys()

# %% [markdown]
# ## PICKLE

# %%
pickle.dump(dat_traj_dict, open(out_dir / "dat_traj_dict", "wb"))

# %% [markdown]
# ## TEST

# %% [markdown]
# ### plot rotated trajectories

# %%
# plot
plt.figure(figsize=(8,8), dpi=200)
for rot_id in [0,1]:
    s_traj, a_traj, occ_map, s_top, xy_traj = dat_traj_dict[(1,3,rot_id)]
    plt.plot(*xy_traj[:].T, lw=.1)
    plt.scatter(*xy_traj[:1].T, lw=1)
    plt.axis('equal')

# %% [markdown]
# ### hex maze from cartesian maze

# %%
# plot
plt.figure(figsize=(10,5), dpi=200)
plt.subplot(121)
plt.imshow(maze, cmap='Greys', origin='lower')
for (x,y), s in xy_s_dict.items():
    plt.text(x,y, str(s), color='r', fontsize=8, ha='center', va='center')
# plt.scatter(*np.array(xy_allowed).T)
plt.axis('off')
plt.axis('equal')

plt.subplot(122)
plt.scatter(hex_0_grid, hex_1_grid, c=maze, cmap='Greys', marker='h', s=650)
for (x,y), s in xy_s_dict.items():
    plt.text(hex_0_grid[y,x], hex_1_grid[y,x], str(s), color='r', fontsize=8, ha='center', va='center')
plt.axis('off')
plt.axis('equal')
plt.xlim([-3, 11])
# plt.ylim([1, 13])

plt.tight_layout()

# %% [markdown]
# ### plot y-flipped

# %%
# plot
plt.figure(figsize=(10,5), dpi=200)
plt.subplot(121)
plt.imshow(maze[::-1], cmap='Greys', origin='lower')
for (x,y), s in xy_s_dict.items():
    plt.text(x,-y+16, str(s), color='r', fontsize=8, ha='center', va='center')
# plt.scatter(*np.array(xy_allowed).T)
plt.axis('off')
plt.axis('equal')

plt.subplot(122)
plt.scatter(hex_0_grid, -hex_1_grid, c=maze, cmap='Greys', marker='h', s=650)
for (x,y), s in xy_s_dict.items():
    plt.text(hex_0_grid[y,x], -hex_1_grid[y,x], str(s), color='r', fontsize=8, ha='center', va='center')
plt.axis('off')
plt.axis('equal')
plt.xlim([-3, 11])
# plt.ylim([1, 13])

plt.tight_layout()

# %% [markdown]
# ### data from mouse 3, session 2 (day 3)

# %%
# prep
mouse_id, rot_id = 3, 0
s_traj, a_traj, occ_map, s_top, xy_traj = dat_traj_dict[(1,mouse_id,rot_id)]
s_count = [Counter(s_traj)[x] for x in range(n_tiles)]
hex_0_all, hex_1_all = np.array(list(s_hex_dict.values())).T
hex_0_all, hex_1_all = np.array(list(s_hex_dict.values())).T

# plot
plt.figure(figsize=(5,5.5), dpi=200)
plt.title(f'mouse {mouse_id}, sessions: {2}; top half occupied: {len(s_top)} tiles')

# map
plt.scatter(hex_0_all, hex_1_all, c=occ_map, cmap='PuRd', marker='h', s=800, vmin=0, edgecolor='none')
plt.scatter(hex_0_all[s_top], hex_1_all[s_top], c='none', marker='o', s=300, edgecolor='k')
for (x,y), s in xy_s_dict.items():
    plt.text(hex_0_grid[y,x], hex_1_grid[y,x], str(s), color='dimgray', fontsize=6, ha='center', va='center')

# setting
plt.axis('off')
plt.axis('equal')
plt.tight_layout()

# %% [markdown]
# ### self-avoiding random walk

# %%
# load
seed = 0
s_traj, a_traj, occ_map, s_top, xy_traj = dat_traj_dict[(0,seed,-1)]

# prep
s_count = [Counter(s_traj)[x] for x in range(n_tiles)]
hex_0_all, hex_1_all = np.array(list(s_hex_dict.values())).T
hex_0_all, hex_1_all = np.array(list(s_hex_dict.values())).T

# plot
plt.figure(figsize=(5,5.5), dpi=200)
plt.title(f'random walk ({seed}); top half occupied: {len(s_top)} tiles')
# map
plt.scatter(hex_0_all, hex_1_all, c=occ_map, cmap='PuRd', marker='h', s=800, vmin=0, edgecolor='none')
plt.scatter(hex_0_all[s_top], hex_1_all[s_top], c='none', marker='o', s=300, edgecolor='k')
for (x,y), s in xy_s_dict.items():
    plt.text(hex_0_grid[y,x], hex_1_grid[y,x], str(s), color='dimgray', fontsize=6, ha='center', va='center')

# setting
plt.axis('off')
plt.axis('equal')
plt.tight_layout()
