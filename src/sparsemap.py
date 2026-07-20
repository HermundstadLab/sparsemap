from src.utils import *
from src.metadata import *

# from src.equal import get_one_tile_id


## LOAD MAZE, s,a --> s1
def get_one_tile_id(xy, xy_pixel_all, tile_id_all):
    idx_pixel = ((xy_pixel_all - xy) ** 2).sum(1).argmin()
    tile_id = tile_id_all[idx_pixel]
    return tile_id


def load_maze_dicts(maze):
    # convert cartesian to hex coordinates
    v0, v1 = np.array([1, 0]), np.array([-1 / 2, 3**0.5 / 2])
    x_grid, y_grid = np.meshgrid(range(maze.shape[1]), range(maze.shape[0]))
    hex_0_grid, hex_1_grid = (
        x_grid[None] * v0[:, None, None] + y_grid[None] * v1[:, None, None]
    )

    # pack to dictionaries
    ij_xy_dict = {
        (i, j): (x_grid[i, j], y_grid[i, j])
        for i, j in product(range(maze.shape[1]), range(maze.shape[0]))
    }
    ij_hex_dict = {
        (i, j): (hex_0_grid[i, j], hex_1_grid[i, j])
        for i, j in product(range(maze.shape[1]), range(maze.shape[0]))
    }

    # get state-action transitions
    xy_allowed = [xy for ij, xy in ij_xy_dict.items() if maze[ij] != -1]
    xy_s_dict = {xy: i for i, xy in enumerate(xy_allowed)}
    a_dict = {
        (1, 0): 0,
        (1, 1): 1,
        (0, 1): 2,
        (-1, 0): 3,
        (-1, -1): 4,
        (0, -1): 5,
    }  # use right skewed hex grid
    sas_dict = {}
    for (x, y), s in xy_s_dict.items():
        sas_ = {
            (s, a): xy_s_dict[(x + ax, y + ay)]
            for (ax, ay), a in a_dict.items()
            if (x + ax, y + ay) in xy_allowed
        }
        sas_dict.update(sas_)
    ssa_dict = {(s, s1): a for (s, a), s1 in sas_dict.items()}

    # get s_nbr_dict
    s_nbr_dict = {}
    for (s0, a), s1 in sas_dict.items():
        try:
            s_nbr_dict[s0] += [s1]
        except:
            s_nbr_dict[s0] = [s1]

    # pack to dictionaries
    xy_ij_dict = {y: x for x, y in ij_xy_dict.items()}
    s_xy_dict = {y: x for x, y in xy_s_dict.items()}
    s_ij_dict = {s: xy_ij_dict[xy] for s, xy in s_xy_dict.items()}
    s_z_dict = {s: maze[ij] for s, ij in s_ij_dict.items()}
    s_hex_dict = {s: (ij_hex_dict[ij]) for s, ij in s_ij_dict.items()}
    hex_a_dict = {tuple(ax * v0 + ay * v1): a for (ax, ay), a in a_dict.items()}
    return (
        hex_0_grid,
        hex_1_grid,
        s_z_dict,
        s_hex_dict,
        hex_a_dict,
        xy_s_dict,
        a_dict,
        sas_dict,
        ssa_dict,
        s_nbr_dict,
    )


def get_s_nbr_dict(sas_dict):
    s_nbr_dict = {}
    for (s0, a), s1 in sas_dict.items():
        try:
            s_nbr_dict[s0] += [s1]
        except:
            s_nbr_dict[s0] = [s1]
    return s_nbr_dict


## CONVERT RAW TRAJECTORY TO TILE TRAJECTORY
def load_one_traj_from_selected_sessions(
    df_equal, df_tile, session_selected, realtime_max, d_tile
):
    """Load one trajectory from selected sessions.

    Args:
        df_equal (DataFrame): The input dataframe containing trajectory data.
        df_tile (DataFrame): The dataframe containing tile information.
        session_selected (list): List of selected session IDs.
        realtime_max (float): Maximum allowed realtime value.
        d_tile (float): Tile dimension for padding.

    Returns:
        DataFrame: Filtered dataframe containing the trajectory for the selected sessions.

    Note:
        The ill-tracked points are filtered out.
    """
    # prep
    # Lx, Ly = df_tile.x.max() - df_tile.x.min(), df_tile.y.max() - df_tile.y.min()
    xpad, ypad = d_tile * 2, d_tile * 2
    dff_equal = df_equal[
        df_equal.session.isin(session_selected)
        & (df_equal.realtime <= realtime_max)
        & (df_equal.xe >= df_tile.x.min() - xpad)
        & (df_equal.xe <= df_tile.x.max() + xpad)
        & (df_equal.ye >= df_tile.y.min() - ypad)
        & (df_equal.ye <= df_tile.y.max() + ypad)
    ]

    # get x, y trajectory
    xy_traj = dff_equal[["xe", "ye"]].values
    return xy_traj, dff_equal


def patch_skipped_transitions(
    s_traj, a_traj, df_tile_mean, xy_pixel_all, tile_id_all, ssa_dict
):
    # get skipped tile transitions
    ss1_skip = np.stack([s_traj[:-1][a_traj == -1], s_traj[1:][a_traj == -1]], axis=1)

    # get connecting tiles
    xy_connect = np.stack(
        [df_tile_mean.loc[ss1].mean(axis=0).values for ss1 in ss1_skip]
    )
    s_connect = [get_one_tile_id(xy, xy_pixel_all, tile_id_all) for xy in xy_connect]

    # insert (rather than assign) np.nan in s_traj where a_traj is -1
    s_traj_1 = np.insert(s_traj, np.where(a_traj == -1)[0] + 1, s_connect)

    # get a_traj from fixed s_traj
    a_traj_1 = np.array(
        [ssa_dict.get((s, s1), -1) for s, s1 in zip(s_traj_1[:-1], s_traj_1[1:])]
    )
    return s_traj_1, a_traj_1


def get_tile_id_traj(xy_traj, xy_pixel_all, tile_id_all):
    idx_pixel_traj = ((xy_pixel_all[None, :] - xy_traj[:, None]) ** 2).sum(2).argmin(1)
    s_traj = tile_id_all[idx_pixel_traj]
    return s_traj


def gen_s_traj_rotated(theta, xy_traj, xy_center, xy_pixel_all, tile_id_all):
    # rotate the trajectory counterclockwise
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    xy_traj_centered = xy_traj - xy_center
    xy_traj_rotated = xy_traj_centered @ rotation_matrix.T + xy_center

    # convert x,y to tile id
    s_traj_rotated = get_tile_id_traj(xy_traj_rotated, xy_pixel_all, tile_id_all)
    return s_traj_rotated, xy_traj_rotated


def get_patched_s_traj(
    s_traj_unpatched, ssa_dict, df_tile_mean, xy_pixel_all, tile_id_all
):
    # load a range of data
    # dff = df_equal[
    #     df_equal.session.isin(session_selected) & (df_equal.realtime <= realtime_max)
    # ]
    # get s_traj as self-avoiding tile walk
    # s_traj = dff_equal.tile_id.values
    s_traj_r = np.array(
        [x for x, y in zip(s_traj_unpatched[:-1], s_traj_unpatched[1:]) if x != y],
        dtype=int,
    )

    # get a_traj as tile transitions
    a_traj_r = np.array(
        [ssa_dict.get((s, s1), -1) for s, s1 in zip(s_traj_r[:-1], s_traj_r[1:])]
    )
    n_skipped_transition = (a_traj_r == -1).sum()
    print(
        f"{n_skipped_transition}/{len(a_traj_r)} skipped transitions are found before patching"
    )
    if n_skipped_transition == 0:
        return s_traj_r, a_traj_r

    # patch skipped transitions
    s_traj_1, a_traj_1 = patch_skipped_transitions(
        s_traj_r, a_traj_r, df_tile_mean, xy_pixel_all, tile_id_all, ssa_dict
    )

    n_skipped_transition = (a_traj_1 == -1).sum()
    print(
        f"{n_skipped_transition}/{len(a_traj_1)} skipped transitions are found after patching"
    )
    return s_traj_1, a_traj_1


def get_rotated_dat_traj_for_one_mouse(
    mouse_id,
    theta_list,
    session_selected,
    df_tile_mean,
    xy_pixel_all,
    tile_id_all,
    ssa_dict,
    n_tiles,
    in_dir,
    df_tile,
    d_tile,
):
    """Get the tile walk trajectory & rotated trajectories for one mouse in selected sessions."""
    # param
    tag = f"mouse_{mouse_id}"
    realtime_max = np.inf  # 3600 * 10 # full data
    center_tile_id = 77
    xy_center = df_tile_mean.loc[center_tile_id, ["x", "y"]].values

    # load
    df_equal = pickle.load(open(in_dir / tag / "df_equal", "rb"))
    xy_traj, dff_equal = load_one_traj_from_selected_sessions(
        df_equal, df_tile, session_selected, realtime_max, d_tile
    )

    # rotate trajectory (30s)
    dat_traj_list = []
    for theta in theta_list:
        s_traj_rotated, xy_traj_rotated = gen_s_traj_rotated(
            theta, xy_traj, xy_center, xy_pixel_all, tile_id_all
        )

        # patch the rotated trajectory
        s_traj_patched, a_traj_patched = get_patched_s_traj(
            s_traj_rotated, ssa_dict, df_tile_mean, xy_pixel_all, tile_id_all
        )

        # pad a_traj at t=0 so that it has same length as s_traj
        a_traj_ext = np.concatenate(([-1], a_traj_patched))

        # get occupancy map & top occupied tiles from the patched trajectory
        occ_map, s_top = get_occ_map_from_s_traj(s_traj_patched, n_tiles)

        # append
        dat_traj = s_traj_patched, a_traj_ext, occ_map, s_top, xy_traj_rotated
        dat_traj_list.append(dat_traj)
    return dat_traj_list


# def get_s_traj_from_data(dff_equal, ssa_dict, df_tile_mean, xy_pixel_all, tile_id_all):
#     # load a range of data
#     # dff = df_equal[
#     #     df_equal.session.isin(session_selected) & (df_equal.realtime <= realtime_max)
#     # ]
#     # get s_traj as self-avoiding tile walk
#     s_traj = dff_equal.tile_id.values
#     s_traj_r = np.array(
#         [x for x, y in zip(s_traj[:-1], s_traj[1:]) if x != y], dtype=int
#     )

#     # get a_traj as tile transitions
#     a_traj_r = np.array(
#         [ssa_dict.get((s, s1), -1) for s, s1 in zip(s_traj_r[:-1], s_traj_r[1:])]
#     )
#     n_skipped_transition = (a_traj_r==-1).sum()
#     print(f'{n_skipped_transition}/{len(a_traj_r)} skipped transitions are found before patching')

#     # patch skipped transitions
#     s_traj_1, a_traj_1 = patch_skipped_transitions(s_traj_r, a_traj_r, df_tile_mean, xy_pixel_all, tile_id_all, ssa_dict)

#     n_skipped_transition = (a_traj_1==-1).sum()
#     print(f'{n_skipped_transition}/{len(a_traj_1)} skipped transitions are found after patching')
#     return s_traj_1, a_traj_1

# def get_s_traj_from_markov_data(seed, dff_mc, xy_pixel_all, tile_id_all, ssa_dict):
#     """use seed = 1-100"""
#     # get s_traj
#     xy_all = dff_mc[[f"x{seed}", f"y{seed}"]].values
#     s_traj = [get_one_tile_id(xy, xy_pixel_all, tile_id_all) for xy in xy_all]

#     # get s_traj
#     s_traj_r = np.array(
#         [x for x, y in zip(s_traj[:-1], s_traj[1:]) if x != y], dtype=int
#     )
#     a_traj_r = np.array(
#         [ssa_dict.get((s, s1), -1) for s, s1 in zip(s_traj_r[:-1], s_traj_r[1:])]
#     )
#     return s_traj_r, a_traj_r


def get_s_traj_from_random_walk(s0, n_iter, sas_dict, seed=42):
    np.random.seed(seed)
    s_traj = [s0]
    a_traj = []
    for iter in range(n_iter * 2):
        s0 = s_traj[-1]
        a0 = np.random.choice(range(ringsize))
        s1 = sas_dict.get((s0, a0), -1)
        if (s1 == -1) | (s1 == s0):
            continue
        s_traj.append(s1)
        a_traj.append(a0)
    return np.array(s_traj)[:n_iter], np.array(a_traj)[: n_iter - 1]


def get_occ_map_from_s_traj(s_traj, n_tiles):
    occ_map = np.array([Counter(s_traj)[x] for x in range(n_tiles)])

    # # get top tiles that cover 75% of occupancy
    # s_sort = np.argsort(occ_map)[::-1]
    # n_75 = np.where(occ_map[s_sort].cumsum() / occ_map.sum() >= 0.75)[0][0]
    # s_75 = s_sort[:n_75]

    # get top half tiles with high occupancy
    s_sort = np.argsort(occ_map)[::-1]
    s_top = s_sort[: n_tiles // 2]
    return occ_map, s_top


## EVAL MAP
def get_e_sh_arr(n_tiles, ringsize):
    """
    e: edge_id
    s: tile_id
    h: heading
    """
    eid_shid_arr = np.array(list(product(range(n_tiles), range(ringsize))))
    # roll_id_arr = np.array(
    #     [[((y + x) % ringsize, x) for x in range(ringsize)] for y in range(ringsize)]
    # )
    return eid_shid_arr


def get_one_ring_as_maze_local_shape(s_base, s_z_dict, sas_dict, z_th=4):
    z_ = np.array(
        [
            s_z_dict.get(y, np.inf)
            for y in [sas_dict.get((s_base, x), -1) for x in range(ringsize)]
        ]
    )
    dz_ = z_ - s_z_dict[s_base]
    dz_clip_ = np.clip(dz_, -z_th, z_th)
    return dz_clip_


def get_sh_dz_arr(s_z_dict, sas_dict):
    """
    s: tile_id
    eid: edge_id
    """
    sh_dz_arr = np.array(
        [
            get_one_ring_as_maze_local_shape(x, s_z_dict, sas_dict)
            for x in list(s_z_dict)
        ]
    )
    # ring_roll_arr = np.stack(
    #     [np.roll(sh_dz_arr, x, axis=1) for x in range(ringsize)], axis=2
    # )
    return sh_dz_arr


# a_pi_dict = {
#     0: (5, 0, 1),
#     1: (0, 1, 2),
#     2: (1, 2, 3),
#     3: (2, 3, 4),
#     4: (3, 4, 5),
#     5: (4, 5, 0),
# }
# def get_sap_arr_v1(a_pi_dict):
#     '''
#     NOTE:
#     - sap stands for (s,a) --> p(s1)
#     - add one extra row of action for the null state
#     '''
#     sas_dict_for_pi = {(s,a):[sas_dict.get((s,pi), -1) for pi in a_pi_dict[a]] for s,a in list(sas_dict)}
#     sas_dict_for_pi = {(s,a):[x for x in s1_ if x != -1] for (s,a),s1_ in sas_dict_for_pi.items()}
#     #
#     sap_dict = {}
#     for (s,a),s1_ in sas_dict_for_pi.items():
#         p1_ = p_null.copy()
#         p1_[s1_] = 1.0 / len(s1_)
#         sap_dict[(s,a)] = p1_
#     sap_arr = np.stack([np.stack([sap_dict.get((s,a), p_null) for s in range(n_tiles)]) for a in range(len(a_dict) + 1)])
#     return sap_arr


def get_p_vonmises_for_enumPI(n_kappa_equal=101):
    # load
    x_ = np.arange(ringsize)

    # sweep kappa to get std of von Mises distribution
    kappa_list = np.linspace(0, 20, 2001)
    std_list = []
    for kappa in kappa_list:
        p_ = vonmises.pdf(x_ * np.pi / 3, loc=0, kappa=kappa)
        p_ = p_ / p_.sum()
        std = (np.array([0, 1, 2, 3, 2, 1]) ** 2 * p_).sum() ** 0.5
        std_list.append(std)

    #  find kappas that yield equal std increments
    std_equal = np.linspace(std_list[0], 0, n_kappa_equal)
    kappa_equal = kappa_list[
        np.abs(np.array(std_list)[:, None] - std_equal[None]).argmin(0)
    ]

    # get von Mises distributions for a=0
    p0_equal = []
    for kappa in kappa_equal:
        p_ = vonmises.pdf(x_ * np.pi / 3, loc=0, kappa=kappa)
        p_ = p_ / p_.sum()
        p0_equal.append(p_)

    # replace last p0 with onehot
    p0_onehot = np.eye(ringsize)[0]
    p0_equal[-1] = p0_onehot

    # get von Mises distributions for all actions
    pi_tensor = np.stack(
        [np.roll(p0_equal, x, axis=1) for x in range(ringsize)], axis=1
    )
    return pi_tensor


def get_sap_arr(pi, sas_dict, ringsize, n_tiles):
    sap_arr = np.zeros([ringsize, n_tiles, n_tiles])  # a,p(s),p(s1)
    for s in range(n_tiles):
        s1_ring = np.array([sas_dict.get((s, b), -1) for b in range(ringsize)])
        bool_ring = s1_ring != -1
        a_valid = np.where(bool_ring)[0]
        for a in a_valid:
            pi_ring = pi[a, bool_ring]
            pi_ring = pi_ring / pi_ring.sum()
            s1_ring_r = s1_ring[bool_ring]
            sap_arr[a, s, s1_ring_r] = pi_ring
    return sap_arr


# def eval_one_map(ring_arr_masked, ring_arr, sap_arr, s_traj, a_traj):
#     # load
#     s_p_mask_arr = get_s_p_mask_arr(ring_arr_masked, ring_arr)

#     # iter
#     init_p = np.zeros(n_tiles)
#     init_p[s_traj[0]] = 1.0
#     p_traj = [init_p]
#     for s1, a in zip(s_traj[1:], a_traj):
#         # propagate normally for a valid transition
#         if a >= 0:
#             p0 = p_traj[-1]
#             p1 = (sap_arr[a] * p0[:, None]).sum(0)
#             p2 = p1 * s_p_mask_arr[s1]
#             p_traj.append(p2 / p2.sum())

#         # for broken transition, reset p to ground truth
#         else:
#             p2 = p_null.copy()
#             p2[s1] = 1.0
#             p_traj.append(p2)
#     return np.array(p_traj)


def get_eid_unknown(map, n_edges):
    """
    map is a list of encoded edge ids
    """
    eid_unknown = np.array(sorted(set(range(n_edges)) - set(map)))
    return eid_unknown


def get_pq_mask_dict_for_ambiguous_edges(sh_dz_arr):
    # pq_mask_dict = {dz: sh_dz_arr == dz for dz in np.unique(sh_dz_arr)}
    pq_mask_dict = {
        (dz, h): np.roll(sh_dz_arr, -h, axis=1) == dz
        for dz in np.unique(sh_dz_arr)
        for h in range(ringsize)
    }
    return pq_mask_dict


def get_pq_mask_dict_for_unknown_edges(map, e_sh_arr, n_tiles, ringsize, n_edges):
    # load
    eid_unknown = get_eid_unknown(map, n_edges)

    # iter
    pq_mask_unknown = np.zeros((n_tiles, ringsize), dtype=bool)
    if len(eid_unknown) > 0:
        sh_masked = e_sh_arr[eid_unknown]
        pq_mask_unknown[*sh_masked.T] = True

    # get dict
    pq_mask_dict = {h: np.roll(pq_mask_unknown, -h, axis=1) for h in range(ringsize)}
    return pq_mask_dict


def get_pq_mask_dict(pq_mask_unknown_dict, pq_mask_amb_dict):
    pq_mask_dict = {
        (dz, h): pq_mask_amb + pq_mask_unknown_dict[h]
        for (dz, h), pq_mask_amb in pq_mask_amb_dict.items()
    }
    return pq_mask_dict


# V1
def diffuse_pq_one_step_v1(p0, q0, a, sap_arr, pi):
    # q_prop
    # q1 = q0 @ matrix_power(pi, 5)
    q1 = q0 @ pi
    # q1 = np.ones(ringsize) / ringsize  # uniform distribution

    # p_prop
    q0_roll = np.roll(q0, a)
    p1 = (sap_arr * p0[None, :, None] * q0_roll[:, None, None]).sum(0).sum(0)
    # p1 = (sap_arr[a] * p0[:, None]).sum(0)
    p1 = p1 / p1.sum()
    return p1, q1


def collapse_pq_by_one_edge_v1(pq0, s, h, pq_mask_dict, sh_dz_arr):
    # load
    dz = sh_dz_arr[s, h]
    pq_mask = pq_mask_dict[(dz, 0)]  # new version uses (dz, h)
    pq0_roll = np.roll(pq0, h, axis=1)

    # update joint distribution
    pq1_roll = pq0_roll * pq_mask
    # pq1 = pq0 * pq_mask
    pq1 = np.roll(pq1_roll, -h, axis=1)
    pq1 = pq1 / pq1.sum()
    return pq1


def collapse_pq_one_step_v1(p0, q0, s, pq_mask_dict, ringsize, sh_dz_arr):
    # build joint distribution
    pq0 = p0[:, None] * q0[None, :]

    # iter thru 6 edges for tile s
    for h in range(ringsize):
        pq1 = collapse_pq_by_one_edge_v1(pq0, s, h, pq_mask_dict, sh_dz_arr)
        pq0 = pq1.copy()

    # get marginals
    p1 = pq1.sum(1)
    # q1 = pq1.sum(0) # this will cause the performance collapse e.g. 0.67 --> 0.16
    q1 = pq1[s]
    q1 = q1 / q1.sum()
    # q1 = np.ones(ringsize) / ringsize  # uniform distribution
    return p1, q1


def pq_prop_for_n_steps_v1(
    s_traj, a_traj, sap_arr, pi, pq_mask_dict, n_tiles, ringsize, sh_dz_arr
):
    # initialize
    init_p = np.zeros(n_tiles)
    init_p[s_traj[0]] = 1.0
    init_q = np.zeros(ringsize)
    init_q[0] = 1.0
    # init_q = np.ones(ringsize) / ringsize  # uniform distribution

    # iter
    p_traj = [init_p]
    q_traj = [init_q]
    for t in range(1, len(s_traj)):
        # load
        p0, q0 = p_traj[-1], q_traj[-1]
        s, a = s_traj[t], a_traj[t]

        # diffuse, collapse
        p1, q1 = diffuse_pq_one_step_v1(p0, q0, a, sap_arr, pi)
        p2, q2 = collapse_pq_one_step_v1(p1, q1, s, pq_mask_dict, ringsize, sh_dz_arr)

        # append
        p_traj.append(p2)
        q_traj.append(q2)

    p_traj = np.array(p_traj)
    q_traj = np.array(q_traj)
    return p_traj, q_traj


# V2
def get_T_tensor(sas_dict, ringsize, n_tiles):
    """permutation matrix for (s,a) --> (s',a) given tile s and heading h"""
    T = np.zeros([n_tiles, ringsize, n_tiles, ringsize])  # p,q --> p',q'
    for s in range(n_tiles):
        for a in range(ringsize):
            s1 = sas_dict.get((s, a), -1)
            if s1 == -1:
                T[s, a, :, a] = 1 / n_tiles
                # T[s, a, s, a] = 1.0
            else:
                T[s, a, s1, a] = 1.0
    return T


def get_T_tensor_dict(T, ringsize):
    """for each heading h, get permutated T, so that h-->0"""
    T_dict = {h: np.roll(T, [-h, -h], axis=[1, 3]) for h in range(ringsize)}
    return T_dict


def diffuse_pq_one_step(pq0, a, T_dict, pi):
    # q_prop
    # pq1 = pq0 @ matrix_power(pi, 5)
    pq1 = pq0 @ pi

    # p_prop
    # pq1_roll = np.roll(pq1, a, axis=1)
    # pq2_roll = (pq1_roll[:, :, None, None] * T).sum((0, 1))
    # pq2_roll = np.einsum('ij,ijkl->kl', pq1_roll, T)
    # pq2 = np.roll(pq2_roll, -a, axis=1)

    # p_prop
    T = T_dict[a]
    pq2 = np.einsum("ij,ijkl->kl", pq1, T)
    # pq2 = (pq1[:, :, None, None] * T).sum((0, 1))
    return pq2


# def collapse_pq_by_one_edge(pq0, s, h, pq_mask_dict, sh_dz_arr):
#     # load
#     dz = sh_dz_arr[s, h]
#     pq_mask = pq_mask_dict[(dz, h)]
#     # pq0_roll = np.roll(pq0, h, axis=1)

#     # update joint distribution
#     # pq1_roll = pq0_roll * pq_mask
#     pq1 = pq0 * pq_mask
#     # pq1 = np.roll(pq1_roll, -h, axis=1)
#     pq1 = pq1 / pq1.sum()
#     return pq1

# def collapse_pq_one_step(pq, s, pq_mask_dict, ringsize, sh_dz_arr):
#     # iter thru 6 edges for tile s
#     for h in range(ringsize):
#         pq = collapse_pq_by_one_edge(pq, s, h, pq_mask_dict, sh_dz_arr)
#         # pq0 = pq1.copy()
#     return pq


def collapse_pq_one_step(pq, s, pq_mask_dict, ringsize, sh_dz_arr):
    """
    NOTE: the single normalization in this function is equivalent to the iterative normalization in the previous version of this function, because the masks are disjoint and thus the order of multiplication does not matter.
    """
    for h in range(ringsize):
        # load mask
        dz = sh_dz_arr[s, h]
        pq_mask = pq_mask_dict[(dz, h)]
        # update joint distribution w/o normalization
        pq = pq * pq_mask
    # normalize
    pq = pq / pq.sum()
    return pq


def pq_prop_for_n_steps(
    s_traj, a_traj, T_dict, pi, pq_mask_dict, n_tiles, ringsize, sh_dz_arr
):
    # initialize
    init_pq = np.zeros([n_tiles, ringsize])
    init_pq[s_traj[0], 0] = 1

    # iter
    pq_traj = [init_pq]
    for t in range(1, len(s_traj)):
        # load
        pq0 = pq_traj[-1]
        s, a = s_traj[t], a_traj[t]

        # diffuse, collapse
        pq1 = diffuse_pq_one_step(pq0, a, T_dict, pi)
        pq2 = collapse_pq_one_step(pq1, s, pq_mask_dict, ringsize, sh_dz_arr)

        # append
        pq_traj.append(pq2)

    pq_traj = np.array(pq_traj)
    return pq_traj


def get_score_map(s_traj, p_traj, s_top):
    score_mask = np.zeros_like(p_traj) + np.nan
    score_mask[range(len(s_traj)), s_traj] = 1
    score_map = np.nanmean(p_traj * score_mask, axis=0)
    # score_traj = np.nansum(p_traj * score_mask, axis=1)
    #
    score_map[np.isnan(score_map)] = 0
    # score_traj[np.isnan(score_traj)] = 0
    score = score_map[s_top].mean()
    return score_map, score


def get_job_dict_for_lossless_map_enum():
    seed_pi_list_0 = [(0, 0, pi) for pi in range(0, 100, 10)]
    seed_pi_list_1 = [(1, seed, pi) for seed in range(5) for pi in range(0, 100, 10)]
    seed_pi_list_2 = [(2, seed, pi) for seed in range(5) for pi in range(0, 100, 10)]
    type_seed_pi_list = seed_pi_list_0 + seed_pi_list_1 + seed_pi_list_2
    job_dict = {
        i + 1: (type_id, seed, pi)
        for i, (type_id, seed, pi) in enumerate(type_seed_pi_list)
    }
    return job_dict
