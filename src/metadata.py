from pathlib import Path
import numpy as np


## PYTHON EVIRONMENT
"""
python 3.11.13
ipython
opencv 4.10.0
matplotlib
pandas
scipy 
scikit-image
scikit-learn
multiprocess
h5py
sparse
nbformat
"""

## SET DIRECTORIES
# bigdata_dir = Path("/media/mat/F8943ADD943A9DD6/")
# bigdata_dir = Path("/mnt/HDD/")
project_dir = Path(__file__).parent.parent

## BASH PARAMETERS
mouse_ids = [3, 4]

## GLOBAL PARAMETERS
# UTILS
q1, q2, q3 = 25, 50, 75  # quartiles

# MINIMAP
maze = np.array(
    [
        [-1] * 17,
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 6, -1, -1],
        [-1, -1, -1, -1, -1, -1, 8, 1, 1, 3, 1, 1, 1, 1, 3, 8, -1],
        [-1, -1, -1, -1, -1, 8, 1, 1, 1, 8, 9, 1, 1, 1, 8, 8, -1],
        [-1, -1, -1, -1, 8, 5, 1, 1, 2, 1, 1, 9, 2, 12, 6, 8, -1],
        [-1, -1, -1, 7, 5, 3, 6, 10, 1, 2, 2, 3, 1, 5, 5, 8, -1],
        [-1, -1, -1, 8, 5, 2, 1, 1, 1, 5, 8, 1, 10, 6, 7, -1, -1],
        [-1, -1, 8, 4, 8, 3, 3, 2, 8, 3, 3, 5, 4, 7, 8, -1, -1],
        [-1, -1, 5, 3, 2, 1, 1, 1, 1, 2, 3, 2, 6, 6, -1, -1, -1],
        [-1, 8, 3, 3, 3, 1, 2, 9, 1, 2, 2, 9, 6, 8, -1, -1, -1],
        [-1, 5, 8, 3, 3, 2, 1, 1, 2, 9, 4, 8, 8, -1, -1, -1, -1],
        [-1, 6, 5, 2, 8, 8, 2, 1, 3, 3, 4, 8, -1, -1, -1, -1, -1],
        [-1, 8, 3, 1, 1, 1, 1, 2, 3, 2, 8, -1, -1, -1, -1, -1, -1],
        [-1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1] * 17,
    ]
)[::-1, :]
n_tiles = (maze != -1).sum()
ringsize = 6
n_edges = n_tiles * ringsize

# # EQUAL
d_tile = 170  # width of center tile of mouse 10 session 6
# # 3 for 25Hz, 5for 40Hz
# sigma_smooth_dict = {
#     3: 3,
#     4: 3,
#     5: 3,
#     6: 3,
#     7: 3,
#     8: 3,
#     9: 5,
#     10: 5,
# }
# fs_equal = 4  # sample rate (points per tile) of equalized trajectory xe, ye
# fs_pair_th = 2  # criterion of a pair of matching paths: d_p2p < 1/2*d_tile
# # dv_pair_th = np.pi / 2  # angle threshold (consider matching if better than orthogonal)
# # v_tern_bins = [0.5, 0.8] # ??
# # q_v_low, q_v_high = q2, q3  # for ternarize velocity
# # l_pair_th = 6  # toklen threshold (minimum is a one tile jump ~1.5 tiles)

# # BUNDLE
# l_bundle_max = 200
# l_bundle_min = 4


# # # TREESEGM
# # soft_exclusion = q1
# # soft_inclusion = q3


# ## LOAD: align
# align_data_dir = project_dir / "data" / "anchors_all_mice"


# ## LOAD: tile
# tile_data_dir = project_dir / "data" / "mouse10_sess6_annotated_tiles"


# ## LOAD: day
# csv_timestamp_dict = {
#     3: [
#         "jakob_data_3dmaze/centroid_timestamps_2023-08-01T13_50_25.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2023-08-02T10_10_43.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2023-08-03T10_22_11.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2023-08-04T12_13_05.csv",
#     ],
#     4: [
#         "jakob_data_3dmaze/centroid_timestamps_2023-08-23T10_00_30.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2023-08-24T09_07_52.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2023-08-25T10_02_17.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2023-08-26T10_39_02.csv",
#     ],
#     5: [
#         "jakob_data_3dmaze/centroid_timestamps_2024-05-13T10_19_20.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2024-05-14T10_42_04.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2024-05-15T10_18_17.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2024-05-16T09_22_20.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2024-05-17T09_16_43.csv",
#     ],
#     6: [
#         "jakob_data_3dmaze/centroid_timestamps_2024-05-28T09_50_08.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2024-05-29T09_16_11.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2024-05-30T11_18_21.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2024-05-31T09_20_53.csv",
#     ],
#     7: [
#         "jakob_data_3dmaze/centroid_timestamps_2024-11-18T12_53_20.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2024-11-19T08_32_30.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2024-11-20T08_56_15.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2024-11-21T09_12_12.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2024-11-22T08_53_41.csv",
#     ],
#     8: [
#         "jakob_data_3dmaze/centroid_timestamps_2025-04-28T10_46_02.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2025-04-29T09_08_47.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2025-04-30T08_50_29.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2025-05-01T10_54_43.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2025-05-02T09_32_03.csv",
#     ],
#     9: [],
#     10: [
#         "jakob_data_3dmaze/centroid_timestamps_2025-09-09T09_22_51.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2025-09-10T09_00_35.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2025-09-11T09_43_51.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2025-09-12T10_25_20.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2025-09-15T11_16_02.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2025-09-16T12_14_44.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2025-09-17T09_57_09.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2025-09-18T10_52_07.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2025-09-19T09_33_31.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2025-09-23T11_40_58.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2025-09-24T11_22_10.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2025-09-25T09_45_20.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2025-09-26T11_26_33.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2025-10-01T10_18_06.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2025-10-02T09_23_21.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2025-10-07T13_51_58.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2025-10-08T11_09_52.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2025-10-09T12_01_13.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2025-10-10T09_40_38.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2025-10-15T09_43_49.csv",
#         "jakob_data_3dmaze/centroid_timestamps_2025-10-16T09_51_54.csv",
#     ],
# }

# h5_dict = {
#     3: [
#         "jakob_data_3dmaze/top_cam_video_coarse2023-08-01T13_50_25.avi.000_top_cam_video_coarse2023-08-01T13_50_25.analysis.h5",
#         "jakob_data_3dmaze/top_cam_video_coarse2023-08-02T10_10_44.avi.000_top_cam_video_coarse2023-08-02T10_10_44.analysis.h5",
#         "jakob_data_3dmaze/top_cam_video_coarse2023-08-03T10_22_12.avi.000_top_cam_video_coarse2023-08-03T10_22_12.analysis.h5",
#         "jakob_data_3dmaze/top_cam_video_coarse2023-08-04T12_13_06.avi.000_top_cam_video_coarse2023-08-04T12_13_06.analysis.h5",
#     ],
#     4: [
#         "jakob_data_3dmaze/top_cam_video_coarse2023-08-23T10_00_31.avi.000_top_cam_video_coarse2023-08-23T10_00_31.analysis.h5",
#         "jakob_data_3dmaze/top_cam_video_coarse2023-08-24T09_07_53.avi.000_top_cam_video_coarse2023-08-24T09_07_53.analysis.h5",
#         "jakob_data_3dmaze/top_cam_video_coarse2023-08-25T10_02_18.avi.000_top_cam_video_coarse2023-08-25T10_02_18.analysis.h5",
#         "jakob_data_3dmaze/top_cam_video_coarse2023-08-26T10_39_03.avi.000_top_cam_video_coarse2023-08-26T10_39_03.analysis.h5",
#     ],
#     5: [
#         "jakob_data_3dmaze/top_cam_video_coarse2024-05-13T10_19_21.avi.000_top_cam_video_coarse2024-05-13T10_19_21.analysis.h5",
#         "jakob_data_3dmaze/top_cam_video_coarse2024-05-14T10_42_05.avi.000_top_cam_video_coarse2024-05-14T10_42_05.analysis.h5",
#         "jakob_data_3dmaze/top_cam_video_coarse2024-05-15T10_18_17.avi.000_top_cam_video_coarse2024-05-15T10_18_17.analysis.h5",
#         "jakob_data_3dmaze/top_cam_video_coarse2024-05-16T09_22_21.avi.000_top_cam_video_coarse2024-05-16T09_22_21.analysis.h5",
#         "jakob_data_3dmaze/top_cam_video_coarse2024-05-17T09_16_44.avi.000_top_cam_video_coarse2024-05-17T09_16_44.analysis.h5",
#     ],
#     6: [
#         "jakob_data_3dmaze/top_cam_video_coarse2024-05-28T09_50_09.avi.000_top_cam_video_coarse2024-05-28T09_50_09.analysis.h5",
#         "jakob_data_3dmaze/top_cam_video_coarse2024-05-29T09_16_11.avi.000_top_cam_video_coarse2024-05-29T09_16_11.analysis.h5",
#         "jakob_data_3dmaze/top_cam_video_coarse2024-05-30T11_18_22.avi.000_top_cam_video_coarse2024-05-30T11_18_22.analysis.h5",
#         "jakob_data_3dmaze/top_cam_video_coarse2024-05-31T09_20_54.avi.000_top_cam_video_coarse2024-05-31T09_20_54.analysis.h5",
#     ],
#     7: [
#         "jakob_data_3dmaze/top_cam_video_coarse2024-11-18T12_53_20.avi.000_top_cam_video_coarse2024-11-18T12_53_20.analysis.h5",
#         "jakob_data_3dmaze/top_cam_video_coarse2024-11-19T08_32_30.avi.000_top_cam_video_coarse2024-11-19T08_32_30.analysis.h5",
#         "jakob_data_3dmaze/top_cam_video_coarse2024-11-20T08_56_15.avi.000_top_cam_video_coarse2024-11-20T08_56_15.analysis.h5",
#         "jakob_data_3dmaze/top_cam_video_coarse2024-11-21T09_12_12.avi.000_top_cam_video_coarse2024-11-21T09_12_12.analysis.h5",
#         "jakob_data_3dmaze/top_cam_video_coarse2024-11-22T08_53_41.avi.000_top_cam_video_coarse2024-11-22T08_53_41.analysis.h5",
#     ],
#     8: [
#         "jakob_data_3dmaze/top_cam_video_coarse2025-04-28T10_46_03.avi.000_top_cam_video_coarse2025-04-28T10_46_03.analysis.h5",
#         "jakob_data_3dmaze/top_cam_video_coarse2025-04-29T09_08_48.avi.000_top_cam_video_coarse2025-04-29T09_08_48.analysis.h5",
#         "jakob_data_3dmaze/top_cam_video_coarse2025-04-30T08_50_30.avi.000_top_cam_video_coarse2025-04-30T08_50_30.analysis.h5",
#         "jakob_data_3dmaze/top_cam_video_coarse2025-05-01T10_54_44.avi.000_top_cam_video_coarse2025-05-01T10_54_44.analysis.h5",
#         "jakob_data_3dmaze/top_cam_video_coarse2025-05-02T09_32_04.avi.000_top_cam_video_coarse2025-05-02T09_32_04.analysis.h5",
#     ],
#     9: [],
#     10: [
#         "jakob_data_3dmaze/v2_compressed_video_2025-09-09T09_22_52.analysis.h5",
#         "jakob_data_3dmaze/v2_compressed_video_2025-09-10T09_00_36.analysis.h5",
#         "jakob_data_3dmaze/v2_compressed_video_2025-09-11T09_43_52.analysis.h5",
#         "jakob_data_3dmaze/v2_compressed_video_2025-09-12T10_25_21.analysis.h5",
#         "jakob_data_3dmaze/v2_compressed_video_2025-09-15T11_16_03.analysis.h5",
#         "jakob_data_3dmaze/v2_compressed_video_2025-09-16T12_14_46.analysis.h5",
#         "jakob_data_3dmaze/v2_compressed_video_2025-09-17T09_57_10.analysis.h5",
#         "jakob_data_3dmaze/v2_compressed_video_2025-09-18T10_52_08.analysis.h5",
#         "jakob_data_3dmaze/v2_compressed_video_2025-09-19T09_33_32.analysis.h5",
#         "jakob_data_3dmaze/v2_compressed_video_2025-09-23T11_40_59.analysis.h5",
#         "jakob_data_3dmaze/v2_compressed_video_2025-09-24T11_22_11.analysis.h5",
#         "jakob_data_3dmaze/compressed_video_2025-09-25T09_45_21.analysis.h5",
#         "jakob_data_3dmaze/v2_compressed_video_2025-09-26T11_26_34.analysis.h5",
#         "jakob_data_3dmaze/v2_compressed_video_2025-10-01T10_18_07.analysis.h5",
#         "jakob_data_3dmaze/v2_compressed_video_2025-10-02T09_23_23.analysis.h5",
#         "jakob_data_3dmaze/v2_compressed_video_2025-10-07T13_51_59.analysis.h5",
#         "jakob_data_3dmaze/v2_compressed_video_2025-10-08T11_09_53.analysis.h5",
#         "jakob_data_3dmaze/v2_compressed_video_2025-10-09T12_01_15.analysis.h5",
#         "jakob_data_3dmaze/v2_compressed_video_2025-10-10T09_40_39.analysis.h5",
#         "jakob_data_3dmaze/v2_compressed_video_2025-10-15T09_43_50.analysis.h5",
#         "jakob_data_3dmaze/v2_compressed_video_2025-10-16T09_51_56.analysis.h5",
#     ],
# }

# n_sessions_dict = {x: len(y) for x, y in h5_dict.items()}
