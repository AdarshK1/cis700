import numpy as np
import torch
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from scipy.spatial.transform import Rotation
import copy
import cv2
import sys
import pickle

sys.path.append("/home/ian/catkin/cis700_ws/src/rosbag-dl-utils")
sys.path.append("/home/adarsh/ros-workspaces/cis700_workspace/src/rosbag-dl-utils")

from base_data_loader import BaseDataset


class CIS700Dataset(BaseDataset):

    def __init__(self, config_file, sub_dir, map_size=30, samples_per_second=120, skip_last_n_seconds=0,
                 skip_first_n_seconds=0):

        super().__init__(config_file, sub_dir, samples_per_second=samples_per_second,
                         skip_last_n_seconds=skip_last_n_seconds, skip_first_n_seconds=skip_first_n_seconds)
        self.map_size = map_size

    def process_map(self, map_arr, verbose=False):
        map_meta = {"origin_position_x": map_arr[-10],
                    "origin_position_y": map_arr[-9],
                    "origin_position_z": map_arr[-8],
                    "origin_orientation_x": map_arr[-7],
                    "origin_orientation_y": map_arr[-6],
                    "origin_orientation_z": map_arr[-5],
                    "origin_orientation_w": map_arr[-4],
                    "height": int(map_arr[-3]),
                    "resolution": map_arr[-2],
                    "width": int(map_arr[-1])}

        if verbose:
            print("Got a map at {} res of hw ({},{}) located @ X:{}, Y:{}, Z:{}".format(map_meta["resolution"],
                                                                                        map_meta["height"],
                                                                                        map_meta["width"],
                                                                                        map_meta["origin_position_x"],
                                                                                        map_meta["origin_position_y"],
                                                                                        map_meta["origin_position_z"]))
        map_part = map_arr[:len(map_arr) - 10]
        map_part = np.reshape(map_part, (map_meta["height"], map_meta["width"]))

        return map_part, map_meta

    def annotate_map(self, map_arr, meta, path, goal, pose, verbose=False):
        map_arr_copy = copy.deepcopy(map_arr)
        map_arr_copy -= np.min(map_arr_copy)
        map_arr_copy *= (255.0 / np.max(map_arr_copy))

        annotation_channel = np.full((map_arr_copy.shape[0], map_arr_copy.shape[1], 3), 100, np.uint8)

        # mark pose
        if pose is not None:
            pose_map_coords_x = int((pose[0] - meta["origin_position_x"]) / meta["resolution"])
            pose_map_coords_y = int((pose[1] - meta["origin_position_y"]) / meta["resolution"])

            if verbose:
                print("robot pose", pose[0], pose[1])
                print("map origin", meta["origin_position_x"], meta["origin_position_y"])
                print("converted", pose_map_coords_x, pose_map_coords_y)

            annotation_channel = cv2.circle(annotation_channel, (pose_map_coords_x, pose_map_coords_y), 5, (255, 0, 0),
                                            3)
            # print(annotation_channel)
            # cv2.imshow("test", annotation_channel)
            # cv2.waitKey(10000)

        # mark goal
        if goal is not None:
            goal_map_coords_x = int((goal[0] - meta["origin_position_x"]) / meta["resolution"])
            goal_map_coords_y = int((goal[1] - meta["origin_position_y"]) / meta["resolution"])

            if verbose:
                print("robot goal", goal[0], goal[1])
                print("goal_converted", goal_map_coords_x, goal_map_coords_y)

            annotation_channel = cv2.circle(annotation_channel, (goal_map_coords_x, goal_map_coords_y), 5, (0, 0, 255),
                                            3)

        # mark path
        if path is not None:
            for pose in path:
                path_map_coords_x = int((pose[0] - meta["origin_position_x"]) / meta["resolution"])
                path_map_coords_y = int((pose[1] - meta["origin_position_y"]) / meta["resolution"])

                annotation_channel = cv2.circle(annotation_channel, (path_map_coords_x, path_map_coords_y), 1,
                                                (255, 0, 255), 1)

        annotation_channel[:, :, 1] = map_arr_copy

        return annotation_channel

    def annotate_map_centered(self, map_arr, meta, path, goal, pose, verbose=False):
        annotation_channel = np.full((int(self.map_size / meta["resolution"]),
                                      int(self.map_size / meta["resolution"]), 3), 0, np.uint8)

        # mark goal
        if goal is not None:
            goal_map_coords_x = np.clip(int((goal[0] - pose[0] + self.map_size / 2) / meta["resolution"]),
                                        0,
                                        annotation_channel.shape[0] - 1)

            goal_map_coords_y = np.clip(int((goal[1] - pose[1] + self.map_size / 2) / meta["resolution"]),
                                        0,
                                        annotation_channel.shape[0] - 1)

            if verbose:
                print("robot goal", goal[0], goal[1])
                print("goal_converted", goal_map_coords_x, goal_map_coords_y)

            annotation_channel = cv2.circle(annotation_channel,
                                            (goal_map_coords_x, goal_map_coords_y),
                                            5,
                                            (0, 125, 0),
                                            -1)

        r = Rotation.from_quat(pose[3:7])
        theta = r.as_euler('zyx')[0]
        t_cos = np.cos(theta)
        t_sin = np.sin(theta)
        for i in range(annotation_channel.shape[0]):
            for j in range(annotation_channel.shape[1]):
                world_x = pose[0] + t_cos*(i * meta["resolution"] - self.map_size / 2) + \
                                    t_sin*(j * meta["resolution"] - self.map_size / 2)
                world_y = pose[1] + t_sin*(i * meta["resolution"] - self.map_size / 2) - \
                                    t_cos*(j * meta["resolution"] - self.map_size / 2)

                world_map_coords_x = int((world_x - meta["origin_position_x"]) / meta["resolution"])
                world_map_coords_y = int((world_y - meta["origin_position_y"]) / meta["resolution"])

                if map_arr.shape[0] > world_map_coords_y > 0 and 0 < world_map_coords_x < map_arr.shape[1]:
                    annotation_channel[i, j, 0] = map_arr[world_map_coords_y, world_map_coords_x]

                # mark path
        if path is not None:
            for path_pose in path:
                path_map_coords_y = np.clip(int((t_cos*(path_pose[0] - pose[0]) + \
                                                 t_sin*(path_pose[1] - pose[1]) + self.map_size / 2) / meta["resolution"]),
                                            0,
                                            annotation_channel.shape[0] - 1)

                path_map_coords_x = np.clip(int((t_sin*(path_pose[0] - pose[0]) - \
                                                 t_cos*(path_pose[1] - pose[1]) + self.map_size / 2) / meta["resolution"]),
                                            0,
                                            annotation_channel.shape[0] - 1)

                if verbose:
                    print("Path Point", pose[0], pose[1])
                    print("Path cvtd", path_map_coords_x, path_map_coords_y)

                if 0 < path_map_coords_y < annotation_channel.shape[0] and \
                        0 < path_map_coords_x < annotation_channel.shape[0]: 
                    annotation_channel = cv2.circle(annotation_channel, (path_map_coords_x, path_map_coords_y), 2,
                                                    (0, 255, 0), -1)

        if verbose:
            cv2.imshow("test", annotation_channel)
            cv2.waitKey(1000)

        return annotation_channel[:, :, :2]

    def __len__(self):
        return self.num_samples

    def norm_stuff(self, arr, zero_to_one=False):
        # print(np.min(arr), np.max(arr))
        arr = np.moveaxis(arr, 2, 0).astype('float64')
        arr -= np.min(arr)

        if not zero_to_one:
            if np.max(arr) > 0:
                arr *= 2.0 / (np.max(arr))
            else:
                arr *= 2.0
            arr -= 1.0
        else:
            if np.max(arr) > 0:
                arr *= 1.0 / (np.max(arr))
        return arr

    def __getitem__(self, idx):
        vals = super().__getitem__(idx)

        # print("keys", vals.keys())

        map_img, map_meta = self.process_map(vals["map"])
        gt_map_img, gt_map_meta = self.process_map(vals["ground_truth_planning_map"])
        # print("gt_map_meta", gt_map_meta)
        # print("map_meta", map_meta)

        # grab and pad indices
        mps = np.array(vals["ground_truth_planning_mp_indices_list"])
        if len(mps) > 5:
            mps = mps[:5]
        elif len(mps) < 5:
            mps = np.pad(mps, (0, 5-mps.shape[0], ))

        # print(mps)


        # this one is centered on the robot and of fixed size
        annotated = self.annotate_map_centered(map_img,
                                               map_meta,
                                               vals["move_base_GlobalPlanner_plan"],
                                               vals["move_base_current_goal"],
                                               # TODO (akulkarni) we didn't bag the goal so need to grab it from gt path
                                               vals["unity_ros_husky_TrueState_odom"], verbose=False)

        # resize ground truth map
        gt_map_img = cv2.resize(gt_map_img, map_img.shape, interpolation=cv2.INTER_AREA)
        gt_map_meta["resolution"] = map_meta["resolution"]

        # annotate the the map with gt path
        annotated_gt = self.annotate_map_centered(gt_map_img,
                                                  gt_map_meta,
                                                  vals["ground_truth_planning_move_base_GlobalPlanner_plan"],
                                                  None,
                                                  vals["unity_ros_husky_TrueState_odom"], verbose=False)

        # pad the rgb and semantic images
        curr_rgb = vals["husky_camera_image_raw"][0]
        curr_semantic = curr_rgb
        #curr_semantic = vals["husky_semantic_camera_image_raw"][0]

        #rgb_padded = np.zeros((annotated.shape[0], annotated.shape[1], 3))
        #rgb_padded[:curr_rgb.shape[0], :curr_rgb.shape[1], :] = curr_rgb
        #semantic_padded = np.zeros((annotated.shape[0], annotated.shape[1], 3))
        #semantic_padded[:curr_semantic.shape[0], :curr_semantic.shape[1], :] = curr_semantic
        rgb_scaled = cv2.resize(curr_rgb, (annotated.shape[0], annotated.shape[1]))
        semantic_scaled = cv2.resize(curr_semantic, (annotated.shape[0], annotated.shape[1]))

        annotated = self.norm_stuff(annotated)
        annotated_gt = self.norm_stuff(annotated_gt, zero_to_one=True)
        rgb_scaled = self.norm_stuff(rgb_scaled)
        semantic_scaled = self.norm_stuff(semantic_scaled)

        #something weird going on, zero input
        valid = False
        if len(vals["ground_truth_planning_move_base_GlobalPlanner_plan"]) > 0 and \
           len(vals["move_base_GlobalPlanner_plan"]) > 0:
            if np.linalg.norm(np.array(vals["unity_ros_husky_TrueState_odom"][0:2]) - 
                    np.array(vals["ground_truth_planning_move_base_GlobalPlanner_plan"][0][0:2])) < 2 and \
               np.linalg.norm(np.array(vals["move_base_GlobalPlanner_plan"][-1][0:2]) - 
                    np.array(vals["ground_truth_planning_move_base_GlobalPlanner_plan"][-1][0:2])) < 2:
                valid = True 

        # return !
        return np.array(annotated), np.array(rgb_scaled), np.array(semantic_scaled), np.array(annotated_gt), mps, valid


class CIS700Pickled(Dataset):

    def __init__(self, pickle_dir="/home/adarsh/HDD1/cis700_final/pickled/20201124-034755/"):
        self.dir = pickle_dir

        self.filenames = sorted(glob.glob(self.dir + "*"))
        print(len(self.filenames))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        sample = pickle.load(open(self.filenames[idx], 'rb'))
        return sample


if __name__ == "__main__":
    N = 100
    # sample_fname = "/media/ian/SSD1/tmp_datasets/UnityLearnedPlanning/20201205-130953/"
    # sample_fname = "/home/adarsh/HDD1/cis700_final/processed/20201204-015130/"
    # sample_fname = "/home/adarsh/HDD1/cis700_final/processed/20201204-015728/"
    sample_fname = "/home/adarsh/HDD1/cis700_final/processed/20201204-015809/"
    # sample_fname = "/home/adarsh/HDD1/cis700_final/processed/20201204-015930/"
    # sample_fname = "/home/adarsh/HDD1/cis700_final/processed/20201204-020050/"
    # sample_fname = "/home/adarsh/HDD1/cis700_final/processed/20201204-020212/"
    # sample_fname = "/home/adarsh/HDD1/cis700_final/processed/20201204-020334/"
    # sample_fname = "/home/adarsh/HDD1/cis700_final/processed/20201204-020456/"
    # sample_fname = "/home/adarsh/HDD1/cis700_final/processed/20201204-020535/"
    # sample_fname = "/home/adarsh/HDD1/cis700_final/processed/20201204-020635/"
    # config_file = "/home/ian/catkin/cis700_ws/src/rosbag-dl-utils/harvester_configs/cis700_prim.yaml"
    config_file = "/home/adarsh/ros-workspaces/cis700_workspace/src/rosbag-dl-utils/harvester_configs/cis700.yaml"
    dset = CIS700Dataset(config_file, sample_fname, samples_per_second=1, map_size=20)

    def torch_to_cv2(out):
        # print(np.min(out), np.max(out))
        # out = out[:, :, :]
        out -= np.min(out)
        out *= 255.0 / (np.max(out))
        out = np.moveaxis(out, 0, 2)
        out = np.append(out, np.zeros((out.shape[0], out.shape[1], 1)), axis=2)
        return out


    for i in range(N):
        annotated, rgb, semantic, out, valid = dset.__getitem__(i)

        #print("annotated shape:", annotated.shape)
        #print("rgb shape:", rgb.shape)
        #print("semantic shape:", semantic.shape)
        #print("out shape:", out.shape)

        cv2.imshow("annotated", torch_to_cv2(annotated))
        cv2.imshow("rgb", torch_to_cv2(rgb)/255)
        cv2.imshow("out", torch_to_cv2(out))
        cv2.waitKey(100)
