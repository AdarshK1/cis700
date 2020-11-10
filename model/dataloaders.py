import numpy as np
import torch
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import copy
import cv2

#TODO(akulkarni) i realized this whole setup requires that the dataset is held in RAM...that kinda sucks
# with big datasets so fix that sometime soon (should be doable just refactoring tbh)


class CIS700Dataset(Dataset):

    def __init__(self, batch=1, sub_dir="cis700_data_gt/"):

        self.data_dir = sub_dir

        #TODO(akulkarni) make it read in the config used when parsing bags into npy arrays
        self.topic_dirs = ["husky_camera_image_raw",
                           "husky_semantic_camera_image_raw",
                           "map",
                           "move_base_simple_goal",
                           "move_base_GlobalPlanner_plan",
                           "move_base_TrajectoryPlannerROS_local_plan",
                           "unity_ros_husky_TrueState_odom",
                           "ground_truth_planning_move_base_GlobalPlanner_plan"]

        self.data_holder = {}
        self.data_list_holder = {}
        self.initial_times = {}
        self.raw_end_times = {}

        for topic_dir in self.topic_dirs:
            self.data_holder[topic_dir] = {}
            print("Parsing Topic: {}".format(topic_dir))
            for idx, item in enumerate(sorted(glob.glob(self.data_dir + topic_dir + "/*"))):

                datum = np.load(item, allow_pickle=True)
                # print(topic_dir, datum.shape)
                if idx == 0:
                    self.initial_times[topic_dir] = datum[-1]

                if topic_dir == "husky_semantic_camera_image_raw" or topic_dir == "husky_camera_image_raw":
                    self.data_holder[topic_dir][datum[-1] - self.initial_times[topic_dir]] = datum[:-1][0]
                # elif topic_dir == "map":
                #     print(datum[:-1].shape)
                else:
                    self.data_holder[topic_dir][datum[-1] - self.initial_times[topic_dir]] = datum[:-1]

                self.raw_end_times[topic_dir] = datum[-1]

            self.data_list_holder[topic_dir] = list([(k, v) for k, v in self.data_holder[topic_dir].items()])

        # print(self.data_holder.keys())
        # for key in self.data_holder.keys():
        #     print(sorted(list(self.data_holder[key].keys())))
        self.end_times = [sorted(list(self.data_holder[key].keys()))[-1] for key in self.data_holder.keys()]
        # self.dataholder[topic_dir] is organized as a dictionary where the key is a normalized time
        # print(self.end_times)

        self.samples_per_second = 30

        # this is super inconvenient, but it works
        self.skip_last_n_seconds = 5
        self.skip_first_n_samples = 15

        self.num_samples = (int(min(np.array(self.end_times))) - self.skip_last_n_seconds) * self.samples_per_second

        self.batch_size = batch

        # make sure the randomness doesnt repeat data points
        self.items_left = None
        self.reset_items_left()

    def reset_items_left(self):
        self.items_left = [i for i in range(self.num_samples)]

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

    def __len__(self):
        return self.num_samples

    def __getitem__(self):

        rand_indices = np.random.choice(self.items_left, self.batch_size) + self.skip_first_n_samples
        inputs_annotated = []
        inputs_rgb = []
        inputs_semantic = []
        outputs = []

        #TODO (akulkarni) comment this bs
        for rand_idx in rand_indices:
            self.items_left = np.delete(self.items_left, np.where(self.items_left == rand_idx))
            indices = {}
            vals = {}
            for topic_dir in self.topic_dirs:
                indices[topic_dir] = (max([idx for idx, (k, v) in enumerate(self.data_holder[topic_dir].items()) if
                                           k < rand_idx // self.samples_per_second]))
                vals[topic_dir] = (self.data_list_holder[topic_dir][indices[topic_dir]][1])
            map_img, map_meta = self.process_map(vals["map"])

            # annotate the input map with pose, goal and path
            annotated = self.annotate_map(map_img,
                                          map_meta,
                                          vals["move_base_GlobalPlanner_plan"],
                                          vals["move_base_simple_goal"],
                                          vals["unity_ros_husky_TrueState_odom"])

            # annotate the the map with gt path
            annotated_gt = self.annotate_map(map_img,
                                             map_meta,
                                             vals["ground_truth_planning_move_base_GlobalPlanner_plan"],
                                             None,
                                             None)

            # pad the rgb and semantic images
            curr_rgb = vals["husky_camera_image_raw"]
            curr_semantic = vals["husky_semantic_camera_image_raw"]

            rgb_padded = np.zeros(annotated.shape)
            rgb_padded[:curr_rgb.shape[0], :curr_rgb.shape[1], :] = curr_rgb
            semantic_padded = np.zeros(annotated.shape)
            semantic_padded[:curr_semantic.shape[0], :curr_semantic.shape[1], :] = curr_semantic

            # Make em all channel first, normalize em from -1 to 1
            annotated = np.moveaxis(annotated, 2, 0).astype('float64')
            annotated -= np.min(annotated)
            annotated *= 2/(np.max(annotated))
            annotated -= 1
            # print("annotated", annotated.dtype, np.max(annotated), np.min(annotated))

            rgb_padded = np.moveaxis(rgb_padded, 2, 0).astype('float64')
            rgb_padded -= np.min(rgb_padded)
            rgb_padded *= 2/(np.max(rgb_padded))
            rgb_padded -= 1
            # print("rgb_padded", rgb_padded.dtype, np.max(rgb_padded), np.min(rgb_padded))

            semantic_padded = np.moveaxis(semantic_padded, 2, 0).astype('float64')
            semantic_padded -= np.min(semantic_padded)
            semantic_padded *= 2/(np.max(semantic_padded))
            semantic_padded -= 1
            # print("semantic_padded", semantic_padded.dtype, np.max(semantic_padded), np.min(semantic_padded))

            annotated_gt = np.moveaxis(annotated_gt, 2, 0).astype('float64')
            annotated_gt -= np.min(annotated_gt)
            annotated_gt *= 2/(np.max(annotated_gt))
            annotated_gt -= 1
            # print("annotated_gt", annotated_gt.dtype, np.max(annotated_gt), np.min(annotated_gt))

            # stick them in the batch arrays
            inputs_annotated.append(annotated)
            inputs_rgb.append(rgb_padded)
            inputs_semantic.append(semantic_padded)
            outputs.append(annotated_gt)

        # return !
        return np.array(inputs_annotated), np.array(inputs_rgb), np.array(inputs_semantic), np.array(outputs)


if __name__ == "__main__":
    dset = CIS700Dataset()
    annotated, rgb, semantic, out = dset.__getitem__(0)

    print("annotated shape:", annotated.shape)
    print("rgb shape:", rgb.shape)
    print("semantic shape:", semantic.shape)
    print("out shape:", out.shape)
