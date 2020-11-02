import numpy as np
import torch
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import copy
import cv2


class CIS700Dataset(Dataset):

    def __init__(self, batch=1):

        self.data_dir = "/home/adarsh/ros-workspaces/cis700_workspace/src/learned_planning_pipeline/bag_harvester/cis700_data_2/"

        self.topic_dirs = ["husky_camera_image_raw",
                           "husky_semantic_camera_image_raw",
                           "map",
                           "move_base_simple_goal",
                           "move_base_GlobalPlanner_plan",
                           "move_base_TrajectoryPlannerROS_local_plan",
                           "unity_ros_husky_TrueState_odom"]

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

    def process_map(self, map_arr):
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

        # print(map_meta["height"], map_meta["width"], map_meta["resolution"])
        print("Got a map at {} res of hw ({},{}) located @ X:{}, Y:{}, Z:{}".format(map_meta["resolution"],
                                                                                    map_meta["height"],
                                                                                    map_meta["width"],
                                                                                    map_meta["origin_position_x"],
                                                                                    map_meta["origin_position_y"],
                                                                                    map_meta["origin_position_z"]))
        map_part = map_arr[:len(map_arr) - 10]
        map_part = np.reshape(map_part, (map_meta["height"], map_meta["width"]))

        return map_part, map_meta

    def annotate_map(self, map_arr, meta, path, goal, pose):
        map_arr_copy = copy.deepcopy(map_arr)
        map_arr_copy -= np.min(map_arr_copy) 
        map_arr_copy *= (255.0 / np.max(map_arr_copy))

        annotation_channel = np.zeros(map_arr_copy.shape)

        # mark pose
        pose_map_coords_x = int((pose[0] - meta["origin_position_x"]) / meta["resolution"])
        pose_map_coords_y = int((pose[1] - meta["origin_position_y"]) / meta["resolution"])

        annotation_channel[pose_map_coords_y, pose_map_coords_x] = 100
        annotation_channel = cv2.circle(annotation_channel, (pose_map_coords_y, pose_map_coords_x), 5, (0, 255, 0))

        # mark goal
        goal_map_coords_x = int((goal[0] - meta["origin_position_x"]) / meta["resolution"])
        goal_map_coords_y = int((goal[1] - meta["origin_position_y"]) / meta["resolution"])

        annotation_channel[goal_map_coords_y, goal_map_coords_x] = 150
        annotation_channel = cv2.circle(annotation_channel, (goal_map_coords_y, goal_map_coords_x), 5, (0, 255, 0))

        # mark path
        # print(path)
        for pose in path:
            path_map_coords_x = int((pose[0] - meta["origin_position_x"]) / meta["resolution"])
            path_map_coords_y = int((pose[1] - meta["origin_position_y"]) / meta["resolution"])

            annotation_channel[path_map_coords_y, path_map_coords_x] = 200
            annotation_channel = cv2.circle(annotation_channel, (path_map_coords_y, path_map_coords_x), 5, (0, 255, 0))

        concated = np.dstack([map_arr_copy, annotation_channel, annotation_channel])
        # print("concated shape", concated.shape)
        cv2.imshow("test", concated)
        cv2.waitKey(10000)
        # img = Image.fromarray(map_arr_copy, "I")
        # img = Image.fromarray(concated, "RGB")
        # img.show()


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        rand_indices = np.random.choice(self.items_left, self.batch_size) + self.skip_first_n_samples
        inputs = []
        outputs = []

        for rand_idx in rand_indices:
            self.items_left = np.delete(self.items_left, np.where(self.items_left == rand_idx))
            indices = {}
            vals = {}
            for topic_dir in self.topic_dirs:
                indices[topic_dir] = (max([idx for idx, (k, v) in enumerate(self.data_holder[topic_dir].items()) if k < rand_idx // self.samples_per_second]))
                vals[topic_dir] = (self.data_list_holder[topic_dir][indices[topic_dir]][1])
            # print(rand_idx, indices)
            # print(vals)
            # print(vals["map"].shape)
            map_img, map_meta = self.process_map(vals["map"])

            self.annotate_map(map_img,
                              map_meta,
                              vals["move_base_GlobalPlanner_plan"],
                              vals["move_base_simple_goal"],
                              vals["unity_ros_husky_TrueState_odom"])


if __name__ == "__main__":
    dset = CIS700Dataset()
    print(dset.__getitem__(0))
