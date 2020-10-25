import numpy as np
import torch
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class CIS700Dataset(Dataset):

    def __init__(self, batch=1):

        self.data_dir = "/home/adarsh/ros-workspaces/cis700_workspace/src/learned_planning_pipeline/bag_harvester/cis700_data"

        self.topic_dirs = ["husky_camera_image_raw",
                           "husky_semantic_camera_image_raw",
                           "map",
                           "move_base_simple_goal",
                           "move_base_TrajectoryPlannerROS_global_plan",
                           "move_base_TrajectoryPlannerROS_local_plan",
                           "unity_ros_husky_TrueState_odom"]

        self.data_holder = {}
        self.data_list_holder = {}
        self.initial_times = {}
        self.raw_end_times = {}

        for topic_dir in self.topic_dirs:
            self.data_holder[topic_dir] = {}
            for idx, item in enumerate(sorted(glob.glob(self.data_dir + topic_dir + "/*")))[:2]:
                datum = np.load(item, allow_pickle=True)
                print(topic_dir, datum.shape)
                if idx == 0:
                    self.initial_times[topic_dir] = datum[-1]

                if topic_dir == "husky_semantic_camera_image_raw" or topic_dir == "husky_camera_image_raw":
                    self.data_holder[topic_dir][datum[-1] - self.initial_times[topic_dir]] = datum[:-1][0]
                else:
                    self.data_holder[topic_dir][datum[-1] - self.initial_times[topic_dir]] = datum[:-1]

                self.raw_end_times[topic_dir] = datum[-1]

            self.data_list_holder[topic_dir] = list([(k, v) for k, v in self.data_holder[topic_dir].items()])

        self.end_times = [sorted(list(self.data_holder[key].keys()))[-1] for key in self.data_holder.keys()]
        # self.dataholder[topic_dir] is organized as a dictionary where the key is a normalized time

        self.samples_per_second = 30

        # this is super inconvenient, but it works
        self.skip_last_n_seconds = 5
        self.skip_first_n_samples = 30

        self.num_samples = (int(min(np.array(self.end_times))) - self.skip_last_n_seconds) * self.samples_per_second

        self.batch_size = batch

        # make sure the randomness doesnt repeat data points
        self.items_left = None
        self.reset_items_left()

    def reset_items_left(self):
        self.items_left = [i for i in range(self.num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        rand_indices = np.random.choice(self.items_left, self.batch_size) + self.skip_first_n_samples
        inputs = []
        outputs = []
        hmaps = []

        for rand_idx in rand_indices:
            new_arr = np.delete(self.items_left, np.where(self.items_left == rand_idx))
            indices = []
            vals = []
            for topic_dir in self.topic_dirs:
                indices.append(max([idx for idx, (k, v) in enumerate(self.data_holder[topic_dir].items()) if k < rand_idx // self.samples_per_second]))
                vals.append(self.data_list_holder[topic_dir][indices[-1]][1])


        #     hmap_val = np.clip(np.nan_to_num(np.moveaxis(hmap_val, 2, 0)), -50, 50)
        #
        #     input = np.clip(np.nan_to_num(np.concatenate([twist_val, odom_val, y_vec_val])), -50, 50)
        #     output = np.clip(np.nan_to_num(np.concatenate([y_vec_val_output, ctrl_vec_val])), -50, 50)
        #
        #     inputs.append(input)
        #     outputs.append(output)
        #     hmaps.append(hmap_val)
        #     # print("Got item", input.shape, output.shape, hmap_val.shape)
        #
        # return np.array(inputs), np.array(hmaps), np.array(outputs)

