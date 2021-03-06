#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, OccupancyGrid
from std_msgs.msg import Int16MultiArray
import dubins
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop, heapify
from copy import deepcopy
import itertools
from tf.transformations import euler_from_quaternion, quaternion_from_euler
# import fileinput
# import cProfile


class Node:
    """
    Container for node data. Nodes are sortable by the value of (f, -g).
    """

    def __init__(self, g, h, state, parent, mp, index=None, parent_index=None, graph_depth=0):
        self.f = g + h          # total-cost
        self.g = g              # cost-to-come
        self.h = h              # heuristic
        self.state = state
        self.parent = parent
        self.mp = mp
        self.index = index
        self.parent_index = parent_index
        self.graph_depth = graph_depth
        self.is_closed = False  # True if node has been closed.

    def __lt__(self, other):
        return (self.f, -self.g) < (other.f, -other.g)

    # def __repr__(self):
    #     return f"Node g={self.g}, h={self.h}, state={self.state}, parent={self.parent}, is_closed={self.is_closed}, index={self.index}, parent_index={self.parent_index}"


class MotionPrimitive():
    def __init__(self, start_state, end_state):
        self.start_state = start_state
        self.end_state = end_state
        self.turning_radius = .3
        self.path = dubins.shortest_path(self.start_state, self.end_state, self.turning_radius)
        self.cost = self.path.path_length()  # + .1*abs((self.end_state[2] - self.start_state[2]) % (np.pi))
        # self.cost = rs.path_length(self.start_state, self.end_state, self.turning_radius)

    def get_sampled_position(self, step_size=1):
        samples, dists = self.path.sample_many(step_size)
        samples = np.array(samples)
        # samples = rs.path_sample(self.start_state, self.end_state, self.turning_radius, step_size)
        return samples


class StateLattice:
    def __init__(self):
        self.octomap_launch = None
        self.gt_occ_grid = None
        self.queue = []      # A priority queue of nodes as a heapq.
        self.goal_tolerance = 2.
        self.mp_length = 5
        self.num_mps_per_spatial_dim = 5
        self.path_sub = rospy.Subscriber("ground_truth_planning/move_base/GlobalPlanner/plan", Path, self.path_callback)
        self.map_sub = rospy.Subscriber("ground_truth_planning/map", OccupancyGrid, self.occ_grid_callback)
        self.path_pub = rospy.Publisher("ground_truth_planning/mp_sampled_path", Path, queue_size=10)
        self.indices_pub = rospy.Publisher("ground_truth_planning/mp_indices_list", Int16MultiArray, queue_size=10)

    def occ_grid_callback(self, msg):
        self.gt_occ_grid = msg
        info = self.gt_occ_grid.info
        self.grid_origin = np.empty((2,))
        self.grid_origin[0] = info.origin.position.x
        self.grid_origin[1] = info.origin.position.y
        self.grid_resolution = info.resolution
        self.grid_dims = (info.height, info.width)
        self.grid = np.array(self.gt_occ_grid.data).reshape(info.height, info.width)

        self.map_sub.unregister()

    def isOccupied(self, metric_point):
        ind = np.floor((metric_point - self.grid_origin)/self.grid_resolution).astype('int')
        if self.grid[ind[1], ind[0]] > 50:
            return True
        return False

    def create_nav_path(self, path):
        nav_path = Path()
        nav_path.header.stamp = rospy.Time.now()
        nav_path.header.frame_id = "world"
        for sample in path:
            ps = PoseStamped()
            ps.header = nav_path.header
            ps.pose.position.x = sample[0]
            ps.pose.position.y = sample[1]
            q = quaternion_from_euler(0, 0, sample[2])
            ps.pose.orientation.x = q[0]
            ps.pose.orientation.y = q[1]
            ps.pose.orientation.z = q[2]
            ps.pose.orientation.w = q[3]
            nav_path.poses.append(ps)
        return nav_path

    def path_callback(self, msg):
        waypoints = np.array([[pose.pose.position.x, pose.pose.position.y, euler_from_quaternion(
            [pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w])[2]] for pose in msg.poses])
        if waypoints.shape[0] > 0:
            self.start_state = waypoints[0, :]
            self.goal_state = waypoints[-1, :]
            path, sampled_path, path_cost, nodes_expanded, indices_list = self.graph_search()

            nav_path = self.create_nav_path(sampled_path)
            self.path_pub.publish(nav_path)
            ind_msg = Int16MultiArray()
            ind_msg.data = indices_list
            self.indices_pub.publish(ind_msg)
            # self.plot_path(path, sampled_path, path_cost)
            # plt.show()
            # rospy.signal_shutdown("ran graph search once")

    def heuristic(self, state):
        return np.linalg.norm(state - self.goal_state)

    def reset_graph_search(self):
        self.node_dict = {}  # A dict where key is an state and the value is a node in the queue.
        self.queue = []      # A priority queue of nodes as a heapq.
        self.neighbor_nodes = []
        self.closed_nodes = []
        node = Node(0, self.heuristic(self.start_state), self.start_state, None, None, graph_depth=0)
        self.node_dict[node.state.tobytes()] = node
        heappush(self.queue, node)

    def build_path(self, node):
        """
        Build path from start point to goal point using the goal node's parents.
        """
        path = [node.state]
        sampled_path = []
        path_cost = 0
        indices_list = []
        while node.parent is not None:
            path.append(node.parent)
            mp = node.mp
            sampled_path.append(np.array(mp.get_sampled_position(step_size=.1)))
            path_cost += mp.cost
            indices_list.append(node.index)
            node = self.node_dict[node.parent.tobytes()]
        path.reverse()
        sampled_path.reverse()
        return np.vstack(path).transpose(), np.vstack(sampled_path), path_cost, indices_list

    def plot_path(self, path, sampled_path, path_cost, ax=None):

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.start_state[0], self.start_state[1], 'o', color='lightgreen', zorder=5)
        ax.plot(self.goal_state[0], self.goal_state[1], 'or', zorder=5)

        if sampled_path is not None:
            ax.plot(sampled_path[:, 0], sampled_path[:, 1], zorder=4)
            # print(f'cost: {path_cost}')
            ax.plot(path[0, :], path[1, :], 'go', zorder=4)

        ax.add_patch(plt.Circle(self.goal_state[:2], self.goal_tolerance, color='b', fill=False, zorder=5))
        closed_nodes_states = np.array([node.state for node in self.closed_nodes]).T
        # ax.plot(closed_nodes_states[0, :], closed_nodes_states[1, :], 'm*', zorder=3)
        neighbor_nodes_states = np.array([node.state for node in self.neighbor_nodes]).T
        if neighbor_nodes_states.size > 0:
            ax.plot(neighbor_nodes_states[0, :], neighbor_nodes_states[1, :], '.',
                    color=('.8'), zorder=2, markeredgewidth=.2, markeredgecolor='k')
        x_coords = np.arange(self.grid_origin[0], self.grid_origin[0]+(self.grid_dims[0]+1)*self.grid_resolution, self.grid_resolution)
        y_coords = np.arange(self.grid_origin[1], self.grid_origin[1]+(self.grid_dims[1]+1)*self.grid_resolution, self.grid_resolution)
        plt.pcolormesh(y_coords, x_coords, -self.grid, zorder=1, cmap='gray')
        ax.set_aspect("equal")
        buf = 20
        ax.set_xlim(min(sampled_path[:, 0])-buf, max(sampled_path[:, 0])+buf)
        ax.set_ylim(min(sampled_path[:, 1])-buf, max(sampled_path[:, 1])+buf)

    def is_mp_collision_free(self, mp):
        samples = mp.get_sampled_position()
        for sample in samples:
            if self.isOccupied(sample[:2]):
                return False
        return True

    def get_neighbor_nodes(self, node):
        end_states = np.array([x for x in itertools.product(np.linspace(-self.mp_length, self.mp_length, self.num_mps_per_spatial_dim),
                                                            np.linspace(-self.mp_length, self.mp_length, self.num_mps_per_spatial_dim), np.linspace(0, 2*np.pi, 1))]) + node.state
        neighbors = []
        for i, end_state in enumerate(end_states):
            mp = MotionPrimitive(deepcopy(node.state), end_state)
            if self.is_mp_collision_free(mp):
                state = mp.end_state
                neighbor_node = Node(mp.cost + node.g, self.heuristic(state), state,
                                     node.state, mp, graph_depth=node.graph_depth+1, index=i)
                neighbors.append(neighbor_node)
        #     samp = np.array(mp.get_sampled_position(.1))
        #     if samp.shape[0] > 0:
        #         plt.plot(samp[:,0],samp[:,1])
        # plt.show()
        node.is_closed = True
        return neighbors

    def graph_search(self):
        self.reset_graph_search()

        # While queue is not empty, pop the next smallest total cost f node
        path = None
        sampled_path = None
        path_cost = None
        nodes_expanded = 0

        while not rospy.is_shutdown() and self.queue:
            node = heappop(self.queue)

            # If node has been closed already, skip.
            if node.is_closed:
                continue
            # Otherwise, expand node and for each neighbor...
            nodes_expanded += 1
            self.closed_nodes.append(node)  # for animation/plotting

            # If node is the goal node, return path.
            norm = np.linalg.norm((node.state[:2] - self.goal_state[:2]))
            if (norm <= self.goal_tolerance).all():
                print("Path found")
                path, sampled_path, path_cost, indices_list = self.build_path(node)
                break

            neighbors = self.get_neighbor_nodes(node)
            for neighbor_node in neighbors:
                old_neighbor = self.node_dict.get(neighbor_node.state.tobytes(), None)
                if old_neighbor == None or neighbor_node.g < old_neighbor.g:
                    heappush(self.queue, neighbor_node)
                    self.node_dict[neighbor_node.state.tobytes()] = neighbor_node
                    if old_neighbor != None:
                        old_neighbor.is_closed = True
                self.neighbor_nodes.append(neighbor_node)  # for plotting

        if self.queue is not None:
            # print(f"Nodes expanded: {nodes_expanded}")
            self.neighbor_nodes = np.array(self.neighbor_nodes)
            self.closed_nodes = np.array(self.closed_nodes)

        if path is None:
            print("No path found")
        return path, sampled_path, path_cost, nodes_expanded, indices_list


if __name__ == "__main__":
    rospy.init_node('waypoints_to_mps', anonymous=True)
    StateLattice()
    rospy.spin()
