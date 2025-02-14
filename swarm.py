"""
Swarm

This module contains the Swarm, Group, and Robot classes. 
It also has some auxiliary functions.
"""

import random
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, DBSCAN

# POSSIBLE BEHAVIORS AND PARAMS
BEHAVIORS = {
    "form_and_move": {
        "states": ["form", "rotate", "move"],
        "params": {
            "formation_shape": ["circle", "square", "triangle", "hexagon"],
            "formation_radius": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        }
    },
    "move_around": {
        "states": ["move"],
        "params": {}
    }
}

class Robot:
    """
    A single robot agent in the swarm
    """
    
    def __init__(self, idx, x, y, max_speed=0.05):
        """
        Initialize a robot at specified coordinates
        
        Args:
            idx (int): Robot index
            x (float): Initial x-position
            y (float): Initial y-position
            max_speed (float): Maximum allowed speed
        """
        self.idx = idx
        self.x = x
        self.y = y
        self.max_speed = max_speed
        self.angle = 0.0
        self.update_target(x, y)
        self.vx = 0.0
        self.vy = 0.0
        self.rotation_speed = 0.0
    
    def update_target_angle(self, target_x, target_y):
        """Update target angle for robot"""
        target_angle = np.arctan2(target_y - self.y, target_x - self.x)
        self.target_angle = target_angle
    
    def update_target(self, target_x, target_y):
        """Update target position for robot"""
        self.target_x = target_x
        self.target_y = target_y
        self.update_target_angle(target_x, target_y)

    def move(self):
        """
        Move robot towards target position with rotation and velocity control
        """
        if not self.is_robot_aligned():
            self.rotation_speed = np.sign(self.angle_diff) * max(0.1 * (abs(self.angle_diff) / np.pi), 0.005)
            # print('Robot is rotating', self.rotation_speed)
        else:
            if not self.is_robot_in_position():
                self.vx = (self.dx / self.dist) * self.max_speed
                self.vy = (self.dy / self.dist) * self.max_speed
                # print('Robot is moving', self.vx, self.vy)

        # Update position based on velocity and rotation speed
        self._update_position()

    def _calculate_distance(self):
        """Calculate distance to target position"""
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = np.hypot(dx, dy)
        return dx, dy, dist
    
    def _calculate_angle_diff(self):
        """Calculate angle difference to target angle"""
        angle_diff = (self.target_angle - self.angle + np.pi) % (2 * np.pi) - np.pi
        return angle_diff

    def is_robot_in_position(self):
        """Check if robot is close to assigned target position"""
        self.dx, self.dy, self.dist = self._calculate_distance()
        return self.dist < 0.05
    
    def is_robot_aligned(self):
        """Check if robot is aligned with assigned target angle"""
        self.angle_diff = self._calculate_angle_diff()
        return abs(self.angle_diff) < 0.01
    
    def _update_position(self):
        """
        Update robot position based on current velocity and rotation speed
        """
        # Update position
        self.x = (self.x + self.vx)
        self.y = (self.y + self.vy)
        # Update angle and wrap to [0, 2*pi]
        self.angle = (self.angle + self.rotation_speed) % (2 * np.pi)
        # Reset velocity and rotation speed after movement
        self.vx = 0.0
        self.vy = 0.0
        self.rotation_speed = 0.0

class Group:
    """
    A group of robots in the swarm that share a common behavior or task
    """
    
    def __init__(self, idx, robots):
        """
        Initialize a robot group
        
        Args:
            idx (int): Group index
            robots (list[Robot]): List of robots in the group
        """
        self.idx = idx
        self.robots = robots
        self.destination = (0, 0)
        self.bhvr = {
            "name": "", 
            "state": 0,
            "params": {},
            "data": {}
        }
        # Initialize virtual center
        self._update_virtual_center()

    def set_behavior(self, behavior_dict):
        """Set behavior for the group"""
        behavior_name = behavior_dict['name']
        behavior_params = behavior_dict['params'] if 'params' in behavior_dict else {}
        behavior_dict["data"] = {}

        if behavior_name == "form_and_move":
            # Assign formation positions
            n = len(self.robots)
            formation_pts = compute_formation_positions(n, behavior_params['formation_shape'], behavior_params['formation_radius'])
            
            # Create cost matrix
            cost_matrix = np.zeros((len(self.robots), len(formation_pts)))
            for i, robot in enumerate(self.robots):
                for j, pos in enumerate(formation_pts):
                    cost_matrix[i, j] = np.linalg.norm([robot.x - pos[0], robot.y - pos[1]])
                    
            # Solve assignment problem
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            robot_idxs = [robot.idx for robot in self.robots]

            # Set formation positions in behavior data
            behavior_dict["data"]["formation_positions"] = {int(i): formation_pts[j].tolist() for i, j in zip(robot_idxs, col_ind)}

            # Initialize behavior state
            behavior_dict["state"] = 0
            self.bhvr = behavior_dict

        elif behavior_name == "move_around":
            # Initialize behavior state
            behavior_dict["state"] = 0
            self.bhvr = behavior_dict

    def step(self):
        """Perform one simulation step for the group"""
        bhvr = self.bhvr

        # Calculate movement based on current behavior
        match bhvr["name"]:
            case "form_and_move":
                match bhvr["state"]:
                    case 0:
                        for robot in self.robots:
                            robot.update_target(
                                self.virtual_center[0] + bhvr["data"]["formation_positions"][robot.idx][0],
                                self.virtual_center[1] + bhvr["data"]["formation_positions"][robot.idx][1]
                            )
                            robot.move()
                        
                        # Check transition to next state
                        if all(r.is_robot_in_position() for r in self.robots):
                            bhvr["state"] = 1
                    
                    case 1:
                        for robot in self.robots:
                            robot.update_target_angle(
                                bhvr["params"]["destination"][0] + bhvr["data"]["formation_positions"][robot.idx][0],
                                bhvr["params"]["destination"][1] + bhvr["data"]["formation_positions"][robot.idx][1]
                            )
                            robot.move()
                        
                        # Check transition to next state
                        if all(r.is_robot_aligned() for r in self.robots):
                            bhvr["state"] = 2

                    case 2:
                        for robot in self.robots:
                            robot.update_target(
                                bhvr["params"]["destination"][0] + bhvr["data"]["formation_positions"][robot.idx][0],
                                bhvr["params"]["destination"][1] + bhvr["data"]["formation_positions"][robot.idx][1]
                            )
                            robot.move()

                        # Update virtual center 
                        self._update_virtual_center()

                        # Check transition to next state
                        if all(r.is_robot_in_position() for r in self.robots):
                            bhvr["state"] = -1

            case "move_around":
                for robot in self.robots:
                    # If they havent reached their target, move them
                    if not robot.is_robot_in_position():
                        robot.move()
                    # If they have reached their target, pick a new target
                    else: 
                        robot.update_target(robot.x + random.uniform(-0.5, 0.5), robot.y + random.uniform(-0.5, 0.5))

    def _update_robot_targets(self):
        """Update robot targets based on current behavior"""

    def _update_virtual_center(self):
        """Update virtual center to current robot positions average"""
        self.virtual_center = (
            np.mean([r.x for r in self.robots]),
            np.mean([r.y for r in self.robots])
        )

class Swarm:
    """
    A swarm of robots that can move and interact with each other forming groups
    """
    
    def __init__(self, robots):
        """
        Initialize a swarm of robots
        
        Args:
            robots (list[Robot]): List of robots in the swarm
        """
        self.robots = robots
        self.groups = [Group(i, [robots[i]]) for i in range(len(robots))]
        self.formation_shapes = ['circle', 'square', 'triangle', 'hexagon']

    def gen_groups_by_lists_of_ids(self, lists_of_ids):
        """
        Generate groups based on lists of robot IDs

        Args:
            lists_of_ids (list[list[int]]): List of lists of robot IDs
        """
        self.groups = [Group(i, [self.robots[i] for i in ids]) for i, ids in enumerate(lists_of_ids)]

    def gen_groups_by_clustering(self, num_groups):
        """
        Generate groups based on clustering and current robot positions
        
        Args:
            num_groups (int): Number of groups to create
        """
        x = [r.x for r in self.robots]
        y = [r.y for r in self.robots]
        coords = np.column_stack((x, y))

        kmeans = KMeans(n_clusters=num_groups, n_init=10).fit(coords)
        
        groups = []
        for group_idx in range(num_groups):
            robot_indices = np.where(kmeans.labels_ == group_idx)[0]
            group_robots = [self.robots[i] for i in robot_indices]
            group = Group(group_idx, group_robots)
            group.set_behavior({
                "name": "form_and_move",
                "params": {
                    "formation_shape": random.choice(self.formation_shapes),
                    "formation_radius": random.uniform(0.5, 3.0),
                    "destination": (random.uniform(0, 8), random.uniform(0, 8))
                }
            })
            groups.append(group)
            
        self.groups = groups
    
    def step(self):
        """Perform one simulation step for entire swarm"""
        for group in self.groups:
            if group.bhvr["state"] == -1:
                continue
                
            group.step()


# AUXILIARY FUNCTIONS

def compute_formation_positions(n, formation_shape, formation_radius):
    """Calculate equally shaped positions along a shape's perimeter"""
    if formation_shape == 'circle':
        return compute_circle_positions(n, formation_radius)
    elif formation_shape == 'square':
        return compute_square_positions(n, formation_radius*2)
    elif formation_shape == 'triangle':
        return compute_triangle_positions(n, formation_radius*2)
    elif formation_shape == 'hexagon':
        return compute_hexagon_positions(n, formation_radius)

# Formation calculation functions   
def compute_circle_positions(n, radius):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack((x, y))

def compute_square_positions(n, side_length):
    positions = []
    for i in range(n):
        t = i * (4.0 / n)
        side = int(t)
        pos_in_side = t - side
        if side == 0:
            x = -0.5 + pos_in_side
            y = 0.5
        elif side == 1:
            x = 0.5
            y = 0.5 - pos_in_side
        elif side == 2:
            x = 0.5 - pos_in_side
            y = -0.5
        else:
            x = -0.5
            y = -0.5 + pos_in_side
        positions.append([x * side_length, y * side_length])
    return np.array(positions)

def compute_triangle_vertices(side_length):
    h = np.sqrt(3) / 6 * side_length
    return np.array([[0, -2 * h], [0.5 * side_length, h], [-0.5 * side_length, h]])

def compute_triangle_positions(n, side_length):
    vertices = compute_triangle_vertices(side_length)
    positions = []
    for i in range(n):
        pos = i * (3 / n)
        if pos < 1:
            point = vertices[0] + pos * (vertices[1] - vertices[0])
        elif pos < 2:
            point = vertices[1] + (pos - 1) * (vertices[2] - vertices[1])
        else:
            point = vertices[2] + (pos - 2) * (vertices[0] - vertices[2])
        positions.append(point)
    return np.array(positions)

def compute_hexagon_vertices(side_length):
    """Compute the vertices of a regular hexagon centered at (0,0) with given side length."""
    h = np.sqrt(3) / 2 * side_length  # Height of an equilateral triangle (half hexagon height)
    
    return np.array([
        [side_length, 0],          # Right
        [0.5 * side_length, h],    # Top-right
        [-0.5 * side_length, h],   # Top-left
        [-side_length, 0],         # Left
        [-0.5 * side_length, -h],  # Bottom-left
        [0.5 * side_length, -h]    # Bottom-right
    ])

def compute_hexagon_positions(n, side_length):
    """Distribute n points evenly along the perimeter of a hexagon with given side length."""
    vertices = compute_hexagon_vertices(side_length)
    positions = []
    
    for i in range(n):
        pos = i * (6 / n)  # Normalize position around the hexagon (6 edges)
        edge_index = int(pos)  # Determine which edge the point belongs to
        t = pos - edge_index  # Fractional position along the edge
        
        point = vertices[edge_index] + t * (vertices[(edge_index + 1) % 6] - vertices[edge_index])
        positions.append(point)
    
    return np.array(positions)