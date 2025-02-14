import logging
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from threading import Thread, Lock
import random
import time
import numpy as np

# Import Swarm and Robot classes
from swarm import Swarm, Robot

# Create the Flask application
app = Flask(__name__)
CORS(app)

# Logging configuration
# Get the werkzeug logger
log = logging.getLogger('werkzeug')

class NoSuccessFilter(logging.Filter):
    def filter(self, record):
        return "200" not in record.getMessage()  # Filters out 200 OK logs

log.addFilter(NoSuccessFilter())

# Simulation variables
simulation_lock = Lock()
simulation_state = {
    "running": False,
    "swarm": None,
    "current_step": 0
}

# SIMULATION ENV
ARENA_WIDTH = 20
ARENA_HEIGHT = 20
FORMATION_RADIUS = 2

# CONSTANTS
NUMBER_OF_ROBOTS = 30
NUMBER_OF_GROUPS = 3
MAX_SPEED = 0.05
FORMATION_SHAPES = ['circle', 'square', 'triangle', 'hexagon']
FIELD_START_X = 1
FIELD_START_Y = 0
FIELD_WIDTH = 5
FIELD_HEIGHT = 2

# Initialize robot positions with some distance from the edge
def initialize_robot_positions(n, x_min=0, y_min=0, x_max=ARENA_WIDTH, y_max=ARENA_HEIGHT, distance_from_edge=FORMATION_RADIUS):
    x = np.random.uniform(x_min + distance_from_edge, x_max - distance_from_edge, n)
    y = np.random.uniform(y_min + distance_from_edge, y_max - distance_from_edge, n)
    return x, y

# Initialize destination positions with some distance from the edge
def initialize_destinations(n):
    x = np.random.uniform(FORMATION_RADIUS, ARENA_WIDTH-FORMATION_RADIUS, n)
    y = np.random.uniform(FORMATION_RADIUS, ARENA_HEIGHT-FORMATION_RADIUS, n)
    return x, y

# Initialize the swarm
def initialize_swarm(reset=False):
    if reset:
        swarm = simulation_state["swarm"]
        swarm.gen_groups_by_clustering(NUMBER_OF_GROUPS)
        dest_x, dest_y = initialize_destinations(NUMBER_OF_GROUPS)
        for group in swarm.groups:
            group.set_behavior({
                "name": "form_and_move",
                "params": {
                    "formation_shape": random.choice(FORMATION_SHAPES),
                    "formation_radius": random.uniform(0.5, 1.0),
                    "destination": (dest_x[group.idx], dest_y[group.idx])
                }
            })
    else:
        x, y = initialize_robot_positions(NUMBER_OF_ROBOTS, FIELD_START_X, FIELD_START_Y, 
                                        FIELD_START_X + FIELD_WIDTH, FIELD_START_Y + FIELD_HEIGHT)
        dest_x, dest_y = initialize_destinations(NUMBER_OF_GROUPS)
        robots = [Robot(idx, x, y) for idx, (x, y) in enumerate(zip(x, y))]
        swarm = Swarm(robots)
        for group in swarm.groups:
            group.set_behavior({
                "name": "move_around",
                "params": {}
            })
    
    return swarm

simulation_state["swarm"] = initialize_swarm()

# Simulation loop
def simulation_loop():
    while True:
        with simulation_lock:
            if simulation_state["running"]:
                simulation_state["swarm"].step()
                simulation_state["current_step"] += 1
        time.sleep(0.01)

sim_thread = Thread(target=simulation_loop)
sim_thread.daemon = True
sim_thread.start()

### API Endpoints ###

@app.route('/state')
def get_state():
    """Return the current state of the simulation"""
    with simulation_lock:
        groups = []
        for group in simulation_state["swarm"].groups:
            robots = [{"idx": r.idx, "x": r.x, "y": r.y, "angle": r.angle, 
                      "target_x": r.target_x, "target_y": r.target_y} 
                     for r in group.robots]
            groups.append({
                "idx": group.idx,
                "virtual_center": group.virtual_center,
                "state": group.bhvr["state"],
                "bhvr": group.bhvr,
                "robots": robots
            })
        return jsonify({
            "running": simulation_state["running"],
            "current_step": simulation_state["current_step"],
            "groups": groups,
            "arena": {"width": ARENA_WIDTH, "height": ARENA_HEIGHT}
        })

@app.route('/control', methods=['POST'])
def control():
    """Handle control commands from the client"""
    command = request.json.get('command')
    with simulation_lock:
        if command == 'reset':
            simulation_state["swarm"] = initialize_swarm(reset=True)
            simulation_state["current_step"] = 0
        elif command == 'pause':
            simulation_state["running"] = not simulation_state["running"]
        elif command == 'stop':
            print("Stopping simulation - What should we do here?")
    return jsonify({"status": "ok"})

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages from the client"""
    message = request.json.get('message')
    return jsonify({"status": "ok"})

@app.route('/')
def index():
    """Serve the index.html file"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_file(path):
    """Serve static files from the 'static' directory"""
    return send_from_directory('./public', path)