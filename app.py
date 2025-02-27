import logging
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from threading import Thread, Lock
import random
import time
import numpy as np

# Import Swarm and Robot classes
from swarm import SwarmAgent, Swarm, Robot

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

# SIMULATION ENV
ARENA_WIDTH = 20
ARENA_HEIGHT = 20
FORMATION_RADIUS = 2

# SIMULATION SETTINGS
NUMBER_OF_ROBOTS = 20
NUMBER_OF_GROUPS = 3
MAX_SPEED = 0.02
FORMATION_SHAPES = ['circle', 'square', 'triangle', 'hexagon']
FIELD_START_X = 1
FIELD_START_Y = 0
FIELD_WIDTH = 7
FIELD_HEIGHT = 5

# Simulation functions
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
def initialize_swarm():
    x, y = initialize_robot_positions(NUMBER_OF_ROBOTS, FIELD_START_X, FIELD_START_Y, 
                                    FIELD_START_X + FIELD_WIDTH, FIELD_START_Y + FIELD_HEIGHT)
    robots = [Robot(idx, x, y, max_speed=MAX_SPEED) for idx, (x, y) in enumerate(zip(x, y))]
    swarm = Swarm(robots)
    
    return swarm

# Swarm Initialization
swarm = initialize_swarm()

# Simulation variables initialization
simulation_lock = Lock()
simulation_state = {
    "running": True,
    "current_step": 0,
    "swarm": swarm
}

# Chat variables
initial_messages = [
    {"role": "ai", "content": "Hello! Welcome to **Obedient Swarm** ;)"},
    {"role": "ai", "content": "Tell me how to group the drones and what behaviors the groups should have, and I'll do it for you."},
]

# Agent Initialization
agent = SwarmAgent(app, swarm)

# Agent variables initialization
agent_state = {
    "agent": agent,
    "messages": initial_messages.copy()
}

# Send message to the agent
def send_message(message):
    agent_state["messages"].append({"role": "user", "content": message})
    agent_state["messages"].append({"role": "ai", "content": "Waiting for AI response..."})
    response_content = agent_state["agent"].send_message(message)
    agent_state["messages"][-1] = {"role": "ai", "content": response_content}


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
            simulation_state["swarm"] = initialize_swarm()
            simulation_state["current_step"] = 0
            agent_state["agent"] = SwarmAgent(app, simulation_state["swarm"])
            agent_state["messages"] = initial_messages.copy()
        elif command == 'pause':
            simulation_state["running"] = not simulation_state["running"]
    return jsonify({"status": "ok"})

@app.route('/message', methods=['POST'])
def message():
    """Handle a new user message"""
    user_message = request.json.get('message')
    app.logger.info(f"User message: {user_message}")
    send_message(user_message)
    return jsonify({"status": "ok"})

@app.route('/chat')
def chat():
    """Return the current state of the chat"""
    return jsonify(agent_state["messages"])

@app.route('/')
def index():
    """Serve the index.html file"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_file(path):
    """Serve static files from the 'static' directory"""
    return send_from_directory('./public', path)