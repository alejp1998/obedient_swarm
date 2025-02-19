import logging
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from threading import Thread, Lock
import random
import time
import numpy as np

from langchain_openai import ChatOpenAI
from typing import Dict, List, Tuple, TypedDict, Annotated
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor, StructuredChatAgent, Tool
from langchain.memory import ConversationBufferMemory

from langchain.globals import set_debug
set_debug(True)  # Shows full prompt assembly in console

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

# CONSTANTS
with open("mykeys/openai_api_key.txt", "r") as f:
    OPENAI_API_KEY = f.read()
# Read system prompt from system_prompt.txt
with open("system_prompt.txt", "r") as f:
    SYSTEM_PROMPT_TEMPLATE = f.read()

# SIMULATION ENV
ARENA_WIDTH = 20
ARENA_HEIGHT = 20
FORMATION_RADIUS = 2

# SIMULATION SETTINGS
NUMBER_OF_ROBOTS = 50
NUMBER_OF_GROUPS = 3
MAX_SPEED = 0.05
FORMATION_SHAPES = ['circle', 'square', 'triangle', 'hexagon']
FIELD_START_X = 1
FIELD_START_Y = 0
FIELD_WIDTH = 5
FIELD_HEIGHT = 2

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

# Simulation variables
simulation_lock = Lock()
simulation_state = {
    "running": False,
    "swarm": initialize_swarm(),
    "current_step": 0
}

# Chat variables
messages = [
    {"role": "ai", "content": "Hello! Welcome to Obedient Swarm.\n"},
    {"role": "ai", "content": "Tell me how to group the drones and what behaviors they should have and I'll do it for you."},
    {"role": "ai", "content": "Example Command 1: I want to group the drones in 4 groups"},
    {"role": "ai", "content": "Example Command 2: Group 1 should form in a circle of radius 1.0 and move towards the lake"},
]

# Agent Configuration

## Agent tools
@tool
def gen_groups_by_lists_of_ids(lists_of_ids: List[List[int]]):
    """Group drones by explicit lists of IDs. Input should be list of lists containing integers. 
    Integers range from 0 to number of robots - 1. Good for deterministic grouping of drones, ensuring each group has exactly the robots we want."""

    if len(lists_of_ids) < 0:
        return "Invalid number of groups"
    
    # Generate groups by lists of IDs
    simulation_state["swarm"].gen_groups_by_lists_of_ids(lists_of_ids)
    
    return f"Drones grouped successfully in {len(lists_of_ids)} groups"


@tool
def gen_groups_by_clustering(num_groups: int):
    """Group drones into a specified number of groups using proximity clustering. 
    The resulting number of robots in each group is not predictable and depends on the proximity of the robots."""
    # Check that number is above 0 and under swarm number of robots
    if num_groups < 0 or num_groups > len(simulation_state["swarm"].robots):
        return "Invalid number of groups"
    
    # Generate groups by clustering
    simulation_state["swarm"].gen_groups_by_clustering(num_groups)
    
    return f"Drones grouped successfully in {num_groups} groups"


@tool
def assign_move_around_behavior_to_group(group_idx: int):
    """Assign move_around behavior to a group."""
    simulation_state["swarm"].assign_move_around_behavior_to_group(group_idx)
    return f"move_around behavior assigned to group {group_idx}"


@tool
def assign_form_and_move_behavior_to_group(group_idx: int, formation_shape: str, formation_radius: float, destination: Tuple[float, float]):
    """Assign form_and_move behavior to a group. Valid radius values are between 0.5 and 2.0. Valid formation shapes are circle, square, triangle, and hexagon. Valid destinations coordinates are between 0 and 20."""
    if formation_radius < 0.5 or formation_radius > 2.0:
        return "Invalid radius value"
    if formation_shape not in FORMATION_SHAPES:
        return "Invalid formation shape"
    if destination[0] < 0 or destination[0] > 20 or destination[1] < 0 or destination[1] > 20:
        return "Invalid destination coordinates"
    
    simulation_state["swarm"].assign_form_and_move_behavior_to_group(group_idx, formation_shape, formation_radius, destination)
    return f"form_and_move behavior assigned to group {group_idx} with formation shape {formation_shape}, radius {formation_radius}, and destination {destination}"

## Initialize components
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
tools = [gen_groups_by_lists_of_ids, gen_groups_by_clustering, assign_move_around_behavior_to_group, assign_form_and_move_behavior_to_group]
memory = ConversationBufferMemory(
    memory_key="history",
    input_key="input",  # Explicitly declare primary input
    return_messages=True
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE),
    MessagesPlaceholder("history"),
    HumanMessagePromptTemplate.from_template("{input}"),
])

## Create agent 
agent = StructuredChatAgent.from_llm_and_tools(
    llm=llm,
    tools=tools,
    memory=memory,
    prompt=prompt,
    input_variables=["input", "chat_history", "tool_names", "tools", "robot_idxs", "group_idxs"]
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

def send_message(message):
    messages.append({"role": "user", "content": message})
    # response = agent_executor.invoke({
    #     "robot_idxs": [robot.idx for robot in simulation_state["swarm"].robots],
    #     "input": message,
    # })
    # app.logger.info(response["output"])
    # messages.append({"role": "ai", "content": response["output"]})
    for chunk in agent_executor.stream({
        "robot_idxs": ", ".join(map(str, [robot.idx for robot in simulation_state["swarm"].robots])),
        "group_idxs": ", ".join(map(str, [group.idx for group in simulation_state["swarm"].groups])),
        "input": message,
    }):
        app.logger.info(chunk)
        if "agent" in chunk:
            message_content = chunk["agent"]["messages"][0].content.text
            messages.append({"role": "ai", "content": message_content})
            
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
            simulation_state["swarm"] = initialize_swarm(reset=False)
            simulation_state["current_step"] = 0
        elif command == 'pause':
            simulation_state["running"] = not simulation_state["running"]
        elif command == 'stop':
            print("Stopping simulation - What should we do here?")
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
    return jsonify(messages)

@app.route('/')
def index():
    """Serve the index.html file"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_file(path):
    """Serve static files from the 'static' directory"""
    return send_from_directory('./public', path)