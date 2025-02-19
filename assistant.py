"""
Assistant

Based on user's input, generate groups of robots and assign behaviors to them.
"""

from langchain_openai import ChatOpenAI
from typing import Dict, List, Tuple, TypedDict, Annotated
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor, StructuredChatAgent, Tool
from langchain.memory import ConversationBufferMemory

from langchain.globals import set_debug
set_debug(True)  # Shows full prompt assembly in console

# CONSTANTS
with open("mykeys/openai_api_key.txt", "r") as f:
    OPENAI_API_KEY = f.read()

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