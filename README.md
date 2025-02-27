# Obedient Swarm

**Obedient Swarm** is a web-based simulation where you can interact with and control a swarm of drones using natural language commands. Powered by a Large Language Model (LLM) agent, your messages are interpreted, and the corresponding actions are executed in real-time within the simulation.

## Features

-   **Natural Language Control:** Command the swarm using simple, intuitive messages via a chat interface.
-   **LLM-Powered Agent:** An LLM agent interprets your commands and translates them into actionable instructions for the drones.
-   **Real-Time Visualization:** Watch the drones' behavior in real-time as they respond to your commands.
-   **Group Management:** Create and manage drone groups with different behaviors.
-   **Variety of Behaviors:** Assign formations and trajectories to groups, or let them move freely.
-   **Interactive Web App:** User-friendly web interface for easy interaction and visualization.
-   **Docker Support:** Easily run the application in a containerized environment.

## Getting Started

### Prerequisites

-   Docker and Docker Compose
-   An OpenAI API key (store it in `mykeys/openai_api_key.txt`).

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/alejp1998/obedient-swarm.git
    cd obedient-swarm
    ```

2.  Set your OpenAI API key:

    Create a directory named `mykeys` and a file named `openai_api_key.txt` inside it. Place your API key in the `openai_api_key.txt` file. **Ensure this directory is not committed to version control.**

3.  Build and run the application using Docker Compose:

    ```bash
    docker-compose up --build
    ```

4.  Open your web browser and navigate to `http://localhost:9000`.

### Usage

1.  **Chat Interface:** Use the chat interface to send commands to the swarm. For example:
    -   "Group the drones into 3 groups."
    -   "Make group 1 move in a circle formation along the river."
    -   "Make group 2 move around randomly."

2.  **Visualization:** Observe the drones' movements and formations in the real-time visualization.

3.  **Experiment:** Try different commands and explore the various behaviors of the swarm.

## Code Structure

-   `app.py`: Flask application that handles the web interface and simulation logic.
-   `swarm.py`: Defines the `Swarm`, `Group`, and `Robot` classes, as well as the LLM agent (`SwarmAgent`).
-   `public/`: Contains static files for the web application (HTML, CSS, JavaScript).
-   `requirements.txt`: Lists the Python dependencies.
-   `mykeys/`: Contains your OpenAI API key. **Do not commit this directory to version control.**
-   `docker-compose.yaml`: Docker Compose configuration for running the application.
-   `Dockerfile-webapp`: Dockerfile for building the web application container.

## Docker Setup

The application is containerized using Docker. The `docker-compose.yaml` file defines the `webapp` service, which builds the application using the `Dockerfile-webapp`.

## License
This project is licensed under the MIT License.