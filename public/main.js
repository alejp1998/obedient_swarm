// main.js

// -------------------- Helper Functions --------------------
function sleep(ms) {
  return new Promise(resolve => {
    setTimeout(resolve, ms);
  });
}

// -------------------- Constants --------------------
const TIME_STEP = 10;


// -------------------- Visualization constants --------------------
const COLORS = [
  '#8B5CF6', // Royal purple
  '#FF6B6B', // Vibrant coral (replaces red)
  '#10B981', // Emerald green (better than basic green)
  '#F59E0B', // Deep orange (more sophisticated)
  '#3B82F6', // Bright sapphire blue
  '#14B8A6', // Tropical teal (better than cyan)
  '#EC4899', // Raspberry pink (modern magenta alternative)
  '#EAB308'  // Gold yellow (less harsh than plain yellow)
];

// -------------------- Canvas and Simulation Data --------------------
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Simulation variables
let arena = {};
let n_updates = 0;
let current_step = 0;
let running = false;
let simData = {};
let groups = [];

// -------------------- Canvas Sizing --------------------
function setCanvasSize() {
  const canvasContainer = document.getElementById('canvasContainer');
  const containerWidth = canvasContainer.clientWidth;
  const containerHeight = canvasContainer.clientHeight;
  
  // Original scaling logic
  if (containerWidth > containerHeight) {
    canvas.style.width = containerHeight + 'px';
    canvas.style.height = containerHeight + 'px';
  } else {
    canvas.style.width = containerWidth + 'px';
    canvas.style.height = containerWidth + 'px';
  }

  // Fixed positioning calculations
  const canvasDisplayWidth = parseInt(canvas.style.width);
  const canvasDisplayHeight = parseInt(canvas.style.height);
  
  canvas.style.position = 'absolute';
  canvas.style.left = `${(containerWidth - canvasDisplayWidth) / 2}px`;
  canvas.style.top = `${(containerHeight - canvasDisplayHeight) / 2}px`;

  // Keep your original render resolution (consider adding devicePixelRatio scaling)
  canvas.width = arena.width * 100;
  canvas.height = arena.height * 100;
}


// -------------------- Helper Functions for Shapes --------------------

// Compute vertices for an equilateral triangle (centered at 0,0)
function computeTriangleVertices(side) {
  const R = side / Math.sqrt(3);
  const vertices = [];
  for (let i = 0; i < 3; i++) {
    const angle = -Math.PI / 2 + i * (2 * Math.PI / 3);
    vertices.push([R * Math.cos(angle), R * Math.sin(angle)]);
  }
  return vertices;
}

// Compute vertices for a regular hexagon (centered at 0,0)
function computeHexagonVertices(r) {
  const vertices = [];
  for (let i = 0; i < 6; i++) {
    const angle = Math.PI / 6 + i * (Math.PI / 3);
    vertices.push([r * Math.cos(angle), r * Math.sin(angle)]);
  }
  return vertices;
}

// -------------------- Drawing Functions --------------------

// Draw a dashed line between two points
function drawDashedLine(ctx, start, end, dashLength = 5) {
  ctx.save();
  ctx.setLineDash([dashLength, dashLength]);
  ctx.beginPath();
  ctx.moveTo(start[0], start[1]);
  ctx.lineTo(end[0], end[1]);
  ctx.stroke();
  ctx.restore();
}

// Draw grid lines (each cell is 100 pixels)
function drawGrid() {
  ctx.save();
  ctx.strokeStyle = 'rgba(200,200,200,1)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= arena.width; i++) {
    ctx.beginPath();
    ctx.moveTo(i * 100, 0);
    ctx.lineTo(i * 100, arena.height * 100);
    ctx.stroke();
  }
  for (let j = 0; j <= arena.height; j++) {
    ctx.beginPath();
    ctx.moveTo(0, j * 100);
    ctx.lineTo(arena.width * 100, j * 100);
    ctx.stroke();
  }
  ctx.restore();
}

// Draw formation outline for a group (circle, square, triangle, hexagon)
function drawFormation(group) {
  const { formation_shape, formation_radius } = group.bhvr.params;
  const center = group.virtual_center;
  const color = group.robots.length > 1 ? COLORS[group.idx%COLORS.length] : '#000';
  const cx = center[0] * 100;
  const cy = center[1] * 100;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 5]);

  if (formation_shape === 'circle') {
    const radius = formation_radius * 100;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
    ctx.stroke();
  } else if (formation_shape === 'square') {
    const size = formation_radius * 2 * 100;
    ctx.strokeRect(cx - size / 2, cy - size / 2, size, size);
  } else if (formation_shape === 'triangle') {
    const vertices = computeTriangleVertices(formation_radius * 2);
    const points = vertices.map(v => [cx + v[0] * 100, cy + v[1] * 100]);
    for (let i = 0; i < 3; i++) {
      drawDashedLine(ctx, points[i], points[(i + 1) % 3]);
    }
  } else if (formation_shape === 'hexagon') {
    const vertices = computeHexagonVertices(formation_radius);
    const points = vertices.map(v => [cx + v[0] * 100, cy + v[1] * 100]);
    for (let i = 0; i < 6; i++) {
      drawDashedLine(ctx, points[i], points[(i + 1) % 6]);
    }
  }
  ctx.setLineDash([]);
  ctx.restore();
}

// Draw centered text
function drawCenteredText(text, x, y) {
  ctx.fillText(text, x * 100, y * 100);
}

// Draw map elements (river, lake, road, bridge, forest, field, town, farm) and labels
function drawMap() {
  ctx.save();
  
  // Configurable positions
  const riverPos = { x: 9.5, y: 0, width: 1, height: 15.5 };      // Vertical river
  const lakePos = { x: 6, y: 15.0, width: 8.0, height: 5.0 };     // Bottom lake
  const roadPos = { x: 0, y: 9.5, width: 20, height: 1 };         // Full-width road
  const bridgePos = { x: 8.5, y: 9, width: 3, height: 2 };     // Centered bridge
  const forestPos = { x: 11, y: 1, width: 5, height: 5 };    // Right-side forest
  const fieldPos = { x: 1.0, y: 0, width: 7.0, height: 5.0 };    // Left field
  const townPos = { x: 1.5, y: 6.5, width: 5, height: 7 };         // Central town
  const farmPos = { x: 15.0, y: 7.5, width: 4, height: 5 };      // Right farm


  // Drawing elements
  ctx.fillStyle = 'rgb(135,206,235)'; // light blue
  ctx.fillRect(riverPos.x * 100, riverPos.y * 100, riverPos.width * 100, riverPos.height * 100);

  ctx.fillStyle = 'rgb(69, 69, 251)'; // darker blue
  ctx.fillRect(lakePos.x * 100, lakePos.y * 100, lakePos.width * 100, lakePos.height * 100);

  ctx.fillStyle = 'rgb(100,100,100)'; // gray
  ctx.fillRect(roadPos.x * 100, roadPos.y * 100, roadPos.width * 100, roadPos.height * 100);

  ctx.fillStyle = 'rgb(100,100,100)'; // same as road
  ctx.fillRect(bridgePos.x * 100, bridgePos.y * 100, bridgePos.width * 100, bridgePos.height * 100);

  ctx.fillStyle = 'rgb(34,139,34)'; // forest green
  ctx.fillRect(forestPos.x * 100, forestPos.y * 100, forestPos.width * 100, forestPos.height * 100);

  ctx.fillStyle = 'rgb(144,238,144)'; // light green
  ctx.fillRect(fieldPos.x * 100, fieldPos.y * 100, fieldPos.width * 100, fieldPos.height * 100);

  ctx.fillStyle = 'rgb(211,211,211)'; // light gray
  ctx.fillRect(townPos.x * 100, townPos.y * 100, townPos.width * 100, townPos.height * 100);

  ctx.fillStyle = 'rgb(255,165,0)'; // light orange
  ctx.fillRect(farmPos.x * 100, farmPos.y * 100, farmPos.width * 100, farmPos.height * 100);

  // Labels
  ctx.fillStyle = '#000';
  ctx.font = '24px Arial';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  drawCenteredText("River", riverPos.x + riverPos.width / 2, riverPos.y + riverPos.height / 2);
  drawCenteredText("Lake",  lakePos.x + lakePos.width / 2, lakePos.y + lakePos.height / 2);
  // drawCenteredText("Road",  roadPos.x + roadPos.width / 2, roadPos.y + roadPos.height / 2);
  drawCenteredText("Bridge",  bridgePos.x + bridgePos.width / 2, bridgePos.y + bridgePos.height / 2);
  drawCenteredText("Forest",  forestPos.x + forestPos.width / 2, forestPos.y + forestPos.height / 2);
  drawCenteredText("Field",  fieldPos.x + fieldPos.width / 2, fieldPos.y + fieldPos.height / 2);
  drawCenteredText("Town",  townPos.x + townPos.width / 2, townPos.y + townPos.height / 2);
  drawCenteredText("Farm",  farmPos.x + farmPos.width / 2, farmPos.y + farmPos.height / 2);

  ctx.restore();
}

// Draw a destination marker (a cross) for the group
function drawDestination(group) {
  const trajectory = group.bhvr.params.trajectory;
  if (!trajectory) return;

  for (let i = 0; i < trajectory.length; i++) {
    const dest = trajectory[i];
    const color = group.robots.length > 1 ? COLORS[group.idx%COLORS.length] : '#000';
    const px = dest[0] * 100;
    const py = dest[1] * 100;
    const denom = 5;
    const size = 100 / denom

    // Draw cross shape
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(px - size, py - size);
    ctx.lineTo(px + size, py + size);
    ctx.moveTo(px + size, py - size);
    ctx.lineTo(px - size, py + size);
    ctx.stroke();
    ctx.restore();

    // Draw index label with a circle background
    ctx.save();
    ctx.fillStyle = color;
    ctx.font = '30px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(i, px, py - size*2);
    ctx.restore();
  }
}

// Draw a robot with a directional line, index label, and target cross
function drawRobot(robot, color) {
  const px = robot.x * 100;
  const py = robot.y * 100;
  const endX = px + Math.cos(robot.angle) * 30;
  const endY = py + Math.sin(robot.angle) * 30;
  const targetX = robot.target_x * 100;
  const targetY = robot.target_y * 100;

  ctx.save();
  // Directional line
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(px, py);
  ctx.lineTo(endX, endY);
  ctx.stroke();

  // Robot body (circle with inner white circle)
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(px, py, 15, 0, 2 * Math.PI);
  ctx.fill();
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  ctx.arc(px, py, 12, 0, 2 * Math.PI);
  ctx.fill();

  // Robot index label
  ctx.fillStyle = '#000';
  ctx.font = '18px Arial';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(robot.idx, px, py);

  // Target marker (small cross)
  const denom = 10;
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(targetX - 100 / denom, targetY - 100 / denom);
  ctx.lineTo(targetX + 100 / denom, targetY + 100 / denom);
  ctx.moveTo(targetX + 100 / denom, targetY - 100 / denom);
  ctx.lineTo(targetX - 100 / denom, targetY + 100 / denom);
  ctx.stroke();
  ctx.restore();
}


// Draw simulation status overlay (header and each group’s status)
function drawStatus() {
  ctx.save();
  ctx.fillStyle = '#000';
  ctx.font = 'bold 26px Arial';
  ctx.textAlign = 'left';
  let yOffset = 0;
  // ctx.fillText(`Simulation Step: ${current_step}`, 10, yOffset);
  yOffset += 25;

  // Create behavior description string
  let behaviorString = ""
  groups.forEach((group, idx) => {
    switch (group.bhvr.name) {
      case "move_around":
        behaviorString = "Move Around";
        break;
      case "form_and_follow_trajectory":
        behaviorString = "Form & Follow Trajectory (" + group.bhvr.params.formation_shape + ")" + " [" + group.bhvr.params.trajectory.join(",") + "]";
        break;
      default:
        behaviorString = "None";
    }

    const robotsInGroup = Array.from(group.robots).map(robot => robot.idx).join(',');
    const statusText = `G${group.idx} [${robotsInGroup}] -> ${behaviorString}`;
    const color = group.robots.length > 1 ? COLORS[group.idx%COLORS.length] : '#000';
    ctx.fillStyle = color;
    ctx.fillText(statusText, 10, yOffset);
    yOffset += 30;
  });
  ctx.restore();
}

// Main update function: clear canvas and redraw every element
function updateDisplay() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawMap();
  drawGrid();
  groups.forEach(group => {
    // drawFormation(group);
    drawDestination(group);
    group.robots.forEach(robot => {
      const color = group.robots.length > 1 ? COLORS[group.idx%COLORS.length] : '#000';
      drawRobot(robot, color);
    });
  });
  drawStatus();
}

// -------------------- Expandable Tree View Functions --------------------
const nodeStates = {};
const elementMap = new Map();
const defaultState = "expanded";

function reconcileTree(parentEl, data, parentPath = "") {
  const existingNodes = new Map();
  Array.from(parentEl.children).forEach(li => {
    const path = li.dataset.path;
    if (path) existingNodes.set(path, li);
  });

  const keys = Object.keys(data).sort((keyA, keyB) => {
    const isParent = value => typeof value === "object" && value !== null;
    const parentStatusDiff = Number(isParent(data[keyA])) - Number(isParent(data[keyB]));
  
    // Natural sort for alphanumeric keys using Intl.Collator
    const collator = new Intl.Collator(undefined, {
      numeric: true,
      sensitivity: 'base'
    });
  
    return parentStatusDiff || collator.compare(keyA, keyB);
  });  

  // Process current nodes (first over ones without children)
  const usedPaths = new Set();
  for (const key of keys) {
    if (!data.hasOwnProperty(key)) continue;
    
    const currentPath = parentPath ? `${parentPath}.${key}` : key;
    usedPaths.add(currentPath);
    
    let li = existingNodes.get(currentPath) || document.createElement('li');
    if (!li.parentElement) parentEl.appendChild(li);
    
    li.dataset.path = currentPath;
    elementMap.set(currentPath, li);

    if (typeof data[key] === "object" && data[key] !== null) {
      updateObjectNode(li, key, data[key], currentPath);
    } else {
      updateLeafNode(li, key, data[key], currentPath);
    }
  }

  // Remove deleted nodes
  existingNodes.forEach((li, path) => {
    if (!usedPaths.has(path)) {
      li.remove();
      elementMap.delete(path);
      delete nodeStates[path];
    }
  });
}

function updateObjectNode(li, key, value, currentPath) {
  let toggleSpan = li.querySelector('.keytoggle');
  let labelSpan = li.querySelector('.keylabel');
  let childUl = li.querySelector('ul');

  // Initialize if new node
  if (!toggleSpan) {
    toggleSpan = document.createElement('span');
    toggleSpan.className = 'keytoggle';
    li.prepend(toggleSpan);
  }

  if (!labelSpan) {
    labelSpan = document.createElement('span');
    labelSpan.className = 'keylabel';
    toggleSpan.after(labelSpan);
  }

  if (!childUl) {
    childUl = document.createElement('ul');
    li.append(childUl);
  }

  // Update state
  const isExpanded = nodeStates[currentPath] ?? (defaultState === "expanded");
  toggleSpan.textContent = isExpanded ? "[-] " : "[+] ";
  labelSpan.textContent = `${key}: `;
  childUl.classList.toggle('is-hidden', !isExpanded);

  // Update toggle handler
  toggleSpan.onclick = (e) => {
    const wasExpanded = childUl.classList.toggle('is-hidden');
    nodeStates[currentPath] = !wasExpanded;
    toggleSpan.textContent = wasExpanded ? "[+] " : "[-] ";
    e.stopPropagation();
  };

  // Same handler for label
  labelSpan.onclick = (e) => {
    const wasExpanded = childUl.classList.toggle('is-hidden');
    nodeStates[currentPath] = !wasExpanded;
    toggleSpan.textContent = wasExpanded ? "[+] " : "[-] ";
    e.stopPropagation();
  };

  // Recurse with empty object protection
  reconcileTree(childUl, value || {}, currentPath);
}

function updateLeafNode(li, key, value, currentPath) {
  // Remove any nested elements
  li.querySelectorAll('.keytoggle, .keylabel, ul').forEach(el => el.remove());
  li.textContent = `${key}: ${JSON.stringify(value)}`;
}

function updateTreeView(data) {
  const treeContainer = document.getElementById("treeView");
  
  if (!treeContainer.firstElementChild || 
      treeContainer.firstElementChild.tagName !== 'UL') {
    treeContainer.innerHTML = '<ul></ul>';
  }
  
  reconcileTree(treeContainer.querySelector('ul'), data);
}


// -------------------- Chat Functionality --------------------

async function sendMessage(message) {
  try {
    await fetch('/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message })
    });
  } catch (err) {
    console.error("Message error:", err);
  }
}

function sendChat() {
  const chatInput = document.getElementById("chatMessage");
  const chatLog = document.getElementById("chatLog");
  const message = chatInput.value.trim();
  if (message) {
    // Append the user message to the chat log
    const userMessage = document.createElement("div");
    userMessage.classList.add("chat-message", "user-message");
    userMessage.textContent = message;
    chatLog.appendChild(userMessage);
    chatInput.value = "";

    // Scroll to the bottom of the chat log
    chatLog.scrollTop = chatLog.scrollHeight;
    
    // Send the message to the server
    sendMessage(message);
  }
}

chatData = [];
function fetchChat() {
  fetch('/chat')
    .then(response => response.json())
    .then(data => {
      // Check if chat data has changed
      if (JSON.stringify(data) === JSON.stringify(chatData)) {
        return;
      }

      // Update chat data
      chatData = data;

      // Update the chat log html
      const chatLog = document.getElementById("chatLog");
      // Clear the chat log
      chatLog.innerHTML = "";
      // Append messages to the chat log
      data.forEach(message => {
        if (message.role === "user") {
          const userMessage = document.createElement("div");
          userMessage.classList.add("chat-message", "user-message");
          userMessage.textContent = message.content;
          chatLog.appendChild(userMessage);
        } else if (message.role === "ai") {
          const aiMessage = document.createElement("div");
          aiMessage.classList.add("chat-message", "ai-message");
          aiMessage.textContent = message.content;
          chatLog.appendChild(aiMessage);
        }

        // Scroll to the bottom of the chat log
        chatLog.scrollTop = chatLog.scrollHeight;
      });
    })
    .catch(error => console.error("Error fetching chat:", error));
}

// -------------------- Server Communication & Control Functions --------------------
async function fetchState() {
  // Get current timestamp
  const start_time = Date.now();
  try {
    // Fetch current chat
    fetchChat();

    // Fetch current state
    const response = await fetch('/state');
    simData = await response.json();
    arena = simData.arena;
    running = simData.running;
    current_step = simData.current_step;
    groups = simData.groups;
    delete simData.arena;
    delete simData.running;
    delete simData.current_step;
    delete simData.groups;
    

    // Update step counter
    document.getElementById('stepCounter').textContent = `Step: ${current_step}`;

    // If simulation is running
    if (running || n_updates === 0) {
      if (n_updates !== 0) {
        // Modify the pause button
        const pauseButton = document.getElementById("pause-button");
        pauseButton.querySelector(".text").textContent = "Pause";
        pauseButton.querySelector(".icon").innerHTML = '<i class="fas fa-pause"></i>';
      }
      
      // Update canvas, display and tree view
      setCanvasSize();
      updateDisplay();
      updateTreeView(groups);
      // Increase number of updates
      n_updates += 1;
    } else {
      // Modify the pause button
      const pauseButton = document.getElementById("pause-button");
      pauseButton.querySelector(".text").textContent = "Resume";
      pauseButton.querySelector(".icon").innerHTML = '<i class="fas fa-play"></i>';
    }
  } catch (error) {
    console.error("Error fetching state:", error);
  }
  // Request a new animation frame after waiting for at least TIME_STEP milliseconds
  const end_time = Date.now();
  await sleep(Math.max(0, TIME_STEP - (end_time - start_time)));
  requestAnimationFrame(fetchState);
}

async function sendCommand(command) {
  try {
    await fetch('/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command })
    });
  } catch (err) {
    console.error("Command error:", err);
  }
}

// Event listener functions
function togglePause() { sendCommand('pause'); }
function toggleReset() { 
  // Set number of updates to 0
  n_updates = 0;
  // Reset simulation state
  sendCommand('reset'); 
}
function toggleStop() { sendCommand('stop'); }


// -------------------- Initialization --------------------

// Add event listener to send button
document.getElementById("sendButton").onclick = sendChat;

// Add event listener to chat input field
document.getElementById("chatMessage").addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    sendChat();
  }
});

// Add event listeners to control buttons
document.getElementById("pause-button").onclick = togglePause;
document.getElementById("reset-button").onclick = toggleReset;
document.getElementById("stop-button").onclick = toggleStop;

// Add canvas sizing event listener
window.addEventListener('resize', setCanvasSize);

// Fetch initial state
fetchState();
