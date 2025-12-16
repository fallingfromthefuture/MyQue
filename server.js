const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const cors = require('cors');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const { v4: uuidv4 } = require('uuid');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Middleware
app.use(cors());
app.use(express.json());

// JWT Secret
const JWT_SECRET = process.env.JWT_SECRET || 'roboos-secret-key-change-in-production';

// In-memory data stores (replace with proper database in production)
const users = [
  {
    id: '1',
    email: 'demo@theroboos.com',
    password: bcrypt.hashSync('demo123', 10),
    name: 'Demo User',
    role: 'admin'
  }
];

const robots = [];
const tasks = [];
const memories = [];
const skills = [
  { id: 'skill_1', name: 'Object Detection', category: 'Vision', status: 'active' },
  { id: 'skill_2', name: 'Path Planning', category: 'Navigation', status: 'active' },
  { id: 'skill_3', name: 'Gripper Control', category: 'Manipulation', status: 'active' },
  { id: 'skill_4', name: 'Speech Recognition', category: 'Communication', status: 'active' }
];

// WebSocket connections store
const connections = new Map();

// Authentication middleware
const authenticate = (req, res, next) => {
  const token = req.headers.authorization?.split(' ')[1];
  if (!token) {
    return res.status(401).json({ error: 'No token provided' });
  }
  
  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    req.user = decoded;
    next();
  } catch (error) {
    return res.status(401).json({ error: 'Invalid token' });
  }
};

// API Routes

// Auth routes
app.post('/api/auth/login', (req, res) => {
  const { email, password } = req.body;
  const user = users.find(u => u.email === email);
  
  if (!user || !bcrypt.compareSync(password, user.password)) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }
  
  const token = jwt.sign(
    { id: user.id, email: user.email, role: user.role },
    JWT_SECRET,
    { expiresIn: '24h' }
  );
  
  res.json({
    token,
    user: {
      id: user.id,
      email: user.email,
      name: user.name,
      role: user.role
    }
  });
});

app.get('/api/auth/me', authenticate, (req, res) => {
  const user = users.find(u => u.id === req.user.id);
  if (!user) {
    return res.status(404).json({ error: 'User not found' });
  }
  res.json({
    id: user.id,
    email: user.email,
    name: user.name,
    role: user.role
  });
});

// Robot routes
app.get('/api/robots', authenticate, (req, res) => {
  res.json(robots);
});

app.post('/api/robots', authenticate, (req, res) => {
  const robot = {
    id: uuidv4(),
    ...req.body,
    status: 'offline',
    lastSeen: new Date().toISOString(),
    createdAt: new Date().toISOString()
  };
  robots.push(robot);
  broadcast({ type: 'robot_added', data: robot });
  res.json(robot);
});

app.put('/api/robots/:id', authenticate, (req, res) => {
  const index = robots.findIndex(r => r.id === req.params.id);
  if (index === -1) {
    return res.status(404).json({ error: 'Robot not found' });
  }
  robots[index] = { ...robots[index], ...req.body };
  broadcast({ type: 'robot_updated', data: robots[index] });
  res.json(robots[index]);
});

app.delete('/api/robots/:id', authenticate, (req, res) => {
  const index = robots.findIndex(r => r.id === req.params.id);
  if (index === -1) {
    return res.status(404).json({ error: 'Robot not found' });
  }
  const deleted = robots.splice(index, 1)[0];
  broadcast({ type: 'robot_deleted', data: { id: deleted.id } });
  res.json({ success: true });
});

// Task routes
app.get('/api/tasks', authenticate, (req, res) => {
  res.json(tasks);
});

app.post('/api/tasks', authenticate, (req, res) => {
  const task = {
    id: uuidv4(),
    ...req.body,
    status: 'pending',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  };
  tasks.push(task);
  broadcast({ type: 'task_created', data: task });
  res.json(task);
});

app.put('/api/tasks/:id', authenticate, (req, res) => {
  const index = tasks.findIndex(t => t.id === req.params.id);
  if (index === -1) {
    return res.status(404).json({ error: 'Task not found' });
  }
  tasks[index] = { ...tasks[index], ...req.body, updatedAt: new Date().toISOString() };
  broadcast({ type: 'task_updated', data: tasks[index] });
  res.json(tasks[index]);
});

// Skill library routes
app.get('/api/skills', authenticate, (req, res) => {
  res.json(skills);
});

// Memory routes
app.get('/api/memory', authenticate, (req, res) => {
  res.json(memories);
});

app.post('/api/memory', authenticate, (req, res) => {
  const memory = {
    id: uuidv4(),
    ...req.body,
    timestamp: new Date().toISOString()
  };
  memories.push(memory);
  // Keep only last 1000 memories
  if (memories.length > 1000) {
    memories.shift();
  }
  broadcast({ type: 'memory_updated', data: memory });
  res.json(memory);
});

// System status route
app.get('/api/status', authenticate, (req, res) => {
  res.json({
    robots: {
      total: robots.length,
      online: robots.filter(r => r.status === 'online').length,
      busy: robots.filter(r => r.status === 'busy').length,
      offline: robots.filter(r => r.status === 'offline').length
    },
    tasks: {
      total: tasks.length,
      pending: tasks.filter(t => t.status === 'pending').length,
      running: tasks.filter(t => t.status === 'running').length,
      completed: tasks.filter(t => t.status === 'completed').length,
      failed: tasks.filter(t => t.status === 'failed').length
    },
    skills: skills.length,
    memorySize: memories.length,
    uptime: process.uptime()
  });
});

// Coordination endpoint - brain decision making
app.post('/api/coordinate', authenticate, (req, res) => {
  const { task, availableRobots } = req.body;
  
  // Simple task allocation algorithm
  const robotsToUse = availableRobots
    .filter(r => r.status === 'online')
    .slice(0, task.robotsNeeded || 1);
  
  const plan = {
    id: uuidv4(),
    task: task.id,
    robots: robotsToUse.map(r => r.id),
    steps: generateTaskSteps(task, robotsToUse),
    estimatedDuration: calculateDuration(task, robotsToUse),
    createdAt: new Date().toISOString()
  };
  
  res.json(plan);
});

// Helper functions
function generateTaskSteps(task, robots) {
  // Simple step generation based on task type
  const steps = [];
  if (task.type === 'delivery') {
    steps.push({ action: 'navigate', target: task.pickup, robot: robots[0]?.id });
    steps.push({ action: 'grasp', target: task.object, robot: robots[0]?.id });
    steps.push({ action: 'navigate', target: task.dropoff, robot: robots[0]?.id });
    steps.push({ action: 'release', target: task.object, robot: robots[0]?.id });
  }
  return steps;
}

function calculateDuration(task, robots) {
  // Simple duration estimation
  return Math.random() * 300 + 60; // 1-6 minutes
}

function broadcast(message) {
  connections.forEach((ws) => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    }
  });
}

// WebSocket handling
wss.on('connection', (ws, req) => {
  const connectionId = uuidv4();
  connections.set(connectionId, ws);
  
  console.log(`WebSocket client connected: ${connectionId}`);
  
  ws.on('message', (message) => {
    try {
      const data = JSON.parse(message);
      
      // Handle different message types
      switch (data.type) {
        case 'robot_status':
          const robot = robots.find(r => r.id === data.robotId);
          if (robot) {
            robot.status = data.status;
            robot.lastSeen = new Date().toISOString();
            broadcast({ type: 'robot_updated', data: robot });
          }
          break;
        
        case 'task_progress':
          const task = tasks.find(t => t.id === data.taskId);
          if (task) {
            task.progress = data.progress;
            task.updatedAt = new Date().toISOString();
            broadcast({ type: 'task_updated', data: task });
          }
          break;
        
        case 'ping':
          ws.send(JSON.stringify({ type: 'pong', timestamp: Date.now() }));
          break;
      }
    } catch (error) {
      console.error('WebSocket message error:', error);
    }
  });
  
  ws.on('close', () => {
    connections.delete(connectionId);
    console.log(`WebSocket client disconnected: ${connectionId}`);
  });
  
  // Send welcome message
  ws.send(JSON.stringify({
    type: 'connected',
    connectionId,
    timestamp: Date.now()
  }));
});

// Error handling
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Internal server error' });
});

// Start server
const PORT = process.env.PORT || 3001;
server.listen(PORT, () => {
  console.log(`RoboOS Backend Server running on port ${PORT}`);
  console.log(`WebSocket Server ready`);
  
  // Create some demo robots and tasks
  initializeDemoData();
});

function initializeDemoData() {
  // Add demo robots
  const demoRobots = [
    {
      id: uuidv4(),
      name: 'Unitree G1',
      type: 'humanoid',
      embodiment: 'dual-arm',
      status: 'online',
      capabilities: ['navigation', 'manipulation', 'vision'],
      location: { x: 10, y: 20, z: 0 },
      battery: 85,
      lastSeen: new Date().toISOString(),
      createdAt: new Date().toISOString()
    },
    {
      id: uuidv4(),
      name: 'AgileX Mobile',
      type: 'mobile',
      embodiment: 'wheeled',
      status: 'online',
      capabilities: ['navigation', 'delivery', 'sensing'],
      location: { x: 15, y: 25, z: 0 },
      battery: 92,
      lastSeen: new Date().toISOString(),
      createdAt: new Date().toISOString()
    },
    {
      id: uuidv4(),
      name: 'RealMan Arm',
      type: 'manipulator',
      embodiment: 'single-arm',
      status: 'idle',
      capabilities: ['manipulation', 'assembly', 'precision'],
      location: { x: 5, y: 10, z: 0 },
      battery: 100,
      lastSeen: new Date().toISOString(),
      createdAt: new Date().toISOString()
    }
  ];
  
  robots.push(...demoRobots);
  
  // Add demo tasks
  const demoTasks = [
    {
      id: uuidv4(),
      title: 'Burger Preparation',
      description: 'Prepare burger with multiple robots',
      type: 'assembly',
      status: 'completed',
      assignedRobots: [demoRobots[0].id, demoRobots[1].id],
      progress: 100,
      priority: 'high',
      createdAt: new Date(Date.now() - 3600000).toISOString(),
      updatedAt: new Date().toISOString()
    },
    {
      id: uuidv4(),
      title: 'Object Delivery',
      description: 'Deliver apple and knife to kitchen',
      type: 'delivery',
      status: 'running',
      assignedRobots: [demoRobots[1].id],
      progress: 45,
      priority: 'medium',
      createdAt: new Date(Date.now() - 1800000).toISOString(),
      updatedAt: new Date().toISOString()
    }
  ];
  
  tasks.push(...demoTasks);
  
  console.log('Demo data initialized');
}
