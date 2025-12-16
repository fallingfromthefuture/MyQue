// RoboOS Backend Server - Minimal Version
// This is a lightweight, production-ready backend for robot coordination
const express = require('express');
const cors = require('cors');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const { v4: uuidv4 } = require('uuid');
const WebSocket = require('ws');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3001;
const JWT_SECRET = process.env.JWT_SECRET || 'change-this-in-production';

// Middleware
app.use(cors());
app.use(express.json());

// In-memory storage (replace with database in production)
const users = [{
    id: '1',
    email: 'demo@theroboos.com',
    password: bcrypt.hashSync('demo123', 10),
    name: 'Demo User'
}];

const robots = [
    { id: '1', name: 'Unitree G1', type: 'Humanoid', status: 'online', battery: 85, position: { x: 0, y: 0 } },
    { id: '2', name: 'AgileX Scout', type: 'Mobile', status: 'busy', battery: 92, position: { x: 5, y: 3 } },
    { id: '3', name: 'UR5e Arm', type: 'Manipulator', status: 'offline', battery: 100, position: { x: 2, y: 1 } }
];

const tasks = [
    { id: '1', name: 'Patrol Area A', robot: '1', status: 'active', progress: 65, priority: 'high' },
    { id: '2', name: 'Pick and Place', robot: '2', status: 'completed', progress: 100, priority: 'medium' }
];

// Auth middleware
const authenticate = (req, res, next) => {
    const token = req.headers.authorization?.split(' ')[1];
    if (!token) return res.status(401).json({ error: 'No token provided' });
    
    try {
        const decoded = jwt.verify(token, JWT_SECRET);
        req.userId = decoded.userId;
        next();
    } catch (err) {
        res.status(401).json({ error: 'Invalid token' });
    }
};

// Routes
app.post('/api/auth/login', async (req, res) => {
    const { email, password } = req.body;
    const user = users.find(u => u.email === email);
    
    if (!user || !bcrypt.compareSync(password, user.password)) {
        return res.status(401).json({ error: 'Invalid credentials' });
    }
    
    const token = jwt.sign({ userId: user.id }, JWT_SECRET, { expiresIn: '24h' });
    res.json({ token, user: { id: user.id, email: user.email, name: user.name } });
});

app.get('/api/auth/me', authenticate, (req, res) => {
    const user = users.find(u => u.id === req.userId);
    res.json({ id: user.id, email: user.email, name: user.name });
});

app.get('/api/robots', authenticate, (req, res) => {
    res.json(robots);
});

app.post('/api/robots', authenticate, (req, res) => {
    const robot = { id: uuidv4(), ...req.body, status: 'offline', battery: 100 };
    robots.push(robot);
    broadcastToClients({ type: 'robot_added', data: robot });
    res.status(201).json(robot);
});

app.put('/api/robots/:id', authenticate, (req, res) => {
    const index = robots.findIndex(r => r.id === req.params.id);
    if (index === -1) return res.status(404).json({ error: 'Robot not found' });
    
    robots[index] = { ...robots[index], ...req.body };
    broadcastToClients({ type: 'robot_updated', data: robots[index] });
    res.json(robots[index]);
});

app.delete('/api/robots/:id', authenticate, (req, res) => {
    const index = robots.findIndex(r => r.id === req.params.id);
    if (index === -1) return res.status(404).json({ error: 'Robot not found' });
    
    robots.splice(index, 1);
    res.json({ success: true });
});

app.get('/api/tasks', authenticate, (req, res) => {
    res.json(tasks);
});

app.post('/api/tasks', authenticate, (req, res) => {
    const task = { id: uuidv4(), ...req.body, status: 'pending', progress: 0 };
    tasks.push(task);
    broadcastToClients({ type: 'task_updated', data: task });
    res.status(201).json(task);
});

app.put('/api/tasks/:id', authenticate, (req, res) => {
    const index = tasks.findIndex(t => t.id === req.params.id);
    if (index === -1) return res.status(404).json({ error: 'Task not found' });
    
    tasks[index] = { ...tasks[index], ...req.body };
    broadcastToClients({ type: 'task_updated', data: tasks[index] });
    res.json(tasks[index]);
});

app.get('/api/status', authenticate, (req, res) => {
    res.json({
        totalRobots: robots.length,
        activeRobots: robots.filter(r => r.status === 'online').length,
        totalTasks: tasks.length,
        activeTasks: tasks.filter(t => t.status === 'active').length,
        systemHealth: 'good'
    });
});

app.post('/api/coordinate', authenticate, (req, res) => {
    // Simple task coordination
    const availableRobots = robots.filter(r => r.status === 'online');
    const pendingTasks = tasks.filter(t => t.status === 'pending');
    
    const assignments = pendingTasks.slice(0, availableRobots.length).map((task, i) => ({
        taskId: task.id,
        robotId: availableRobots[i].id,
        estimatedTime: Math.random() * 60 + 10
    }));
    
    res.json({ assignments });
});

// Health check
app.get('/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Start HTTP server
const server = app.listen(PORT, () => {
    console.log(`✓ RoboOS Backend running on port ${PORT}`);
    console.log(`✓ Health check: http://localhost:${PORT}/health`);
});

// WebSocket server
const wss = new WebSocket.Server({ server });
const clients = new Set();

wss.on('connection', (ws) => {
    clients.add(ws);
    console.log('WebSocket client connected');
    
    ws.send(JSON.stringify({ 
        type: 'connected', 
        message: 'Connected to RoboOS',
        timestamp: new Date().toISOString()
    }));
    
    // Handle incoming messages
    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message);
            console.log('Received:', data.type);
            
            if (data.type === 'ping') {
                ws.send(JSON.stringify({ type: 'pong', timestamp: new Date().toISOString() }));
            }
        } catch (err) {
            console.error('WebSocket message error:', err);
        }
    });
    
    ws.on('close', () => {
        clients.delete(ws);
        console.log('WebSocket client disconnected');
    });
});

// Broadcast function
function broadcastToClients(data) {
    const message = JSON.stringify(data);
    clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(message);
        }
    });
}

// Periodic updates (simulating robot status changes)
setInterval(() => {
    robots.forEach(robot => {
        if (robot.status === 'online' && Math.random() > 0.7) {
            robot.battery = Math.max(0, robot.battery - Math.floor(Math.random() * 3));
            broadcastToClients({ type: 'robot_updated', data: robot });
        }
    });
}, 10000);

console.log('✓ WebSocket server running');
console.log('✓ Ready to accept connections');

module.exports = app;
