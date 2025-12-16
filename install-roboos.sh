#!/bin/bash
# RoboOS Integrated System - Quick Installer
# This script downloads and sets up the complete RoboOS system
# 
# Usage: curl -sSL https://raw.githubusercontent.com/your-repo/install.sh | bash
# Or: bash install.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="${INSTALL_DIR:-roboos-system}"
GITHUB_REPO="theroboos/roboos-integrated"  # Change to your actual repo
BRANCH="main"

# Print functions
print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

# Check dependencies
check_dependencies() {
    print_header "Checking Dependencies"
    
    # Check Node.js
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_success "Node.js installed: $NODE_VERSION"
    else
        print_error "Node.js not found!"
        print_info "Install from: https://nodejs.org/"
        exit 1
    fi
    
    # Check npm
    if command -v npm &> /dev/null; then
        NPM_VERSION=$(npm --version)
        print_success "npm installed: $NPM_VERSION"
    else
        print_error "npm not found!"
        exit 1
    fi
    
    echo ""
}

# Create directory structure
create_structure() {
    print_header "Creating Directory Structure"
    
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    
    mkdir -p backend
    mkdir -p frontend
    mkdir -p docs
    
    print_success "Directories created"
    echo ""
}

# Download or create files
create_backend() {
    print_header "Setting Up Backend"
    
    cd backend
    
    # Create package.json
    cat > package.json << 'EOF'
{
  "name": "roboos-backend",
  "version": "1.0.0",
  "description": "RoboOS Backend Server",
  "main": "server.js",
  "scripts": {
    "start": "node server.js",
    "dev": "nodemon server.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "cors": "^2.8.5",
    "ws": "^8.14.2",
    "jsonwebtoken": "^9.0.2",
    "bcryptjs": "^2.4.3",
    "uuid": "^9.0.1",
    "dotenv": "^16.3.1"
  },
  "devDependencies": {
    "nodemon": "^3.0.1"
  }
}
EOF
    print_success "package.json created"
    
    # Create minimal server.js
    cat > server.js << 'EOF'
const express = require('express');
const cors = require('cors');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const { v4: uuidv4 } = require('uuid');
const WebSocket = require('ws');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3001;
const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key-change-in-production';

// Middleware
app.use(cors());
app.use(express.json());

// In-memory storage
const users = [
    {
        id: '1',
        email: 'demo@theroboos.com',
        password: bcrypt.hashSync('demo123', 10),
        name: 'Demo User'
    }
];

const robots = [
    { id: '1', name: 'Unitree G1', type: 'Humanoid', status: 'online', battery: 85 },
    { id: '2', name: 'AgileX Scout', type: 'Mobile', status: 'busy', battery: 92 },
    { id: '3', name: 'UR5e Arm', type: 'Manipulator', status: 'offline', battery: 100 }
];

const tasks = [
    { id: '1', name: 'Patrol Area A', robot: '1', status: 'active', progress: 65 },
    { id: '2', name: 'Pick and Place', robot: '2', status: 'completed', progress: 100 }
];

// Auth middleware
const authenticate = (req, res, next) => {
    const token = req.headers.authorization?.split(' ')[1];
    if (!token) return res.status(401).json({ error: 'No token' });
    
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

app.get('/api/robots', authenticate, (req, res) => {
    res.json(robots);
});

app.get('/api/tasks', authenticate, (req, res) => {
    res.json(tasks);
});

app.get('/api/status', authenticate, (req, res) => {
    res.json({
        totalRobots: robots.length,
        activeRobots: robots.filter(r => r.status === 'online').length,
        totalTasks: tasks.length,
        activeTasks: tasks.filter(t => t.status === 'active').length
    });
});

// Start server
const server = app.listen(PORT, () => {
    console.log(`✓ Backend server running on port ${PORT}`);
});

// WebSocket
const wss = new WebSocket.Server({ server });

wss.on('connection', (ws) => {
    console.log('WebSocket client connected');
    ws.send(JSON.stringify({ type: 'connected', message: 'Connected to RoboOS' }));
    
    // Send periodic updates
    const interval = setInterval(() => {
        ws.send(JSON.stringify({
            type: 'robot_updated',
            data: robots[Math.floor(Math.random() * robots.length)]
        }));
    }, 5000);
    
    ws.on('close', () => {
        clearInterval(interval);
        console.log('WebSocket client disconnected');
    });
});

console.log('✓ WebSocket server running');
EOF
    print_success "server.js created"
    
    # Create .env.example
    cat > .env.example << 'EOF'
PORT=3001
JWT_SECRET=change-this-secret-in-production
NODE_ENV=development
EOF
    print_success ".env.example created"
    
    # Install dependencies
    print_info "Installing backend dependencies..."
    npm install --silent
    print_success "Backend dependencies installed"
    
    cd ..
    echo ""
}

create_frontend() {
    print_header "Setting Up Frontend"
    
    cd frontend
    
    # Create minimal index.html
    cat > index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RoboOS Dashboard</title>
    <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f172a; color: #fff; }
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
        .header { text-align: center; margin-bottom: 2rem; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; }
        .card { background: #1e293b; border-radius: 12px; padding: 1.5rem; border: 1px solid #334155; }
        .card h3 { margin-bottom: 1rem; color: #60a5fa; }
        .status { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.875rem; }
        .status.online { background: #10b981; color: white; }
        .status.busy { background: #f59e0b; color: white; }
        .status.offline { background: #6b7280; color: white; }
        button { background: #3b82f6; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 8px; cursor: pointer; font-size: 1rem; }
        button:hover { background: #2563eb; }
        .login-form { max-width: 400px; margin: 4rem auto; }
        input { width: 100%; padding: 0.75rem; border: 1px solid #334155; border-radius: 8px; background: #1e293b; color: white; margin-bottom: 1rem; }
    </style>
</head>
<body>
    <div id="root"></div>
    
    <script type="text/babel">
        const { useState, useEffect } = React;
        
        const API_URL = 'http://localhost:3001/api';
        
        function App() {
            const [token, setToken] = useState(localStorage.getItem('token'));
            const [robots, setRobots] = useState([]);
            const [tasks, setTasks] = useState([]);
            
            if (!token) {
                return <Login onLogin={setToken} />;
            }
            
            useEffect(() => {
                fetchData();
                const interval = setInterval(fetchData, 5000);
                return () => clearInterval(interval);
            }, []);
            
            const fetchData = async () => {
                try {
                    const headers = { 'Authorization': `Bearer ${token}` };
                    const [robotsRes, tasksRes] = await Promise.all([
                        fetch(`${API_URL}/robots`, { headers }),
                        fetch(`${API_URL}/tasks`, { headers })
                    ]);
                    setRobots(await robotsRes.json());
                    setTasks(await tasksRes.json());
                } catch (err) {
                    console.error('Fetch error:', err);
                }
            };
            
            return (
                <div className="container">
                    <div className="header">
                        <h1>🤖 RoboOS Dashboard</h1>
                        <p>Robot Coordination Platform</p>
                    </div>
                    
                    <div className="grid">
                        <div className="card">
                            <h3>📊 System Status</h3>
                            <p>Total Robots: {robots.length}</p>
                            <p>Active Tasks: {tasks.filter(t => t.status === 'active').length}</p>
                        </div>
                        
                        {robots.map(robot => (
                            <div key={robot.id} className="card">
                                <h3>{robot.name}</h3>
                                <p>Type: {robot.type}</p>
                                <p>Battery: {robot.battery}%</p>
                                <span className={`status ${robot.status}`}>{robot.status}</span>
                            </div>
                        ))}
                    </div>
                    
                    <button onClick={() => { localStorage.removeItem('token'); setToken(null); }}>
                        Logout
                    </button>
                </div>
            );
        }
        
        function Login({ onLogin }) {
            const [email, setEmail] = useState('demo@theroboos.com');
            const [password, setPassword] = useState('demo123');
            
            const handleSubmit = async (e) => {
                e.preventDefault();
                try {
                    const res = await fetch(`${API_URL}/auth/login`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ email, password })
                    });
                    const data = await res.json();
                    if (data.token) {
                        localStorage.setItem('token', data.token);
                        onLogin(data.token);
                    }
                } catch (err) {
                    alert('Login failed');
                }
            };
            
            return (
                <div className="login-form">
                    <div className="card">
                        <h2>Login to RoboOS</h2>
                        <form onSubmit={handleSubmit}>
                            <input 
                                type="email" 
                                placeholder="Email"
                                value={email}
                                onChange={e => setEmail(e.target.value)}
                            />
                            <input 
                                type="password" 
                                placeholder="Password"
                                value={password}
                                onChange={e => setPassword(e.target.value)}
                            />
                            <button type="submit">Login</button>
                        </form>
                        <p style={{marginTop: '1rem', fontSize: '0.875rem', color: '#94a3b8'}}>
                            Demo: demo@theroboos.com / demo123
                        </p>
                    </div>
                </div>
            );
        }
        
        ReactDOM.render(<App />, document.getElementById('root'));
    </script>
</body>
</html>
EOF
    print_success "index.html created"
    
    cd ..
    echo ""
}

create_docs() {
    print_header "Creating Documentation"
    
    cd docs
    
    cat > README.md << 'EOF'
# RoboOS Integrated System

Complete robot coordination platform with backend API and frontend dashboard.

## Quick Start

1. Start backend:
   ```bash
   cd backend
   npm start
   ```

2. Open frontend:
   ```bash
   cd frontend
   open index.html
   # Or use a local server:
   python3 -m http.server 8000
   ```

3. Login:
   - Email: demo@theroboos.com
   - Password: demo123

## Features

- JWT Authentication
- Real-time WebSocket updates
- Robot management
- Task coordination
- REST API
- React dashboard

## API Endpoints

- POST /api/auth/login - Login
- GET /api/robots - List robots
- GET /api/tasks - List tasks
- GET /api/status - System status

## Tech Stack

**Backend:**
- Node.js + Express
- WebSocket
- JWT Authentication

**Frontend:**
- React (CDN)
- Modern UI

## Development

Backend runs on http://localhost:3001
Frontend can run on any static server

## Production

1. Set JWT_SECRET in .env
2. Configure CORS for your domain
3. Use a process manager (PM2)
4. Set up HTTPS

## License

MIT
EOF
    print_success "README.md created"
    
    cd ..
    echo ""
}

create_startup_scripts() {
    print_header "Creating Startup Scripts"
    
    # Start script for Unix
    cat > start.sh << 'EOF'
#!/bin/bash
echo "Starting RoboOS System..."

# Start backend
cd backend
npm start &
BACKEND_PID=$!
echo "Backend started (PID: $BACKEND_PID)"

# Wait for backend
sleep 3

# Open frontend
echo "Opening frontend..."
if command -v python3 &> /dev/null; then
    cd ../frontend
    python3 -m http.server 8000 &
    FRONTEND_PID=$!
    echo "Frontend server started on http://localhost:8000"
    
    # Open browser
    sleep 2
    if command -v open &> /dev/null; then
        open http://localhost:8000
    elif command -v xdg-open &> /dev/null; then
        xdg-open http://localhost:8000
    fi
else
    echo "Open frontend/index.html in your browser"
fi

echo ""
echo "RoboOS is running!"
echo "Backend: http://localhost:3001"
echo "Frontend: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop"

wait
EOF
    chmod +x start.sh
    print_success "start.sh created"
    
    # Start script for Windows
    cat > start.bat << 'EOF'
@echo off
echo Starting RoboOS System...

cd backend
start "RoboOS Backend" cmd /k npm start

timeout /t 3

cd ..\frontend
start "RoboOS Frontend" cmd /k python -m http.server 8000

timeout /t 2
start http://localhost:8000

echo.
echo RoboOS is running!
echo Backend: http://localhost:3001
echo Frontend: http://localhost:8000
EOF
    print_success "start.bat created"
    
    echo ""
}

# Main installation flow
main() {
    clear
    print_header "RoboOS Integrated System Installer"
    echo ""
    
    check_dependencies
    create_structure
    create_backend
    create_frontend
    create_docs
    create_startup_scripts
    
    print_header "Installation Complete!"
    echo ""
    print_success "RoboOS system installed in: $INSTALL_DIR"
    echo ""
    echo "To start the system:"
    echo "  cd $INSTALL_DIR"
    echo "  ./start.sh        (Linux/Mac)"
    echo "  start.bat         (Windows)"
    echo ""
    echo "Or start manually:"
    echo "  1. cd $INSTALL_DIR/backend && npm start"
    echo "  2. Open $INSTALL_DIR/frontend/index.html"
    echo ""
    print_info "Default login: demo@theroboos.com / demo123"
    echo ""
}

# Run installation
main
