const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const WebSocket = require('ws');

class SAMBridge {
  constructor() {
    this.serverProcess = null;
    this.isServerRunning = false;
    this.serverStartPromise = null;
    this.baseUrl = 'http://127.0.0.1:8888';  // Changed to IPv4
    this.sessionId = 'electron-session';
    this.toolCallListeners = new Set();
    this.wsConnection = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
  }

  // Helper function to make HTTP requests
  async _makeHttpRequest(url, options = {}) {
    const http = require('http');
    const https = require('https');
    const urlModule = require('url');

    const parsedUrl = new URL(url);
    const client = parsedUrl.protocol === 'https:' ? https : http;

    return new Promise((resolve, reject) => {
      const requestOptions = {
        hostname: parsedUrl.hostname,
        port: parsedUrl.port,
        path: parsedUrl.pathname + parsedUrl.search,
        method: options.method || 'GET',
        headers: options.headers || {},
        timeout: options.timeout || 10000
      };

      const req = client.request(requestOptions, (res) => {
        let data = '';
        res.on('data', chunk => data += chunk);
        res.on('end', () => {
          try {
            const response = {
              status: res.statusCode,
              statusText: res.statusMessage,
              ok: res.statusCode >= 200 && res.statusCode < 300,
              json: () => Promise.resolve(JSON.parse(data)),
              text: () => Promise.resolve(data)
            };
            resolve(response);
          } catch (e) {
            resolve({
              status: res.statusCode,
              statusText: res.statusMessage,
              ok: res.statusCode >= 200 && res.statusCode < 300,
              json: () => Promise.reject(new Error('Invalid JSON')),
              text: () => Promise.resolve(data)
            });
          }
        });
      });

      req.on('error', reject);
      req.on('timeout', () => {
        req.destroy();
        reject(new Error('Request timeout'));
      });

      // Send request body if provided
      if (options.body) {
        req.write(options.body);
      }

      req.end();
    });
  }

  // ===== SERVER MANAGEMENT =====
  async ensureServerRunning() {
    if (this.isServerRunning && this.serverProcess && !this.serverProcess.killed) {
      return true;
    }

    if (this.serverStartPromise) {
      return this.serverStartPromise;
    }

    this.serverStartPromise = this.startServer();
    const result = await this.serverStartPromise;
    this.serverStartPromise = null;
    return result;
  }

  async startServer() {
    try {
      console.log('🚀 Starting SAM agent server...');

      const samPath = this.findSAMPath();
      console.log(`Found SAM agent at: ${samPath}`);

      // Start the Python server process with UTF-8 environment (same as Prism)
      const env = {
        ...process.env,
        PYTHONIOENCODING: 'utf-8',
        PYTHONLEGACYWINDOWSSTDIO: '1',
        PYTHONUTF8: '1'
      };

      this.serverProcess = spawn('python', [samPath, '--api', '--api-port', '8888'], {
        cwd: path.dirname(samPath),
        stdio: ['pipe', 'pipe', 'pipe'],
        env: env
      });

      // Handle server output
      this.serverProcess.stdout.on('data', (data) => {
        const output = data.toString();
        console.log('[SAM Server]:', output);

        if (output.includes('SAM API server starting')) {
          this.isServerRunning = true;
          this.setupWebSocketConnection();
        }
      });

      this.serverProcess.stderr.on('data', (data) => {
        const error = data.toString();
        console.error('[SAM Server Error]:', error);
      });

      this.serverProcess.on('close', (code) => {
        console.log(`SAM server process exited with code ${code}`);
        this.isServerRunning = false;
        this.wsConnection = null;
      });

      this.serverProcess.on('error', (error) => {
        console.error('Failed to start SAM server:', error);
        this.isServerRunning = false;
        throw error;
      });

      // Wait for server to be ready
      await this.waitForServerReady();
      return true;

    } catch (error) {
      console.error('Error starting SAM server:', error);
      this.isServerRunning = false;
      throw error;
    }
  }

  async waitForServerReady(timeout = 60000) {
  const startTime = Date.now();
  let lastError = null;

  console.log(`⏰ Waiting for SAM server at ${this.baseUrl}/health...`);

  while (Date.now() - startTime < timeout) {
    try {
      console.log(`🔍 Health check attempt: ${this.baseUrl}/health`);

      const response = await this._makeHttpRequest(`${this.baseUrl}/health`, {
        method: 'GET',
        timeout: 5000
      });

      console.log(`📡 Health check response: ${response.status}`);

      if (response.ok) {
        const healthData = await response.json();
        console.log('✅ SAM server is ready:', healthData);
        return true;
      } else {
        lastError = `HTTP ${response.status}`;
        console.log(`⚠️ Health check failed: ${lastError}`);
      }
    } catch (error) {
      lastError = error.message;
      console.log(`⚠️ Health check error: ${lastError}`);
    }

    await new Promise(resolve => setTimeout(resolve, 2000));
  }

  throw new Error(`SAM server failed to start within timeout period. Last error: ${lastError}`);
}

  setupWebSocketConnection() {
    try {
      const wsUrl = `ws://localhost:8888/ws/tool-events`;
      this.wsConnection = new WebSocket(wsUrl);

      this.wsConnection.on('open', () => {
        console.log('🔌 WebSocket connected to SAM server');
        this.reconnectAttempts = 0;
      });

      this.wsConnection.on('message', (data) => {
        try {
          const event = JSON.parse(data.toString());
          this.notifyToolCallListeners(event);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      });

      this.wsConnection.on('close', () => {
        console.log('🔌 WebSocket disconnected from SAM server');
        this.wsConnection = null;
        this.attemptReconnect();
      });

      this.wsConnection.on('error', (error) => {
        console.error('WebSocket error:', error);
      });

    } catch (error) {
      console.error('Failed to setup WebSocket connection:', error);
    }
  }

  attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts && this.isServerRunning) {
      this.reconnectAttempts++;
      console.log(`Attempting WebSocket reconnection (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

      setTimeout(() => {
        this.setupWebSocketConnection();
      }, 2000 * this.reconnectAttempts);
    }
  }

  stopServer() {
    if (this.serverProcess && !this.serverProcess.killed) {
      console.log('🛑 Stopping SAM server...');
      this.serverProcess.kill('SIGTERM');
      this.isServerRunning = false;
    }

    if (this.wsConnection) {
      this.wsConnection.close();
      this.wsConnection = null;
    }
  }

  // ===== API METHODS =====
  async sendMessage(message) {
  await this.ensureServerRunning();

  try {
    const response = await this._makeHttpRequest(`${this.baseUrl}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: message,
        session_id: this.sessionId,
        auto_approve: true,
        max_iterations: 10,
        verbose: false
      }),
      timeout: 60000
    });

    if (!response.ok) {
      throw new Error(`Query failed: ${response.statusText}`);
    }

    const result = await response.json();
    return { success: true, response: result };

  } catch (error) {
    console.error('Error sending message:', error);

    // Handle connection errors by trying to restart server
    if (error.message.includes('ECONNREFUSED') || error.message.includes('timeout')) {
      console.log('Connection failed, attempting to restart server...');
      this.isServerRunning = false;
      this.serverStartPromise = null;

      try {
        await this.ensureServerRunning();
        return await this.sendMessage(message);
      } catch (retryError) {
        return { success: false, error: `Server restart failed: ${retryError.message}` };
      }
    }

    return { success: false, error: error.message };
  }
}

  async executeToolDirect(toolName, toolArgs) {
  await this.ensureServerRunning();

  try {
    const response = await this._makeHttpRequest(`${this.baseUrl}/execute-tool`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        tool_name: toolName,
        arguments: toolArgs,
        session_id: this.sessionId
      }),
      timeout: 30000
    });

    if (!response.ok) {
      throw new Error(`Tool execution failed: ${response.statusText}`);
    }

    const result = await response.json();
    return result;

  } catch (error) {
    console.error(`Error executing tool ${toolName}:`, error);
    return { success: false, error: error.message };
  }
}

 async getCapabilities() {
  await this.ensureServerRunning();

  try {
    const response = await this._makeHttpRequest(`${this.baseUrl}/tools`, {
      method: 'GET',
      timeout: 10000
    });

    if (!response.ok) {
      throw new Error(`Failed to get capabilities: ${response.statusText}`);
    }

    const tools = await response.json();
    return { success: true, tools };

  } catch (error) {
    console.error('Error getting capabilities:', error);
    return { success: false, error: error.message };
  }
}

 async getHealth() {
  try {
    const response = await this._makeHttpRequest(`${this.baseUrl}/health`, {
      method: 'GET',
      timeout: 5000
    });

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }

    const health = await response.json();
    return { success: true, health };

  } catch (error) {
    return { success: false, error: error.message };
  }
}

 // ===== TOOL CALL EVENT SYSTEM =====
 addToolCallListener(listener) {
   this.toolCallListeners.add(listener);
 }

 removeToolCallListener(listener) {
   this.toolCallListeners.delete(listener);
 }

 notifyToolCallListeners(event) {
   this.toolCallListeners.forEach(listener => {
     try {
       listener(event);
     } catch (error) {
       console.error('Error in tool call listener:', error);
     }
   });
 }

 // ===== UTILITY METHODS =====
 findSAMPath() {
   const possiblePaths = [
     path.join(__dirname, '..', '..', 'sam_agent.py'),
     path.join(__dirname, '..', 'sam_agent.py'),
     path.join(process.cwd(), '..', 'sam_agent.py'),
     path.join(process.cwd(), 'sam_agent.py'),
   ];

   for (const samPath of possiblePaths) {
     console.log(`Checking for SAM agent at: ${samPath}`);
     if (fs.existsSync(samPath)) {
       console.log(`Found SAM agent at: ${samPath}`);
       return samPath;
     }
   }

   throw new Error('sam_agent.py not found. Please ensure the SAM agent is in the project directory.');
 }

 isConnected() {
   return this.isServerRunning && this.wsConnection && this.wsConnection.readyState === WebSocket.OPEN;
 }

 getConnectionStatus() {
   return {
     serverRunning: this.isServerRunning,
     websocketConnected: this.wsConnection && this.wsConnection.readyState === WebSocket.OPEN,
     reconnectAttempts: this.reconnectAttempts
   };
 }
}

module.exports = SAMBridge;