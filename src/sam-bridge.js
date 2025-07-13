const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const WebSocket = require('ws');

class SAMBridge {
  constructor() {
    this.serverProcess = null;
    this.isServerRunning = false;
    this.serverStartPromise = null;
    this.baseUrl = 'http://localhost:8888';
    this.sessionId = 'electron-session';
    this.toolCallListeners = new Set();
    this.wsConnection = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
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

      // Start the Python server process
      this.serverProcess = spawn('python', [samPath, '--api', '--api-port', '8888'], {
        cwd: path.dirname(samPath),
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env }
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

  async waitForServerReady(timeout = 30000) {
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      try {
        const response = await fetch(`${this.baseUrl}/health`);
        if (response.ok) {
          console.log('✅ SAM server is ready');
          return true;
        }
      } catch (error) {
        // Server not ready yet, continue waiting
      }

      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    throw new Error('SAM server failed to start within timeout period');
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
      const response = await fetch(`${this.baseUrl}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: message,
          session_id: this.sessionId,
          auto_approve: true,
          max_iterations: 10,
          verbose: false
        })
      });

      if (!response.ok) {
        throw new Error(`Query failed: ${response.statusText}`);
      }

      const result = await response.json();
      return { success: true, response: result };

    } catch (error) {
      console.error('Error sending message:', error);

      // Handle connection errors by trying to restart server
      if (error.message.includes('fetch') || error.message.includes('ECONNREFUSED')) {
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
      const response = await fetch(`${this.baseUrl}/execute-tool`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tool_name: toolName,
          arguments: toolArgs,
          session_id: this.sessionId
        })
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
     const response = await fetch(`${this.baseUrl}/tools`);
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
     const response = await fetch(`${this.baseUrl}/health`);
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