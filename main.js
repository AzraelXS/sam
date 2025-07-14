const { app, BrowserWindow, ipcMain, dialog, Menu, Tray, shell, nativeImage } = require('electron');
const path = require('path');
const fs = require('fs-extra');
const SAMBridge = require('./src/sam-bridge');

let mainWindow;
let tray;
let samBridge;
let isDev = process.argv.includes('--dev');

// Enable live reload for development
if (isDev) {
  try {
    require('electron-reload')(__dirname, {
      electron: path.join(__dirname, 'node_modules', '.bin', 'electron'),
      hardResetMethod: 'exit'
    });
  } catch (e) {
    console.log('electron-reload not available, skipping live reload');
  }
}

// Register ALL IPC handlers ONCE
function setupIpcHandlers() {
  // ===== CORE SAM COMMUNICATION =====
  ipcMain.handle('send-message', async (event, message) => {
    if (samBridge) {
      return await samBridge.sendMessage(message);
    }
    return { success: false, error: 'SAM bridge not available' };
  });

  ipcMain.handle('execute-tool-direct', async (event, toolName, toolArgs) => {
    console.log(`🔧 [IPC] Direct tool execution: ${toolName}`, toolArgs);

    if (samBridge) {
      // Send starting notification
      if (mainWindow) {
        mainWindow.webContents.send('tool-call-update', {
          name: toolName,
          arguments: toolArgs,
          status: 'starting',
          timestamp: Date.now()
        });
      }

      try {
        const result = await samBridge.executeToolDirect(toolName, toolArgs);
        console.log(`🔧 [IPC] Tool ${toolName} completed`);

        // Send completion notification
        if (mainWindow) {
          mainWindow.webContents.send('tool-call-update', {
            name: toolName,
            arguments: toolArgs,
            status: 'completed',
            result: result,
            timestamp: Date.now()
          });
        }

        return result;
      } catch (error) {
        console.error(`🔧 [IPC] Tool ${toolName} failed:`, error);

        // Send error notification
        if (mainWindow) {
          mainWindow.webContents.send('tool-call-update', {
            name: toolName,
            arguments: toolArgs,
            status: 'error',
            error: error.message,
            timestamp: Date.now()
          });
        }

        return { success: false, error: error.message };
      }
    }

    return { success: false, error: 'SAM bridge not available' };
  });

  ipcMain.handle('get-capabilities', async () => {
    if (samBridge) {
      return await samBridge.getCapabilities();
    }
    return { success: false, error: 'SAM bridge not available' };
  });

  ipcMain.handle('get-health', async () => {
    if (samBridge) {
      return await samBridge.getHealth();
    }
    return { success: false, error: 'SAM bridge not available' };
  });

  ipcMain.handle('get-connection-status', () => {
    if (samBridge) {
      return samBridge.getConnectionStatus();
    }
    return { serverRunning: false, websocketConnected: false, reconnectAttempts: 0 };
  });

  // ===== FILE OPERATIONS =====
  ipcMain.handle('open-file-dialog', async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ['openFile', 'multiSelections'],
      filters: [
        { name: 'Text Files', extensions: ['txt', 'md', 'py', 'js', 'json', 'html', 'css'] },
        { name: 'All Files', extensions: ['*'] }
      ]
    });

    if (!result.canceled && result.filePaths.length > 0) {
      const files = [];

      for (const filePath of result.filePaths) {
        try {
          const stats = await fs.stat(filePath);
          const fileName = path.basename(filePath);
          const fileExt = path.extname(filePath).toLowerCase();

          // Read file content (limit size for safety)
          let content = '';
          if (stats.size < 1024 * 1024) { // 1MB limit
            content = await fs.readFile(filePath, 'utf-8');
          } else {
            content = '[File too large to display]';
          }

          files.push({
            name: fileName,
            path: filePath,
            size: stats.size,
            type: fileExt,
            content: content
          });
        } catch (error) {
          return { success: false, error: `Error reading ${filePath}: ${error.message}` };
        }
      }

      return { success: true, files };
    }

    return { success: false, error: 'No files selected' };
  });

  ipcMain.handle('save-conversation', async (event, conversation) => {
    const result = await dialog.showSaveDialog(mainWindow, {
      defaultPath: `sam-conversation-${new Date().toISOString().split('T')[0]}.json`,
      filters: [
        { name: 'JSON Files', extensions: ['json'] },
        { name: 'Text Files', extensions: ['txt'] }
      ]
    });

    if (!result.canceled) {
      try {
        await fs.writeFile(result.filePath, JSON.stringify(conversation, null, 2));
        return { success: true, path: result.filePath };
      } catch (error) {
        return { success: false, error: error.message };
      }
    }

    return { success: false, error: 'Save cancelled' };
  });

  // ===== APP INFO & UTILITIES =====
  ipcMain.handle('get-app-info', async () => {
    return {
      version: app.getVersion(),
      name: app.getName(),
      platform: process.platform,
      arch: process.arch,
      electronVersion: process.versions.electron,
      nodeVersion: process.versions.node
    };
  });

  ipcMain.handle('open-external-link', async (event, url) => {
    await shell.openExternal(url);
    return { success: true };
  });

  ipcMain.handle('show-item-in-folder', async (event, filePath) => {
    shell.showItemInFolder(filePath);
    return { success: true };
  });

  ipcMain.handle('fix-input-focus', async () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.focus();
      return { success: true };
    }
    return { success: false };
});

  // ===== DEVELOPMENT HELPERS =====
  if (isDev) {
    ipcMain.handle('dev-reload', () => {
      mainWindow.reload();
      return { success: true };
    });

    ipcMain.handle('dev-toggle-devtools', () => {
      mainWindow.webContents.toggleDevTools();
      return { success: true };
    });
  }
}

function createWindow() {
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 800,
    minHeight: 600,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      preload: path.join(__dirname, 'preload.js')
    },
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    show: false,
    icon: path.join(__dirname, 'assets', 'icons', 'icon.png')
  });

  // Load the app
  mainWindow.loadFile('index.html');

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();

    if (isDev) {
      mainWindow.webContents.openDevTools();
    }
  });

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Initialize SAM bridge
  samBridge = new SAMBridge();

  // Set up tool call listener to forward events to renderer
  samBridge.addToolCallListener((event) => {
    if (mainWindow) {
      mainWindow.webContents.send('tool-call-update', event);
    }
  });

  console.log('🤖 SAM Bridge initialized');
}

// App event handlers
app.whenReady().then(() => {
  setupIpcHandlers();
  createWindow();

  // Create system tray
  if (!tray) {
    const trayIcon = nativeImage.createFromPath(path.join(__dirname, 'assets', 'icons', 'tray.png'));
    tray = new Tray(trayIcon);

    const contextMenu = Menu.buildFromTemplate([
      { label: 'Show SAM', click: () => mainWindow?.show() },
      { label: 'Hide SAM', click: () => mainWindow?.hide() },
      { type: 'separator' },
      { label: 'Quit SAM', click: () => app.quit() }
    ]);

    tray.setContextMenu(contextMenu);
    tray.setToolTip('SAM AI Agent');

    tray.on('click', () => {
      if (mainWindow) {
        if (mainWindow.isVisible()) {
          mainWindow.hide();
        } else {
          mainWindow.show();
        }
      }
    });
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  if (samBridge) {
    samBridge.stopServer();
  }
});

// Security: Prevent new window creation
app.on('web-contents-created', (event, contents) => {
  contents.on('new-window', (event, navigationUrl) => {
    event.preventDefault();
    shell.openExternal(navigationUrl);
  });
});

// Handle certificate errors
app.on('certificate-error', (event, webContents, url, error, certificate, callback) => {
  if (isDev) {
    // In development, ignore certificate errors
    event.preventDefault();
    callback(true);
  } else {
    // In production, use default behavior
    callback(false);
  }
});

module.exports = { mainWindow, samBridge };