const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('samAPI', {
  // ===== CORE SAM COMMUNICATION =====
  sendMessage: (message) => ipcRenderer.invoke('send-message', message),
  executeToolDirect: (toolName, toolArgs) => ipcRenderer.invoke('execute-tool-direct', toolName, toolArgs),
  getCapabilities: () => ipcRenderer.invoke('get-capabilities'),
  getHealth: () => ipcRenderer.invoke('get-health'),
  getConnectionStatus: () => ipcRenderer.invoke('get-connection-status'),

  // ===== EVENT LISTENERS =====
  onToolCallUpdate: (callback) => {
    ipcRenderer.on('tool-call-update', (event, data) => callback(data));
  },

  removeToolCallListener: () => {
    ipcRenderer.removeAllListeners('tool-call-update');
  },

  // ===== FILE OPERATIONS =====
  openFileDialog: () => ipcRenderer.invoke('open-file-dialog'),
  saveConversation: (conversation) => ipcRenderer.invoke('save-conversation', conversation),

  // ===== APP INFO & UTILITIES =====
  getAppInfo: () => ipcRenderer.invoke('get-app-info'),
  openExternalLink: (url) => ipcRenderer.invoke('open-external-link', url),
  showItemInFolder: (filePath) => ipcRenderer.invoke('show-item-in-folder', filePath),
  fixInputFocus: () => ipcRenderer.invoke('fix-input-focus'),

  // ===== DEVELOPMENT HELPERS =====
  devReload: () => ipcRenderer.invoke('dev-reload'),
  devToggleDevTools: () => ipcRenderer.invoke('dev-toggle-devtools'),
});

// Also expose as electronAPI for backward compatibility during transition
contextBridge.exposeInMainWorld('electronAPI', {
  sendMessage: (message) => ipcRenderer.invoke('send-message', message),
  executeToolDirect: (toolName, toolArgs) => ipcRenderer.invoke('execute-tool-direct', toolName, toolArgs),
  getAppInfo: () => ipcRenderer.invoke('get-app-info'),
  openFileDialog: () => ipcRenderer.invoke('open-file-dialog'),
  saveConversation: (conversation) => ipcRenderer.invoke('save-conversation', conversation),
  fixInputFocus: () => ipcRenderer.invoke('fix-input-focus'),

  onToolCallUpdate: (callback) => {
    ipcRenderer.on('tool-call-update', (event, data) => callback(data));
  }
});

console.log('🔌 SAM Preload script loaded');