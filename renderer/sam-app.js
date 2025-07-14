class SAMApp {
  constructor() {
    this.conversation = [];
    this.isThinking = false;
    this.settings = this.loadSettings();

    this.init();
  }

  async init() {
    console.log('🤖 Initializing SAM App...');

    // Initialize DOM elements
    this.initializeDOM();

    // Set up event listeners
    this.setupEventListeners();

    // Apply saved settings
    this.applySettings();

    // Load app info
    await this.loadAppInfo();

    // Set up tool call listener
    this.setupToolCallListener();

    // Check SAM agent status
    await this.checkSAMStatus();

    console.log('✅ SAM App initialized');
  }

  initializeDOM() {
    // Core elements
    this.messagesContainer = document.getElementById('messages-container');
    this.messageInput = document.getElementById('message-input');
    this.sendButton = document.getElementById('send-button');
    this.sidebar = document.getElementById('sidebar');
    this.sidebarToggle = document.getElementById('sidebar-toggle');

    // Modal elements
    this.settingsModal = document.getElementById('settings-modal');
    this.aboutModal = document.getElementById('about-modal');

    // Navigation buttons
    this.newChatBtn = document.getElementById('new-chat-btn');
    this.clearChatBtn = document.getElementById('clear-chat-btn');
    this.saveChatBtn = document.getElementById('save-chat-btn');
    this.settingsBtn = document.getElementById('settings-btn');
    this.aboutBtn = document.getElementById('about-btn');
    this.toolsBtn = document.getElementById('tools-btn');
    this.capabilitiesBtn = document.getElementById('capabilities-btn');
  }

  setupEventListeners() {
    // Message input and sending
    this.messageInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });

    this.messageInput.addEventListener('input', () => {
      this.autoResizeTextarea();
    });

    this.sendButton.addEventListener('click', () => {
      this.sendMessage();
    });

    // Sidebar controls
    this.sidebarToggle.addEventListener('click', () => {
      this.toggleSidebar();
    });

    // Navigation buttons
    this.newChatBtn.addEventListener('click', () => {
      this.newChat();
    });

    this.clearChatBtn.addEventListener('click', () => {
      this.clearChat();
    });

    this.saveChatBtn.addEventListener('click', () => {
      this.saveChat();
    });

    this.settingsBtn.addEventListener('click', () => {
      this.openSettings();
    });

    this.aboutBtn.addEventListener('click', () => {
      this.openAbout();
    });

    this.toolsBtn.addEventListener('click', () => {
      this.showTools();
    });

    this.capabilitiesBtn.addEventListener('click', () => {
      this.showCapabilities();
    });

    // Modal controls
    this.setupModalControls();

    // Settings controls
    this.setupSettingsControls();
  }

  setupModalControls() {
    // Settings modal
    document.getElementById('close-settings-modal').addEventListener('click', () => {
      this.closeSettings();
    });

    document.getElementById('cancel-settings-btn').addEventListener('click', () => {
      this.closeSettings();
    });

    document.getElementById('save-settings-btn').addEventListener('click', () => {
      this.saveSettings();
    });

    // About modal
    document.getElementById('close-about-modal').addEventListener('click', () => {
      this.closeAbout();
    });

    document.getElementById('close-about-ok-btn').addEventListener('click', () => {
      this.closeAbout();
    });

    // Close modals when clicking outside
    this.settingsModal.addEventListener('click', (e) => {
      if (e.target === this.settingsModal) {
        this.closeSettings();
      }
    });

    this.aboutModal.addEventListener('click', (e) => {
      if (e.target === this.aboutModal) {
        this.closeAbout();
      }
    });
  }

  setupSettingsControls() {
    // Theme change
    document.getElementById('theme-select').addEventListener('change', (e) => {
      this.settings.theme = e.target.value;
      this.applyTheme();
    });

    // Font size change
    document.getElementById('font-size-select').addEventListener('change', (e) => {
      this.settings.fontSize = e.target.value;
      this.applyFontSize();
    });

    // Auto-scroll change
    document.getElementById('auto-scroll-checkbox').addEventListener('change', (e) => {
      this.settings.autoScroll = e.target.checked;
    });

    // Sound notifications change
    document.getElementById('sound-notifications-checkbox').addEventListener('change', (e) => {
      this.settings.soundNotifications = e.target.checked;
    });
  }

  setupToolCallListener() {
    if (window.samAPI && window.samAPI.onToolCallUpdate) {
      window.samAPI.onToolCallUpdate((event) => {
        this.handleToolCallUpdate(event);
      });
    }
  }

  handleToolCallUpdate(event) {
    console.log('🔧 Tool call update:', event);

    switch (event.status) {
      case 'starting':
        this.addToolMessage(`🔧 Executing tool: ${event.name}...`);
        break;
      case 'completed':
        this.addToolMessage(`✅ Tool ${event.name} completed successfully`);
        break;
      case 'error':
        this.addToolMessage(`❌ Tool ${event.name} failed: ${event.error}`);
        break;
    }
  }

  async sendMessage() {
    const message = this.messageInput.value.trim();
    if (!message || this.isThinking) return;

    try {
      // Add user message to conversation
      await this.addMessage('user', message);

      // Clear input
      this.messageInput.value = '';
      this.autoResizeTextarea();

      // Show thinking indicator
      this.showThinking();

      // Send message to SAM
      const result = await window.samAPI.sendMessage(message);

      // Hide thinking indicator
      this.hideThinking();

      if (result.success) {
        await this.addMessage('assistant', result.response.response || result.response);
      } else {
        await this.addMessage('assistant', `❌ Error: ${result.error}`);
      }

    } catch (error) {
      this.hideThinking();
      await this.addMessage('assistant', `❌ Error: ${error.message}`);
    }
  }

  async addMessage(role, content) {
    const message = {
      role,
      content,
      timestamp: Date.now()
    };

    this.conversation.push(message);

    // Create message element
    const messageEl = this.createMessageElement(message);
    this.messagesContainer.appendChild(messageEl);

    // Scroll to bottom
    if (this.settings.autoScroll) {
      this.scrollToBottom();
    }

    // Process markdown and code highlighting
    this.processMessage(messageEl);
  }

  addToolMessage(content) {
    const toolMessage = document.createElement('div');
    toolMessage.className = 'tool-message';
    toolMessage.innerHTML = `
      <div class="tool-content">
        <span class="tool-text">${content}</span>
        <span class="tool-timestamp">${new Date().toLocaleTimeString()}</span>
      </div>
    `;

    this.messagesContainer.appendChild(toolMessage);

    if (this.settings.autoScroll) {
      this.scrollToBottom();
    }
  }

  createMessageElement(message) {
    const messageEl = document.createElement('div');
    messageEl.className = `message ${message.role}`;

    const timeStr = new Date(message.timestamp).toLocaleTimeString();

    messageEl.innerHTML = `
      <div class="message-content">
        <div class="message-text">${this.escapeHtml(message.content)}</div>
        <div class="message-timestamp">${timeStr}</div>
      </div>
    `;

    return messageEl;
  }

  processMessage(messageEl) {
    const textEl = messageEl.querySelector('.message-text');
    if (!textEl) return;

    // Process markdown
    if (window.marked) {
      textEl.innerHTML = marked.parse(textEl.textContent);
    }

    // Highlight code blocks
    if (window.hljs) {
      textEl.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightElement(block);
      });
    }
  }

  showThinking() {
    if (this.thinkingIndicator) return;

    this.isThinking = true;
    this.sendButton.disabled = true;

    this.thinkingIndicator = document.createElement('div');
    this.thinkingIndicator.className = 'thinking-indicator';
    this.thinkingIndicator.innerHTML = `
      <div class="thinking-dots">
        <div class="thinking-dot"></div>
        <div class="thinking-dot"></div>
        <div class="thinking-dot"></div>
      </div>
      <span>SAM is thinking...</span>
    `;

    this.messagesContainer.appendChild(this.thinkingIndicator);
    this.scrollToBottom();
  }

  hideThinking() {
    if (this.thinkingIndicator) {
      this.thinkingIndicator.remove();
      this.thinkingIndicator = null;
    }

    this.isThinking = false;
    this.sendButton.disabled = false;
  }

  autoResizeTextarea() {
    this.messageInput.style.height = 'auto';
    this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 200) + 'px';
  }

  scrollToBottom() {
    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
  }

  toggleSidebar() {
    this.sidebar.classList.toggle('collapsed');
  }

  newChat() {
    this.conversation = [];
    this.messagesContainer.innerHTML = `
      <div class="message assistant">
        <div class="message-content">
          <div class="message-text">
            <p>👋 Hello! I'm <strong class="text-gradient">SAM</strong> (Semi-Autonomous Model), your AI assistant.</p>
            <p>I can help you with various tasks including:</p>
            <ul>
              <li>💻 Code development and debugging</li>
              <li>📁 File and system operations</li>
              <li>🌐 Web research and analysis</li>
              <li>📝 Text processing and analysis</li>
              <li>🔧 Tool execution and automation</li>
            </ul>
            <p>What would you like me to help you with today?</p>
          </div>
          <div class="message-timestamp">Just now</div>
        </div>
      </div>
    `;
  }

  clearChat() {
    if (confirm('Are you sure you want to clear the current chat?')) {
      this.newChat();
    }
  }

  async saveChat() {
    if (this.conversation.length === 0) {
      alert('No conversation to save.');
      return;
    }

    try {
      const result = await window.samAPI.saveConversation(this.conversation);
      if (result.success) {
        alert(`Chat saved to: ${result.path}`);
      } else {
        alert(`Failed to save chat: ${result.error}`);
      }
    } catch (error) {
      alert(`Error saving chat: ${error.message}`);
    }
  }

  async showTools() {
    try {
      const result = await window.samAPI.getCapabilities();
      if (result.success) {
        const tools = result.tools;
        const toolList = Object.entries(tools).map(([name, info]) =>
          `• **${name}** (${info.category}): ${info.description}`
        ).join('\n');

        await this.addMessage('assistant', `🔧 **Available Tools:**\n\n${toolList}`);
      } else {
        await this.addMessage('assistant', `❌ Failed to get tools: ${result.error}`);
      }
    } catch (error) {
      await this.addMessage('assistant', `❌ Error getting tools: ${error.message}`);
    }
  }

  async showCapabilities() {
    try {
      const health = await window.samAPI.getHealth();
      const capabilities = await window.samAPI.getCapabilities();
      const status = await window.samAPI.getConnectionStatus();

      let response = '⚡ **SAM Capabilities:**\n\n';

      if (health.success) {
        response += `🤖 **Model:** ${health.health.model}\n`;
        response += `🔧 **Tools:** ${health.health.tools_count}\n`;
        response += `🔌 **Plugins:** ${health.health.plugins_count}\n`;
        response += `⏱️ **Uptime:** ${Math.round(health.health.uptime_seconds)}s\n\n`;
      }

      response += `🔗 **Connection Status:**\n`;
      response += `• Server: ${status.serverRunning ? '✅ Running' : '❌ Stopped'}\n`;
      response += `• WebSocket: ${status.websocketConnected ? '✅ Connected' : '❌ Disconnected'}\n`;

      if (capabilities.success) {
        const toolCount = Object.keys(capabilities.tools).length;
        response += `\n🔧 **Available Tools:** ${toolCount}\n`;
      }

      await this.addMessage('assistant', response);
    } catch (error) {
      await this.addMessage('assistant', `❌ Error getting capabilities: ${error.message}`);
    }
  }

  async checkSAMStatus() {
    try {
      const status = await window.samAPI.getConnectionStatus();
      console.log('🔗 SAM Status:', status);

      if (!status.serverRunning) {
        console.log('🚀 SAM server not running, will start on first message');
      }
    } catch (error) {
      console.error('❌ Error checking SAM status:', error);
    }
  }

  // Settings methods
  openSettings() {
    this.loadSettingsToUI();
    this.settingsModal.style.display = 'flex';
  }

  closeSettings() {
    this.settingsModal.style.display = 'none';
  }

  openAbout() {
    this.aboutModal.style.display = 'flex';
  }

  closeAbout() {
    this.aboutModal.style.display = 'none';
  }

  loadSettingsToUI() {
    document.getElementById('theme-select').value = this.settings.theme;
    document.getElementById('font-size-select').value = this.settings.fontSize;
    document.getElementById('auto-scroll-checkbox').checked = this.settings.autoScroll;
    document.getElementById('sound-notifications-checkbox').checked = this.settings.soundNotifications;
  }

  saveSettings() {
    this.settings = {
      theme: document.getElementById('theme-select').value,
      fontSize: document.getElementById('font-size-select').value,
      autoScroll: document.getElementById('auto-scroll-checkbox').checked,
      soundNotifications: document.getElementById('sound-notifications-checkbox').checked
    };

    this.applySettings();
    localStorage.setItem('sam-settings', JSON.stringify(this.settings));
    this.closeSettings();
  }

  loadSettings() {
    const defaultSettings = {
      theme: 'dark',
      fontSize: 'medium',
      autoScroll: true,
      soundNotifications: false
    };

    try {
      const saved = localStorage.getItem('sam-settings');
      return saved ? { ...defaultSettings, ...JSON.parse(saved) } : defaultSettings;
    } catch (error) {
      console.warn('Failed to load settings:', error);
      return defaultSettings;
    }
  }

  applySettings() {
    this.applyTheme();
    this.applyFontSize();
  }

  applyTheme() {
    const { theme } = this.settings;

    if (theme === 'light') {
      document.body.setAttribute('data-theme', 'light');
    } else if (theme === 'dark') {
      document.body.removeAttribute('data-theme');
    } else if (theme === 'auto') {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      if (prefersDark) {
        document.body.removeAttribute('data-theme');
      } else {
        document.body.setAttribute('data-theme', 'light');
      }
    }
  }

  applyFontSize() {
    const { fontSize } = this.settings;

    // Remove existing font size classes
    document.body.classList.remove('font-small', 'font-medium', 'font-large');

    // Apply new font size class
    document.body.classList.add(`font-${fontSize}`);
  }

  async loadAppInfo() {
    try {
      const info = await window.samAPI.getAppInfo();
      const appInfoEl = document.getElementById('app-info');
      if (appInfoEl) {
        appInfoEl.innerHTML = `
          <strong>${info.name}</strong> v${info.version}<br>
          Platform: ${info.platform} (${info.arch})<br>
          Electron: v${info.electronVersion}<br>
          Node.js: v${info.nodeVersion}
        `;
      }
    } catch (error) {
      console.error('Failed to load app info:', error);
    }
  }

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

// Add font size CSS classes
const fontSizeStyles = document.createElement('style');
fontSizeStyles.textContent = `
  .font-small {
    font-size: 13px;
  }
  .font-small .message-text {
    font-size: 13px;
  }
  .font-small #message-input {
    font-size: 13px;
  }
  .font-medium {
    font-size: 14px;
  }
  .font-large {
    font-size: 16px;
  }
  .font-large .message-text {
    font-size: 16px;
  }
  .font-large #message-input {
    font-size: 16px;
  }

  .tool-message {
    margin-bottom: 12px;
    padding: 8px 16px;
    background: var(--info-bg);
    border-radius: 8px;
    border-left: 3px solid var(--sam-accent);
  }

  .tool-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .tool-text {
    color: var(--sam-accent);
    font-style: italic;
    font-size: 14px;
  }

  .tool-timestamp {
    color: var(--text-muted);
    font-size: 11px;
  }
`;
document.head.appendChild(fontSizeStyles);

// Initialize the app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  window.samApp = new SAMApp();
});

// Handle system theme changes
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  if (window.samApp && window.samApp.settings.theme === 'auto') {
    window.samApp.applyTheme();
  }
});