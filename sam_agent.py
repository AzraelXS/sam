# !/usr/bin/env python3
"""
SAM Agent - Semi-Autonomous Model AI Agent
Enhanced with full Model Context Protocol (MCP) support
"""




import json
import logging
import time
import traceback
import re
import inspect
import asyncio
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable
from enum import Enum

# Import configuration
from config import SAMConfig

if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Set up logging (will be configured after loading config)
logger = logging.getLogger("SAMAgent")

# Optional imports with availability flags
try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not available - some functionality may be limited")

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available - API server functionality disabled")


# Tool categories
class ToolCategory(Enum):
    UTILITY = "utility"
    DEVELOPMENT = "development"
    FILESYSTEM = "filesystem"
    SYSTEM = "system"
    COMMUNICATION = "communication"
    WEB = "web"
    DATA = "data"
    SECURITY = "security"
    MULTIMEDIA = "multimedia"


@dataclass
class ToolInfo:
    """Information about a registered tool"""
    function: Callable
    description: str
    parameters: Dict[str, Any]
    category: ToolCategory
    requires_approval: bool = False
    usage_count: int = 0


def load_all_plugins(sam):
    """Auto-load all plugins from the plugins directory"""
    plugins_dir = Path(__file__).parent / "plugins"
    if not plugins_dir.exists():
        print("⚠️ Plugins directory not found")
        return

    loaded_count = 0
    print(f"🔍 Scanning for plugins in {plugins_dir}")

    # Load all .py files in plugins directory
    for plugin_file in plugins_dir.glob("*.py"):
        if plugin_file.name.startswith("_"):
            continue  # Skip __init__.py, __pycache__, etc.

        if sam.plugin_manager.load_plugin_from_file(str(plugin_file), sam):
            loaded_count += 1

    if loaded_count > 0:
        print(f"📦 Loaded {loaded_count} plugins, {len(sam.local_tools)} total tools")

# ===== PLUGIN SYSTEM =====
class SAMPlugin:
    """Base class for SAM plugins"""

    def __init__(self, name: str, version: str = "1.0.0", description: str = ""):
        self.name = name
        self.version = version
        self.description = description
        self.enabled = True

    def register_tools(self, agent):
        """Override this method to register tools with the agent"""
        pass

    def on_load(self, agent):
        """Called when plugin is loaded"""
        pass

    def on_unload(self, agent):
        """Called when plugin is unloaded"""
        pass


class PluginManager:
    """Manages SAM plugins"""

    def __init__(self):
        self.plugins: Dict[str, SAMPlugin] = {}

    def load_plugin_from_file(self, plugin_path: str, agent) -> bool:
        """Load a plugin from a Python file"""
        try:
            import importlib.util
            plugin_path_obj = Path(plugin_path)

            spec = importlib.util.spec_from_file_location("plugin", plugin_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[plugin_path_obj.stem] = module
            spec.loader.exec_module(module)

            # First try factory function
            if hasattr(module, 'create_plugin') and callable(getattr(module, 'create_plugin')):
                try:
                    plugin = module.create_plugin()
                    return self.register_plugin(plugin, agent)
                except Exception as e:
                    logger.error(f"Error calling create_plugin() in {plugin_path}: {str(e)}")
                    return False

            # Look for plugin class
            plugin_class = None
            for item_name in dir(module):
                item = getattr(module, item_name)
                if (isinstance(item, type) and
                        issubclass(item, SAMPlugin) and
                        item != SAMPlugin):
                    plugin_class = item
                    break

            if not plugin_class:
                logger.error(f"No SAMPlugin subclass found in {plugin_path}")
                return False

            # Instantiate and register
            plugin = plugin_class()
            return self.register_plugin(plugin, agent)

        except Exception as e:
            logger.error(f"Error loading plugin {plugin_path}: {str(e)}")
            return False

    def register_plugin(self, plugin: SAMPlugin, agent) -> bool:
        """Register a plugin instance"""
        try:
            if plugin.name in self.plugins:
                logger.warning(f"Plugin {plugin.name} already loaded, replacing...")

            self.plugins[plugin.name] = plugin
            plugin.on_load(agent)
            plugin.register_tools(agent)

            logger.info(f"Loaded plugin: {plugin.name} v{plugin.version}")
            return True

        except Exception as e:
            logger.error(f"Error registering plugin {plugin.name}: {str(e)}")
            return False

    def unload_plugin(self, plugin_name: str):
        """Unload a plugin"""
        if plugin_name in self.plugins:
            del self.plugins[plugin_name]
            logger.info(f"Unloaded plugin: {plugin_name}")


# ===== API MODELS (if FastAPI available) =====
if FASTAPI_AVAILABLE:
    class QueryRequest(BaseModel):
        message: str
        max_iterations: int = 5
        verbose: bool = False
        auto_approve: Optional[bool] = None
        session_id: Optional[str] = None


    class QueryResponse(BaseModel):
        response: str
        session_id: str
        status: str
        timestamp: str
        error: Optional[str] = None


    class ToolExecutionRequest(BaseModel):
        tool_name: str
        arguments: Dict[str, Any]


    class ToolExecutionResponse(BaseModel):
        success: bool
        result: Optional[str] = None
        error: Optional[str] = None
        execution_time: float
        tool_name: str


    class HealthResponse(BaseModel):
        status: str
        model: str
        tools_count: int
        plugins_count: int
        uptime_seconds: float


# ===== MAIN SAM AGENT CLASS =====
class SAMAgent:
    """Semi-Autonomous Model AI Agent with MCP support"""

    def __init__(self, model_name: str = None, context_limit: int = None, safety_mode: bool = True,
                 auto_approve: bool = False):
        """Initialize SAM Agent"""

        # Load configuration
        self.config = self._load_config()

        # Store raw config for provider operations
        if not hasattr(self, 'raw_config'):
            self.raw_config = {}

        # Configure logging based on config
        self._configure_logging()

        # Model configuration - use provider-aware logic
        # Model configuration - use provider-aware logic
        provider = self.raw_config.get('provider', 'lmstudio')
        if provider == 'claude':
            provider_config = self.raw_config.get('providers', {}).get('claude', {})
            self.base_url = "https://api.anthropic.com/v1"
            self.api_key = provider_config.get('api_key', '')  # ADD THIS LINE
            self.model_name = model_name or provider_config.get('model_name', 'claude-sonnet-4-20250514')
            self.context_limit = context_limit or provider_config.get('context_limit', 200000)
        else:
            # For LMStudio, prefer the provider-specific config, then fall back to model section
            lmstudio_config = self.raw_config.get('providers', {}).get('lmstudio', {})
            model_config = self.raw_config.get('model', {})

            self.base_url = self.config.lmstudio.base_url
            self.api_key = lmstudio_config.get('api_key', self.config.lmstudio.api_key)  # ADD THIS LINE
            # Try multiple possible locations for model name
            self.model_name = (model_name or
                               lmstudio_config.get('model_name') or
                               model_config.get('name') or
                               'qwen2.5-coder-14b-instruct')
            self.context_limit = context_limit or model_config.get('context_limit', 20000)

        # Agent state
        self.conversation_history = []
        self.safety_mode = safety_mode
        self.auto_approve = auto_approve
        self.stop_requested = False
        self.stop_message = ""

        # Tool management
        self.local_tools = {}
        self.tool_info = {}
        self.tools_by_category = {category: [] for category in ToolCategory}

        # MCP (Model Context Protocol) support
        self.mcp_sessions = {}
        self.mcp_tools = {}
        self._mcp_auto_connect_pending = True

        # Plugin system
        self.plugin_manager = PluginManager()

        logger.info(f"SAM Agent initialized with model: {self.model_name}")
        logger.info(f"Context limit: {self.context_limit:,} tokens")
        logger.info(f"Safety mode: {'ON' if self.safety_mode else 'OFF'}")

        # Note: MCP auto-connection will happen when run() is first called

    def switch_provider(self, provider_name: str) -> str:
        """Switch between providers"""
        if provider_name not in self.raw_config.get('providers', {}):
            return f"❌ Provider '{provider_name}' not found in config"

        self.raw_config['provider'] = provider_name

        # Update relevant settings based on provider
        provider_config = self.raw_config['providers'][provider_name]

        if provider_name == 'claude':
            self.context_limit = provider_config.get('context_limit', 200000)
            self.model_name = provider_config.get('model_name', 'claude-sonnet-4-20250514')
            # Claude doesn't need base_url update since it's handled in the completion method
        else:
            # For LMStudio, first set the fallback context limit from config
            # Try provider config first, then model config
            fallback_context = (provider_config.get('context_limit') or
                                self.raw_config.get('model', {}).get('context_limit', 20000))
            self.context_limit = fallback_context

            # Try to get actual context length from API if enabled
            if self.raw_config.get('features', {}).get('use_loaded_context_length', True):
                model_info = self._update_context_limit_from_api()
                if not model_info:
                    # If API query failed, keep the fallback value
                    logger.info(f"📊 Using configured context limit: {self.context_limit:,} tokens")

            self.model_name = provider_config.get('model_name', 'qwen2.5-coder-14b-instruct')

            # Update instance variables for LMStudio
            self.base_url = provider_config.get('base_url', self.base_url)
            self.api_key = provider_config.get('api_key', self.api_key)

        return f"✅ Switched to {provider_name} provider (model: {self.model_name}, context: {self.context_limit:,})"

    def get_current_provider(self) -> str:
        """Get current provider info"""
        current = self.raw_config.get('provider', 'lmstudio')  # default to lmstudio
        available = list(self.raw_config.get('providers', {}).keys())
        return f"📋 Current: {current} | Available: {', '.join(available)}"

    def _get_model_info(self):
        """Get model information from LMStudio API including loaded context length"""
        # List of endpoints to try
        endpoints_to_try = [
            f"{self.base_url.replace('/v1', '')}/v1/models",  # OpenAI-compatible
            f"{self.base_url.replace('/v1', '')}/api/v0/models"  # LMStudio REST API
        ]

        for endpoint_url in endpoints_to_try:
            try:
                response = requests.get(endpoint_url, timeout=10)

                if response.status_code == 200:
                    models_data = response.json()
                    logger.info(f"✅ Successfully connected to {endpoint_url}")

                    # Handle different response formats
                    models_list = models_data.get('data', models_data if isinstance(models_data, list) else [])

                    # Find the current model
                    for model in models_list:
                        if model.get('id') == self.model_name:
                            loaded_context = model.get('loaded_context_length', self.context_limit)
                            max_context = model.get('max_context_length', loaded_context)

                            logger.info(f"📊 Model: {self.model_name}")
                            logger.info(f"📊 Loaded context: {loaded_context:,} tokens")
                            logger.info(f"📊 Max context: {max_context:,} tokens")

                            return {
                                'model_id': model.get('id'),
                                'loaded_context_length': loaded_context,
                                'max_context_length': max_context,
                                'state': model.get('state', 'unknown')
                            }

                    # If we got a successful response but couldn't find the specific model
                    logger.info(f"📊 Connected to endpoint but model '{self.model_name}' not found in list")
                    return None
                else:
                    logger.debug(f"Endpoint {endpoint_url} returned status {response.status_code}")

            except Exception as e:
                logger.debug(f"Failed to connect to {endpoint_url}: {e}")
                continue

        # If we get here, none of the endpoints worked
        logger.warning(f"⚠️  Could not get model info from any endpoint")
        return None

    def _update_context_limit_from_api(self):
        """Update context limit based on actual loaded model"""
        model_info = self._get_model_info()

        if model_info and model_info.get('loaded_context_length'):
            old_limit = self.context_limit
            self.context_limit = model_info['loaded_context_length']

            if old_limit != self.context_limit:
                logger.info(f"🔄 Updated context limit: {old_limit:,} → {self.context_limit:,} tokens")

            return model_info

        return None

    def _configure_logging(self):
        """Configure logging based on config settings"""
        try:
            # Get logging level from config
            log_level_str = getattr(self.config.logging, 'level', 'INFO')
            log_level = getattr(logging, log_level_str.upper(), logging.INFO)

            # Configure logging
            logging.basicConfig(
                level=log_level,
                format=getattr(self.config.logging, 'format',
                               "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
                force=True  # Override any existing configuration
            )

            # Set console handler based on config
            console_enabled = getattr(self.config.logging, 'console_enabled', True)
            if not console_enabled:
                # Remove console handlers if disabled
                root_logger = logging.getLogger()
                for handler in root_logger.handlers[:]:
                    if isinstance(handler, logging.StreamHandler):
                        root_logger.removeHandler(handler)

        except Exception as e:
            # Fallback to INFO if config fails
            logging.basicConfig(level=logging.INFO, force=True)
            logger.warning(f"Failed to configure logging from config: {e}")

    async def _ensure_mcp_auto_connect(self):
        """Ensure MCP auto-connection happens once when needed"""
        if (self._mcp_auto_connect_pending and
                hasattr(self.config, 'mcp') and
                self.config.mcp.enabled and
                getattr(self.config.mcp, 'servers', None)):
            self._mcp_auto_connect_pending = False
            logger.info("Performing delayed MCP auto-connection...")
            await self._auto_connect_mcp_servers()
        elif self._mcp_auto_connect_pending:
            self._mcp_auto_connect_pending = False
            logger.info("MCP auto-connection skipped (disabled or no servers)")

    def _load_config(self) -> SAMConfig:
        """Load configuration from config.json or create default"""
        config_path = Path("config.json")

        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    logger.info("Loaded configuration from config.json")

                # Store the raw config data for provider switching
                self.raw_config = config_data

                # Create SAMConfig from loaded data
                config = SAMConfig()

                # Update config with loaded data
                if 'lmstudio' in config_data:
                    for key, value in config_data['lmstudio'].items():
                        if hasattr(config.lmstudio, key):
                            setattr(config.lmstudio, key, value)

                # Handle model config - your config.json uses "name" but config.py expects "model_name"
                if 'model' in config_data:
                    model_data = config_data['model']
                    if 'name' in model_data:
                        config.model.model_name = model_data['name']
                    for key, value in model_data.items():
                        if key != 'name' and hasattr(config.model, key):
                            setattr(config.model, key, value)

                if 'mcp' in config_data:
                    for key, value in config_data['mcp'].items():
                        if hasattr(config.mcp, key):
                            setattr(config.mcp, key, value)

                if 'logging' in config_data:
                    for key, value in config_data['logging'].items():
                        if hasattr(config.logging, key):
                            setattr(config.logging, key, value)

                return config

            except Exception as e:
                logger.warning(f"Failed to load config.json: {e}, using defaults")
                self.raw_config = {}
                return SAMConfig()
        else:
            logger.info("No config.json found, using default configuration")
            self.raw_config = {}
            return SAMConfig()

    # ===== LLM COMMUNICATION =====
    def generate_chat_completion(self, messages: List[Dict], **kwargs) -> str:
        """Generate chat completion using the configured provider"""
        provider = self.raw_config.get('provider', 'lmstudio')

        if provider == 'claude':
            return self._generate_claude_completion(messages, **kwargs)
        else:
            return self._generate_lmstudio_completion(messages, **kwargs)

    def _generate_claude_completion(self, messages: List[Dict], **kwargs) -> str:
        """Generate completion using Claude API"""
        try:
            # Import anthropic here to avoid dependency issues
            try:
                import anthropic
            except ImportError:
                return "Error: anthropic package not installed. Run: pip install anthropic"

            # Get Claude config
            claude_config = self.raw_config.get('providers', {}).get('claude', {})
            api_key = claude_config.get('api_key')

            if not api_key:
                return "Error: Claude API key not found in config"

            # Create client
            client = anthropic.Anthropic(api_key=api_key)

            # Prepare parameters
            model = claude_config.get('model_name', 'claude-sonnet-4-20250514')
            final_max_tokens = kwargs.get('max_tokens', 4000)
            final_temperature = kwargs.get('temperature', 0.3)

            # Convert messages to Claude format and strip all content
            claude_messages = []
            system_content = ""

            for msg in messages:
                if msg['role'] == 'system':
                    # Strip system content and ensure no trailing whitespace
                    content = str(msg['content']).strip()
                    if content:
                        system_content += content + "\n"
                else:
                    # Strip all message content to prevent trailing whitespace issues
                    cleaned_msg = {
                        'role': msg['role'],
                        'content': str(msg['content']).strip()
                    }
                    # Only add non-empty messages
                    if cleaned_msg['content']:
                        claude_messages.append(cleaned_msg)

            # Ensure system content is properly stripped
            system_content = system_content.strip() if system_content else None

            # Create message with system prompt
            response = client.messages.create(
                model=model,
                max_tokens=final_max_tokens,
                temperature=final_temperature,
                system=system_content,
                messages=claude_messages
            )

            # Extract text response and ensure no trailing whitespace
            response_text = ""
            for content_block in response.content:
                if hasattr(content_block, 'text'):
                    response_text += content_block.text

            # Critical: Strip all whitespace from the response
            return response_text.strip()

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return f"Error calling Claude API: {str(e)}"

    def _generate_lmstudio_completion(self, messages: List[Dict], **kwargs) -> str:
        """Generate completion using LMStudio API"""
        if not REQUESTS_AVAILABLE:
            return "❌ Error: requests library not available"

        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.3),
                "max_tokens": kwargs.get("max_tokens", 2000),
                "stream": False
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                error_msg = f"API Error {response.status_code}: {response.text}"
                logger.error(error_msg)
                return error_msg

        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return error_msg

    # ===== MCP (MODEL CONTEXT PROTOCOL) SUPPORT =====
    async def _auto_connect_mcp_servers(self):
        """Automatically connect to configured MCP servers"""
        if not hasattr(self.config, 'mcp') or not self.config.mcp.servers:
            logger.info("No MCP servers configured for auto-connection")
            return

        # Filter for enabled servers only
        enabled_servers = {
            name: config for name, config in self.config.mcp.servers.items()
            if config.get('enabled', True)  # Default to True if not specified
        }

        if not enabled_servers:
            logger.info("No enabled MCP servers found for auto-connection")
            return

        logger.info(f"Auto-connecting to {len(enabled_servers)} enabled MCP servers...")

        for server_name, server_config in enabled_servers.items():
            try:
                server_type = server_config.get('type', 'stdio')
                server_path = server_config.get('path', '')
                headers = server_config.get('headers', {})

                success = await self.connect_to_mcp_server(
                    server_name=server_name,
                    server_type=server_type,
                    server_path_or_url=server_path,
                    headers=headers
                )

                if success:
                    logger.info(f"✅ Connected to MCP server: {server_name}")
                else:
                    logger.warning(f"❌ Failed to connect to MCP server: {server_name}")

            except Exception as e:
                logger.error(f"❌ Error connecting to MCP server {server_name}: {str(e)}")

    async def connect_to_mcp_server(self, server_name: str, server_type: str,
                                    server_path_or_url: str, headers: Dict[str, str] = None) -> bool:
        """Connect to an MCP server"""
        server_type = server_type.lower()

        logger.info(f"Connecting to MCP server: {server_name} ({server_type})")

        try:
            if server_type == 'stdio':
                return await self._connect_stdio_mcp(server_name, server_path_or_url)
            elif server_type == 'websocket':
                return await self._connect_websocket_mcp(server_name, server_path_or_url, headers)
            elif server_type in ['http', 'sse']:
                return await self._connect_http_mcp(server_name, server_path_or_url, headers)
            else:
                logger.error(f"Unsupported MCP server type: {server_type}")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to MCP server {server_name}: {str(e)}")
            return False

    async def _connect_stdio_mcp(self, server_name: str, server_path: str) -> bool:
        """Connect to a stdio MCP server"""
        try:
            from mcp import stdio_client
            from mcp.client.stdio import StdioServerParameters
        except ImportError:
            logger.error("MCP client not available. Install with: pip install mcp")
            return False

        if not os.path.exists(server_path):
            logger.error(f"MCP server script not found at {server_path}")
            return False

        try:
            # Determine command based on script extension
            if server_path.endswith('.js'):
                command = "node"
            elif server_path.endswith('.py'):
                command = "python" if sys.platform == "win32" else "python3"
            else:
                logger.error(f"Unsupported server script type: {server_path}")
                return False

            server_params = StdioServerParameters(command=command, args=[server_path])
            session = await stdio_client(server_params)

            self.mcp_sessions[server_name] = session
            await self._register_mcp_tools(server_name, session)

            logger.info(f"Successfully connected to stdio MCP server: {server_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to stdio MCP server {server_name}: {str(e)}")
            return False

    async def _connect_websocket_mcp(self, server_name: str, ws_url: str, headers: Dict[str, str] = None) -> bool:
        """Connect to a WebSocket MCP server"""
        try:
            from mcp_transports import WebSocketMCPTransport
            import websockets
        except ImportError:
            logger.error("WebSocket support not available. Install 'websockets' package.")
            return False

        try:
            connection_kwargs = {'ping_interval': 30, 'ping_timeout': 10, 'close_timeout': 10}
            if headers:
                connection_kwargs['additional_headers'] = headers

            websocket = await websockets.connect(ws_url, **connection_kwargs)
            session = WebSocketMCPTransport(websocket, server_name)
            await session.initialize()

            self.mcp_sessions[server_name] = session
            await self._register_mcp_tools(server_name, session)

            logger.info(f"Successfully connected to WebSocket MCP server: {server_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to WebSocket MCP server {server_name}: {str(e)}")
            return False

    async def _connect_http_mcp(self, server_name: str, base_url: str, headers: Dict[str, str] = None) -> bool:
        """Connect to an HTTP/SSE MCP server"""
        try:
            from mcp_transports import StreamableHTTPMCPTransport
            import httpx
        except ImportError:
            logger.error("HTTP support not available. Install 'httpx' package.")
            return False

        try:
            client = httpx.AsyncClient(timeout=httpx.Timeout(30.0), follow_redirects=True)
            session = StreamableHTTPMCPTransport(client, base_url, server_name, headers)
            await session.initialize()

            self.mcp_sessions[server_name] = session
            await self._register_mcp_tools(server_name, session)

            logger.info(f"Successfully connected to HTTP MCP server: {server_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to HTTP MCP server {server_name}: {str(e)}")
            return False

    async def _register_mcp_tools(self, server_name: str, session):
        """Register tools from an MCP server"""
        try:
            tools_result = await session.list_tools()
            if not tools_result or not tools_result.tools:
                logger.warning(f"No tools found in server: {server_name}")
                return

            for tool in tools_result.tools:
                tool_info = {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": getattr(tool, 'input_schema', {}),
                    "server": server_name
                }

                self.mcp_tools[tool.name] = (server_name, tool_info)
                logger.debug(f"Registered MCP tool: {tool.name} from server: {server_name}")

            logger.info(f"Registered {len(tools_result.tools)} tools from server: {server_name}")

        except Exception as e:
            logger.error(f"Error registering tools from server {server_name}: {str(e)}")

    async def _execute_mcp_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute an MCP tool - THE MISSING METHOD!"""
        if tool_name not in self.mcp_tools:
            return f"❌ MCP tool not found: {tool_name}"

        server_name, tool_info = self.mcp_tools[tool_name]
        session = self.mcp_sessions.get(server_name)

        if not session:
            return f"❌ MCP server {server_name} is not connected"

        try:
            print(f"\n🌐 Executing MCP tool: {tool_name} on server: {server_name}")
            start_time = time.time()

            # Execute the tool via MCP
            tool_result = await session.run_tool(tool_name, args)
            result = tool_result.result

            execution_time = time.time() - start_time
            print(f"✅ MCP tool completed in {execution_time:.3f}s")

            # Display raw results
            print(f"\n📊 MCP TOOL RESULTS:")
            print("=" * 60)
            print(str(result))
            print("=" * 60)

            return str(result)

        except Exception as e:
            error_msg = f"Error executing MCP tool {tool_name}: {str(e)}"
            logger.error(error_msg)
            print(f"\n❌ MCP TOOL ERROR:")
            print("=" * 60)
            print(error_msg)
            print("=" * 60)
            return error_msg

    async def disconnect_mcp_servers(self):
        """Disconnect from all MCP servers"""
        for server_name, session in self.mcp_sessions.items():
            try:
                await session.close()
                logger.info(f"Disconnected from MCP server: {server_name}")
            except Exception as e:
                logger.error(f"Error disconnecting from server {server_name}: {str(e)}")

        self.mcp_sessions = {}
        self.mcp_tools = {}

    def list_mcp_servers(self) -> Dict[str, Any]:
        """List all connected MCP servers"""
        servers = {}
        for server_name, session in self.mcp_sessions.items():
            server_tools = [tool for tool, (srv, _) in self.mcp_tools.items() if srv == server_name]
            servers[server_name] = {
                "status": "connected",
                "tools": server_tools,
                "tool_count": len(server_tools)
            }
        return servers

    # ===== SAFETY AND CONTROL =====
    def get_safety_status(self) -> str:
        """Get current safety configuration status"""
        return (f"🛡️ Safety Mode: {'ON' if self.safety_mode else 'OFF'} | "
                f"🤖 Auto-approve: {'ON' if self.auto_approve else 'OFF'}")

    def get_detailed_safety_status(self) -> Dict[str, Any]:
        """Get detailed safety status for API responses"""
        return {
            "safety_mode": self.safety_mode,
            "auto_approve": self.auto_approve,
            "tools_count": len(self.local_tools) + len(self.mcp_tools),
            "local_tools": len(self.local_tools),
            "mcp_tools": len(self.mcp_tools),
            "mcp_servers": len(self.mcp_sessions)
        }

    def set_safety_mode(self, enabled: bool) -> str:
        """Enable or disable safety mode"""
        self.safety_mode = enabled
        status = "ON" if enabled else "OFF"
        result = f"🛡️ Safety mode {status}"
        logger.info(result)
        return result

    def set_auto_approve(self, enabled: bool) -> str:
        """Enable or disable auto-approve mode"""
        self.auto_approve = enabled
        status = "ON" if enabled else "OFF"
        result = f"🤖 Auto-approve {status}"
        logger.info(result)
        return result

    def _prompt_for_approval(self, tool_name: str, args: Dict[str, Any], tool_info: ToolInfo = None) -> bool:
        """Prompt user for tool execution approval"""
        print(f"\n" + "=" * 60)
        print(f"🛡️  TOOL APPROVAL REQUIRED")
        print("=" * 60)

        print(f"🔧 Tool: {tool_name}")
        if tool_info:
            print(f"📂 Category: {tool_info.category.value}")
            print(f"📄 Description: {tool_info.description}")

        # Show arguments in a cleaner format
        print(f"\n📋 Arguments:")
        for key, value in args.items():
            # Truncate long values for display
            display_value = str(value)
            if len(display_value) > 80:
                display_value = display_value[:80] + "..."
            print(f"   {key}: {display_value}")

        print(f"\n⚡ Options: [y]es | [n]o | [i]nfo | [s]top")
        print("=" * 60)

        while True:
            try:
                response = input("🤔 Approve? ").strip().lower()

                if response in ['y', 'yes']:
                    print("✅ Tool execution approved")
                    return True
                elif response in ['n', 'no']:
                    print("❌ Tool execution denied")
                    return False
                elif response in ['i', 'info']:
                    self._show_tool_info(tool_name, tool_info)
                    continue
                elif response in ['s', 'stop']:
                    self.request_stop()
                    return False
                else:
                    print("❌ Please enter: y, n, i, or s")

            except (EOFError, KeyboardInterrupt):
                print("\n❌ Tool execution denied (interrupted)")
                return False

    def request_stop(self) -> str:
        """Request SAM to stop executing tools"""
        self.stop_requested = True
        logger.info("Stop requested - SAM will cease tool execution")
        print("🛑 Stop requested. Requesting SAM to stop proposing tool calls.")
        self.stop_message = (
            "<platform_message>"
            "TOOL EXECUTION FOR THIS REQUEST HAS BEEN TERMINATED BY THE USER. "
            "The user has requested to stop all pending tool calls. "
            "Do not attempt further tool calls related to the current request and stop execution. "
            "Provide a response without using any tools."
            "</platform_message>"
        )
        return self.stop_message

    def _show_tool_info(self, tool_name: str, tool_info: ToolInfo = None):
        """Show detailed tool information"""
        print(f"\n📋 DETAILED TOOL INFO: {tool_name}")
        print("-" * 60)

        if tool_info:
            print(f"📂 Category: {tool_info.category.value}")
            print(f"📄 Description: {tool_info.description}")
            print(f"📊 Usage count: {tool_info.usage_count}")
            print(f"🛡️  Requires approval: {'Yes' if tool_info.requires_approval else 'No'}")
            print(f"🔧 Parameters:")
            for param_name, param_info in tool_info.parameters.items():
                required = "required" if param_info.get('required', False) else "optional"
                param_type = param_info.get('type', 'unknown')
                default = param_info.get('default', 'N/A')
                print(f"  • {param_name} ({param_type}, {required})")
                if default != 'N/A':
                    print(f"    Default: {default}")
        else:
            print("ℹ️  No detailed information available")

        print("-" * 60)

    # ===== TOOL REGISTRATION =====
    def register_local_tool(self, function: Callable, category: ToolCategory = ToolCategory.UTILITY,
                            requires_approval: bool = False):
        """Register a local Python function as a tool"""
        func_name = function.__name__

        # Get function signature and documentation
        sig = inspect.signature(function)
        doc = inspect.getdoc(function) or f"Function {func_name}"

        # Build parameters dictionary
        parameters = {}
        for param_name, param in sig.parameters.items():
            param_info = {"description": f"Parameter {param_name}"}

            # Add type information if available
            if param.annotation != inspect.Parameter.empty:
                param_info["type"] = str(param.annotation.__name__)

            # Add default value if available
            if param.default != inspect.Parameter.empty:
                param_info["default"] = param.default

            parameters[param_name] = param_info

        try:
            # Store tool information
            self.tool_info[func_name] = ToolInfo(
                function=function,
                description=doc,
                parameters=parameters,
                category=category,
                requires_approval=requires_approval
            )

            # Store callable function
            self.local_tools[func_name] = {
                "function": function,
                "category": category.value,
                "requires_approval": requires_approval
            }

            # Add to category tracking
            if category not in self.tools_by_category:
                self.tools_by_category[category] = []
            self.tools_by_category[category].append(func_name)

            logger.info(f"Registered local tool: {func_name} ({category.value})")

        except Exception as e:
            logger.error(f"Failed to register tool {func_name}: {str(e)}")
            print(f"❌ Failed to register tool {func_name}: {str(e)}")

    def get_tools_by_category(self, category: ToolCategory) -> List[str]:
        """Get all tools in a specific category"""
        return self.tools_by_category[category].copy()

    def get_tool_categories(self) -> List[ToolCategory]:
        """Get all categories that have tools"""
        return list(self.tools_by_category.keys())

    # ===== TOOL EXECUTION WITH SAFETY =====
    async def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a tool with safety checks and approval system"""
        try:
            # Get tool info for safety checks
            tool_info = self.tool_info.get(tool_name)

            # Show raw tool call details BEFORE approval/execution
            print(f"\n🔧 RAW TOOL CALL:")
            print(f"Tool: {tool_name}")
            print(f"Arguments: {json.dumps(args, indent=2)}")
            print()  # Add blank line here

            # Check if approval is required
            requires_approval = (
                    self.safety_mode and
                    (not self.auto_approve or
                     (tool_info and tool_info.requires_approval))
            )

            if requires_approval:
                # Prompt for approval
                if not self._prompt_for_approval(tool_name, args, tool_info):
                    return f"❌ Tool execution denied by user: {tool_name}"
                print()  # Add blank line after approval

            # Update usage count
            if tool_info:
                tool_info.usage_count += 1

            # Execute local tool
            if tool_name in self.local_tools:
                print(f"\n🔧 Executing tool: {tool_name}")
                start_time = time.time()

                # Execute the tool
                result = self.local_tools[tool_name]["function"](**args)

                execution_time = time.time() - start_time
                print(f"✅ Tool completed in {execution_time:.3f}s")

                # Display raw results
                print()  # Add blank line before results
                print(f"\n📊 RAW RESULTS:")
                print("=" * 60)
                print(str(result))
                print("=" * 60)

                return str(result)

            elif tool_name in self.mcp_tools:
                # Execute MCP tool
                return await self._execute_mcp_tool(tool_name, args)
            else:
                return f"❌ Unknown tool: {tool_name}"

        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            logger.error(error_msg)
            print(f"\n❌ TOOL ERROR:")
            print("=" * 60)
            print(error_msg)
            print("=" * 60)
            return error_msg

    def _extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls from LLM response"""
        tool_calls = []
        seen_calls = set()

        # Multiple patterns to catch different JSON formatting
        patterns = [
            r'```json\s*(.*?)```',  # Standard JSON blocks
            r'```\s*(\{.*?"name".*?\})\s*```',  # General code blocks with JSON
            r'(\{[^{}]*"name"[^{}]*\})',  # Inline JSON objects
            r'```(?:python)?\s*(.*?)```',  # Python code blocks that might contain tool calls
            r'```(?:json)?\s*(.*?)```',  # Code blocks without explicit json
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    # Clean the match
                    cleaned = match.strip()

                    # Parse JSON
                    tool_call = json.loads(cleaned)

                    # Validate structure
                    if isinstance(tool_call, dict) and 'name' in tool_call:
                        if 'arguments' not in tool_call:
                            tool_call['arguments'] = {}

                        # Create a unique identifier for this tool call
                        call_signature = json.dumps(tool_call, sort_keys=True)

                        # Only add if we haven't seen this exact call before
                        if call_signature not in seen_calls:
                            tool_calls.append(tool_call)
                            seen_calls.add(call_signature)

                except json.JSONDecodeError:
                    continue

        return tool_calls

    async def run(self, user_input: str, max_iterations: int = 5,
                  verbose: bool = False) -> str:
        """Main execution loop with safety controls"""
        try:
            # Ensure MCP auto-connection happens on first run
            await self._ensure_mcp_auto_connect()

            # Handle safety commands first
            safety_commands = {
                'safety': self.get_safety_status,
                'safety on': lambda: self.set_safety_mode(True),
                'safety off': lambda: self.set_safety_mode(False),
                'auto on': lambda: self.set_auto_approve(True),
                'auto off': lambda: self.set_auto_approve(False),
            }

            user_input_lower = user_input.lower().strip()
            if user_input_lower in safety_commands:
                return safety_commands[user_input_lower]()

            # Reset stop flag for new requests
            self.stop_requested = False
            self.stop_message = ""

            # Add user message to conversation
            self.conversation_history.append({
                "role": "user",
                "content": user_input
            })

            last_response = ""
            tool_call_count = 0  # Track total tool calls to prevent runaway execution

            for iteration in range(max_iterations):
                if verbose:
                    print(f"\n🔄 Iteration {iteration + 1}/{max_iterations}")

                # Build available tools context
                tools_context = self._build_tools_context()

                # Prepare messages for LLM
                messages = [
                    {
                        "role": "system",
                        "content": f"""You are SAM (Secret Agent Man), an AI assistant with access to tools for various tasks.

    CRITICAL TOOL USAGE INSTRUCTIONS:
    - When you need to use a tool, respond with a JSON object in this EXACT format:
    {{"name": "tool_name", "arguments": {{"param1": "value1", "param2": "value2"}}}}
    - Put the JSON in a code block with ```json
    - Use tools whenever they would be helpful for the user's request
    - Always provide the tool call first, then explain what you're doing
    - For multiple tools, use separate JSON objects in separate code blocks
    - When you receive tool results from the user, respond naturally about what you found

    {tools_context}

    Current safety settings: {self.get_safety_status()}
    {self.stop_message}"""
                    }
                ]

                # Add conversation history
                messages.extend(self.conversation_history)

                if verbose:
                    print(f"📊 {self._get_context_status()}")

                # Generate response
                response = self.generate_chat_completion(messages)
                last_response = response

                if verbose:
                    print(f"\n🤖 Raw LLM Response:")
                    print(response)

                # Check for stop condition
                if self.stop_requested:
                    break

                # Extract and execute tool calls
                tool_calls = self._extract_tool_calls(response)

                if not tool_calls:
                    # No tools to execute, add response and finish
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    break

                # Add the assistant's tool-calling response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })

                # Execute tools and collect results
                tool_results = []
                for tool_call in tool_calls:
                    if self.stop_requested:
                        break

                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("arguments", {})

                    if verbose:
                        print(f"\n🔧 Executing: {tool_name}")

                    try:
                        result = await self._execute_tool(tool_name, tool_args)
                        tool_results.append(f"Tool '{tool_name}' executed successfully:\n{result}")
                        tool_call_count += 1

                        # Safety limit on tool calls
                        if tool_call_count >= 10:
                            tool_results.append("⚠️ Maximum tool call limit reached for this request")
                            break

                    except Exception as e:
                        error_msg = f"Tool '{tool_name}' failed: {str(e)}"
                        tool_results.append(error_msg)
                        logger.error(error_msg)

                # Feed tool results back to LLM as a "user" message (simulating human providing results)
                if tool_results:
                    tool_results_message = "Here are the results from the tool execution:\n\n" + "\n\n".join(
                        tool_results)

                    self.conversation_history.append({
                        "role": "user",
                        "content": tool_results_message
                    })

                    # Continue the loop so LLM can respond to the tool results naturally
                    continue
                else:
                    # No tools executed, we're done
                    break

            return last_response

        except Exception as e:
            logger.error(f"Error in run: {str(e)}")
            return f"❌ Error: {str(e)}"


    def _build_tools_context(self) -> str:
        """Build the available tools context for the LLM"""
        if not self.local_tools and not self.mcp_tools:
            return "\n\n<available_tools>\nNo tools available.\n</available_tools>"

        tools_list = []

        # Add local tools
        for tool_name, tool_data in self.local_tools.items():
            tool_info = self.tool_info.get(tool_name)
            if tool_info:
                tools_list.append(f"- {tool_name}: {tool_info.description} (Category: {tool_info.category.value})")
            else:
                tools_list.append(f"- {tool_name}: {tool_data.get('category', 'unknown')}")

        # Add MCP tools with server information
        for tool_name, (server_name, tool_info) in self.mcp_tools.items():
            description = tool_info.get('description', 'MCP tool')
            tools_list.append(f"- {tool_name}: {description} (MCP Server: {server_name})")

        tools_context = f"""
    
    <available_tools>
    Available tools ({len(tools_list)} total):
    {chr(10).join(tools_list)}
    
    Local tools: {len(self.local_tools)}
    MCP tools: {len(self.mcp_tools)} from {len(self.mcp_sessions)} servers
    </available_tools>"""

        return tools_context


    def _estimate_token_count(self, text: str) -> int:
        """Rough token count estimation"""
        return len(text) // 4


    def _get_context_status(self) -> str:
        """Get current context usage status"""
        total_tokens = 0
        message_breakdown = {"system": 0, "user": 0, "assistant": 0}

        for msg in self.conversation_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            tokens = self._estimate_token_count(content)
            total_tokens += tokens

            if role in message_breakdown:
                message_breakdown[role] += tokens

        percent_used = (total_tokens / self.context_limit) * 100

        warning = ""
        if percent_used > 90:
            warning = "⚠️ CRITICAL: Context nearly full!"
        elif percent_used > 75:
            warning = "⚠️ WARNING: Context usage high"

        return (
            f"CONTEXT STATUS: ~{total_tokens:,} tokens used (~{percent_used:.1f}% of {self.context_limit:,}). "
            f"Messages: {len(self.conversation_history)} "
            f"Tools: {len(self.local_tools)} local, {len(self.mcp_tools)} MCP. "
            f"{warning}"
        )


    def list_tools(self) -> Dict[str, Any]:
        """List all available tools"""
        local_tools = {}
        mcp_tools = {}

        # Local tools
        for tool_name, tool_info in self.tool_info.items():
            local_tools[tool_name] = {
                "description": tool_info.description,
                "category": tool_info.category.value,
                "requires_approval": tool_info.requires_approval,
                "usage_count": tool_info.usage_count,
                "parameters": tool_info.parameters
            }

        # MCP tools
        for tool_name, (server_name, tool_info) in self.mcp_tools.items():
            mcp_tools[tool_name] = {
                "description": tool_info.get('description', ''),
                "server": server_name,
                "input_schema": tool_info.get('input_schema', {}),
                "category": "mcp"
            }

        return {
            "local_tools": local_tools,
            "mcp_tools": mcp_tools,
            "mcp_servers": self.list_mcp_servers(),
            "total_count": len(local_tools) + len(mcp_tools)
        }


# ===== CLI INTERFACE =====
def main():
    """CLI interface for SAM Agent"""
    import argparse

    # Add argument parsing
    parser = argparse.ArgumentParser(description="SAM Agent - Secret Agent Man")
    parser.add_argument("--api", action="store_true", help="Run as HTTP API server")
    parser.add_argument("--api-host", type=str, default="0.0.0.0", help="API server host")
    parser.add_argument("--api-port", type=int, default=8888, help="API server port")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # If --api flag is provided, run API server instead of interactive mode
    if args.api:
        if FASTAPI_AVAILABLE:
            print("🌐 Starting SAM API Server...")

            # Initialize SAM
            sam = SAMAgent(safety_mode=True, auto_approve=True)  # Enable auto_approve for API mode

            # Auto-load all plugins from the plugins directory
            load_all_plugins(sam)

            # Start API server
            server = SAMAPIServer(sam, host=args.api_host, port=args.api_port)
            print(f"🚀 Starting server on http://{server.host}:{server.port}")
            print("📚 API docs available at http://localhost:8888/docs")
            server.run()
            return
        else:
            print("❌ FastAPI not available. Install with: pip install fastapi uvicorn")
            return

    # If no --api flag, continue with original interactive mode
    print("🤖 Starting SAM initialization...")

    # Initialize SAM - will auto-load config.json
    sam = SAMAgent()

    # Auto-load all plugins from the plugins directory
    load_all_plugins(sam)

    # Test API connection and show model info
    try:
        test_response = sam.generate_chat_completion([
            {"role": "user", "content": "Hello, are you working?"}
        ])
        if "error" not in test_response.lower():
            print("✅ API connection test successful!")
            print(f"📊 Using model: {sam.model_name}")
            print(f"📊 Context limit: {sam.context_limit:,} tokens")
        else:
            print(f"❌ API test failed: {test_response}")
    except Exception as e:
        print(f"❌ API connection failed: {e}")

    # Display capabilities
    tools_info = sam.list_tools()
    print(f"\n=== 🤖 SAM CAPABILITIES ===")
    print(f"🤖 Model: {sam.model_name}")
    print(f"🧠 Context: {sam.context_limit:,} tokens")
    print(f"🔧 Local tools: {len(sam.local_tools)}")
    print(f"🌐 MCP tools: {len(sam.mcp_tools)}")
    print(f"📡 MCP servers: {len(sam.mcp_sessions)}")
    print(f"🔌 Plugins: {len(sam.plugin_manager.plugins)}")
    print(f"🛡️ Safety mode: {'ON' if sam.safety_mode else 'OFF'}")
    print(f"🤖 Auto-approve: {'ON' if sam.auto_approve else 'OFF'}")

    # Interactive loop
    print(f"\n=== 🤖 SAM Agent Interactive Mode ===")
    print("Type 'exit' to quit, 'tools' to list available tools")
    print("Commands: 'debug' (toggle debug), 'reset' (clear history), 'tools' (list tools)")
    print("Providers: 'provider claude/lmstudio', 'providers' (list available)")
    print("Safety: 'safety on/off', 'auto on/off', 'safety' (status)")
    print("MCP Commands: 'mcp servers', 'mcp connect <server>', 'mcp disconnect <server>'")

    debug_mode = False

    while True:
        try:
            # Add extra spacing for better readability
            print()  # Empty line before prompt
            user_input = input(f"💬 You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("👋 Goodbye!")
                # Clean up MCP connections
                asyncio.run(sam.disconnect_mcp_servers())
                break

            # Handle safety commands
            elif user_input.lower().startswith('safety') or user_input.lower().startswith('auto'):
                safety_commands = {
                    'safety': sam.get_safety_status,
                    'safety on': lambda: sam.set_safety_mode(True),
                    'safety off': lambda: sam.set_safety_mode(False),
                    'auto on': lambda: sam.set_auto_approve(True),
                    'auto off': lambda: sam.set_auto_approve(False),
                }

                if user_input.lower() in safety_commands:
                    result = safety_commands[user_input.lower()]()
                    print(result)
                    continue  # This is crucial - it prevents the "SAM is thinking" code from running


            elif user_input.lower().startswith('provider '):
                provider_name = user_input.split(' ', 1)[1].strip()
                result = sam.switch_provider(provider_name)
                print(result)
                continue
            elif user_input.lower() == 'providers':
                result = sam.get_current_provider()
                print(result)
                continue

            elif user_input.lower() == 'debug':
                debug_mode = not debug_mode
                print(f"🐛 Debug mode: {'ON' if debug_mode else 'OFF'}")
                continue

            elif user_input.lower() == 'reset':
                sam.conversation_history = []
                print("🔄 Conversation history cleared")
                continue

            elif user_input.lower() == 'tools':
                # List tools with current usage counts
                current_tools = sam.list_tools()
                total_count = len(current_tools.get('local_tools', {})) + len(current_tools.get('mcp_tools', {}))
                print(f"\n🔧 Available Tools ({total_count} total):")

                # Local tools
                for name, info in current_tools.get('local_tools', {}).items():
                    approval = "🛡️" if info.get('requires_approval', False) else "✅"
                    usage = info.get('usage_count', 0)
                    usage_text = f" (used {usage}x)" if usage > 0 else ""
                    print(
                        f"  {approval} {name}: {info.get('description', 'No description')} ({info.get('category', 'unknown')}){usage_text}")

                # MCP tools
                if current_tools.get('mcp_tools'):
                    print(f"\n🌐 MCP Tools:")
                    for name, info in current_tools.get('mcp_tools', {}).items():
                        server = info.get('server', 'unknown')
                        description = info.get('description', 'No description')
                        print(f"  🌐 {name}: {description} (Server: {server})")
                continue

            # Handle MCP-specific commands
            elif user_input.lower().startswith('mcp '):
                mcp_command = user_input[4:].strip()

                if mcp_command == 'servers':
                    servers = sam.list_mcp_servers()
                    if servers:
                        print(f"\n🌐 Connected MCP Servers ({len(servers)}):")
                        for name, info in servers.items():
                            print(f"  📡 {name}: {info['tool_count']} tools")
                            for tool in info['tools']:
                                print(f"    - {tool}")
                    else:
                        print("🌐 No MCP servers connected")
                    continue

                elif mcp_command.startswith('connect '):
                    server_name = mcp_command[8:].strip()
                    if hasattr(sam.config, 'mcp') and sam.config.mcp.servers and server_name in sam.config.mcp.servers:
                        server_config = sam.config.mcp.servers[server_name]
                        result = asyncio.run(sam.connect_to_mcp_server(
                            server_name=server_name,
                            server_type=server_config.get('type', 'stdio'),
                            server_path_or_url=server_config.get('path', ''),
                            headers=server_config.get('headers', {})
                        ))
                        if result:
                            print(f"✅ Connected to MCP server: {server_name}")
                        else:
                            print(f"❌ Failed to connect to MCP server: {server_name}")
                    else:
                        print(f"❌ Server '{server_name}' not found in configuration")
                    continue

                elif mcp_command.startswith('disconnect '):
                    server_name = mcp_command[11:].strip()
                    if server_name in sam.mcp_sessions:
                        session = sam.mcp_sessions[server_name]
                        asyncio.run(session.close())
                        del sam.mcp_sessions[server_name]
                        # Remove tools from this server
                        tools_to_remove = [tool for tool, (srv, _) in sam.mcp_tools.items() if srv == server_name]
                        for tool in tools_to_remove:
                            del sam.mcp_tools[tool]
                        print(f"✅ Disconnected from MCP server: {server_name}")
                    else:
                        print(f"❌ Server '{server_name}' is not connected")
                    continue

            print("🤖 SAM is thinking...")

            # Run SAM with the user input (async)
            response = asyncio.run(sam.run(user_input, verbose=debug_mode))

            # Add spacing and format the response properly
            print(f"\n🤖 SAM: {response}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            asyncio.run(sam.disconnect_mcp_servers())
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            if debug_mode:
                traceback.print_exc()


# ===== API SERVER (if FastAPI available) =====
if FASTAPI_AVAILABLE:
    class SAMAPIServer:
        """FastAPI server wrapper for SAM Agent"""

        def __init__(self, sam_agent: SAMAgent, host: str = "0.0.0.0", port: int = 8000):
            self.sam_agent = sam_agent
            self.host = host
            self.port = port
            self.app = FastAPI(
                title="SAM Agent API",
                description="Secret Agent Man API with Safety Controls and MCP Support",
                version="1.0.0"
            )
            self.start_time = time.time()
            self._setup_routes()

        def _setup_routes(self):
            """Setup FastAPI routes"""

            @self.app.post("/query", response_model=QueryResponse)
            async def process_query(request: QueryRequest):
                try:
                    # Handle auto-approve override
                    original_auto_approve = self.sam_agent.auto_approve
                    if request.auto_approve is not None:
                        self.sam_agent.auto_approve = request.auto_approve

                    # Process the query
                    response = await self.sam_agent.run(
                        request.message,
                        max_iterations=request.max_iterations,
                        verbose=request.verbose
                    )

                    # Restore original setting
                    self.sam_agent.auto_approve = original_auto_approve

                    return QueryResponse(
                        response=response,
                        session_id=request.session_id or "default",
                        status="success",
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                    )

                except Exception as e:
                    logger.error(f"API Error: {str(e)}")
                    return QueryResponse(
                        response="",
                        session_id=request.session_id or "default",
                        status="error",
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        error=str(e)
                    )

            @self.app.get("/health", response_model=HealthResponse)
            async def health_check():
                return HealthResponse(
                    status="healthy",
                    model=self.sam_agent.model_name,
                    tools_count=len(self.sam_agent.local_tools) + len(self.sam_agent.mcp_tools),
                    plugins_count=len(self.sam_agent.plugin_manager.plugins),
                    uptime_seconds=time.time() - self.start_time
                )

            @self.app.get("/tools")
            async def list_tools():
                return self.sam_agent.list_tools()

            @self.app.get("/mcp/servers")
            async def list_mcp_servers():
                return self.sam_agent.list_mcp_servers()

            @self.app.post("/mcp/connect")
            async def connect_mcp_server(server_name: str, server_type: str, server_url: str):
                success = await self.sam_agent.connect_to_mcp_server(server_name, server_type, server_url)
                return {"success": success, "server": server_name}

            @self.app.post("/execute-tool", response_model=ToolExecutionResponse)
            async def execute_tool_direct(request: ToolExecutionRequest):
                start_time = time.time()
                try:
                    result = await self.sam_agent._execute_tool(request.tool_name, request.arguments)
                    execution_time = time.time() - start_time

                    return ToolExecutionResponse(
                        success=True,
                        result=result,
                        execution_time=execution_time,
                        tool_name=request.tool_name
                    )

                except Exception as e:
                    execution_time = time.time() - start_time
                    return ToolExecutionResponse(
                        success=False,
                        error=str(e),
                        execution_time=execution_time,
                        tool_name=request.tool_name
                    )

            @self.app.get("/safety")
            async def get_safety_status():
                return self.sam_agent.get_detailed_safety_status()

            @self.app.post("/safety/mode")
            async def set_safety_mode(enabled: bool):
                result = self.sam_agent.set_safety_mode(enabled)
                return {"message": result, "safety_mode": self.sam_agent.safety_mode}

            @self.app.post("/safety/auto-approve")
            async def set_auto_approve(enabled: bool):
                result = self.sam_agent.set_auto_approve(enabled)
                return {"message": result, "auto_approve": self.sam_agent.auto_approve}

        def run(self):
            """Run the API server"""
            import uvicorn
            uvicorn.run(self.app, host=self.host, port=self.port)


    def run_api_server():
        """Start SAM as an API server"""
        print("🌐 Starting SAM API Server...")

        # Initialize SAM
        sam = SAMAgent(safety_mode=True, auto_approve=False)

        # Load core tools
        try:
            from plugins.core_tools import CoreToolsPlugin
            core_plugin = CoreToolsPlugin()
            core_plugin.register_tools(sam)
            print("✅ Core tools plugin loaded!")
        except ImportError as e:
            print(f"⚠️  Could not load core tools: {e}")

        # Start API server
        server = SAMAPIServer(sam)
        print(f"🚀 Starting server on http://{server.host}:{server.port}")
        print("📚 API docs available at http://localhost:8000/docs")
        server.run()

else:
    def run_api_server():
        print("❌ FastAPI not available. Install with: pip install fastapi uvicorn")

if __name__ == "__main__":
    main()