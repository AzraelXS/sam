#!/usr/bin/env python3
"""
SAM: Semi-Autonomous Model - AI Agent with MCP support.
Core agent module - refactored for modularity and clean architecture.
"""

import argparse
import asyncio
from datetime import datetime
import json
import logging
import os
import sys
import platform
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum
from collections import defaultdict
import inspect
import importlib.util

import dotenv
import requests
import urllib3

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import threading

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Project setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SAM")


# ===== API MODELS =====
class QueryRequest(BaseModel):
    message: str
    session_id: str = "default"
    auto_approve: bool = True
    max_iterations: int = 10
    verbose: bool = False


class QueryResponse(BaseModel):
    response: str
    session_id: str
    status: str = "success"
    timestamp: str
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model: str
    tools_count: int
    plugins_count: int
    uptime_seconds: float


class ToolExecutionRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    session_id: str = "default"


class ToolExecutionResponse(BaseModel):
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    execution_time: float
    tool_name: str


# ===== TOOL CLASSIFICATION SYSTEM =====
class ToolCategory(Enum):
    FILESYSTEM = "filesystem"
    WEB_BROWSING = "web"
    COMMUNICATION = "communication"
    DEVELOPMENT = "development"
    SYSTEM = "system"
    UTILITY = "utility"
    AI_SERVICES = "ai_services"


class ToolInfo:
    def __init__(self, function: Callable, description: str, parameters: Dict,
                 category: ToolCategory, requires_approval: bool = False):
        self.function = function
        self.description = description
        self.parameters = parameters
        self.category = category
        self.requires_approval = requires_approval
        self.usage_count = 0


# ===== PLUGIN SYSTEM =====
class SAMPlugin:
    """Base class for SAM plugins"""

    def __init__(self, name: str, version: str = "1.0.0", description: str = ""):
        self.name = name
        self.version = version
        self.description = description
        self.enabled = True

    def register_tools(self, agent: 'SAMAgent'):
        """Register plugin tools with the agent"""
        pass

    def on_load(self, agent: 'SAMAgent'):
        """Called when plugin is loaded"""
        pass

    def on_unload(self, agent: 'SAMAgent'):
        """Called when plugin is unloaded"""
        pass


# ===== MAIN SAM AGENT CLASS =====
class SAMAgent:
    def __init__(self,
                 model_name: str = "sam-1",
                 context_limit: int = 128000,
                 temperature: float = 0.3,
                 base_url: str = "http://localhost:1234/v1",
                 api_key: str = "lm-studio"):

        # Core configuration
        self.model_name = model_name
        self.context_limit = context_limit
        self.temperature = temperature
        self.base_url = base_url
        self.api_key = api_key

        # State management
        self.conversation_history = []
        self.local_tools = {}
        self.mcp_tools = {}
        self.tool_info = {}
        self.mcp_sessions = {}
        self.auto_approve = True
        self.stop_requested = False
        self.stop_message = ""

        # Statistics
        self._refusal_stats = {
            "total_queries": 0,
            "refusals_detected": 0,
            "successful_retries": 0
        }

        # Plugin system
        self.plugin_manager = PluginManager(self)

        # System prompt
        self.system_prompt = """You are SAM (Semi-Autonomous Model), a highly capable AI agent with access to various tools and capabilities.

Core Principles:
- Be helpful, accurate, and efficient
- Use tools when necessary to accomplish tasks
- Provide clear, concise responses
- Ask for clarification when needed
- Respect user preferences and boundaries

When using tools:
1. Consider if the tool is necessary for the task
2. Use appropriate parameters
3. Interpret results accurately
4. Explain what you did and why

You have access to various categories of tools including filesystem operations, web browsing, development tools, and more. Use them wisely to assist the user effectively."""

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

        logger.info(f"Registered local tool: {func_name} ({category.value})")

    def generate_chat_completion(self, messages: List[Dict]) -> str:
        """Generate chat completion using LM Studio API"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": -1,
                "stream": False
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=300
            )

            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                else:
                    raise Exception("No choices in API response")
            else:
                raise Exception(f"API request failed: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Chat completion error: {str(e)}")
            raise

    def _format_tools_for_prompt(self) -> str:
        """Format available tools for the system prompt"""
        if not self.local_tools and not self.mcp_tools:
            return "No tools currently available."

        tools_text = "<available_tools>\n"
        tools_text += "To use a tool, respond with a JSON code block like:\n"
        tools_text += "```json\n{\n  \"name\": \"tool_name\",\n  \"arguments\": {\n    \"param1\": \"value1\"\n  }\n}\n```\n\n"

        # Group tools by category
        categories = defaultdict(list)

        for tool_name, tool_data in self.local_tools.items():
            if tool_name in self.tool_info:
                info = self.tool_info[tool_name]
                categories[info.category.value].append({
                    'name': tool_name,
                    'description': info.description,
                    'parameters': info.parameters
                })

        # Format by category
        for category, tools in categories.items():
            tools_text += f"\n=== {category.upper()} TOOLS ===\n"
            for tool in tools:
                tools_text += f"Tool: {tool['name']}\n"
                tools_text += f"Description: {tool['description']}\n"
                if tool['parameters']:
                    tools_text += f"Parameters: {', '.join(tool['parameters'].keys())}\n"
                tools_text += "\n"

        tools_text += "</available_tools>"
        return tools_text

    async def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a tool with the given arguments"""
        try:
            # Check if tool exists in local tools
            if tool_name in self.local_tools:
                tool_info = self.tool_info.get(tool_name)

                # Check approval if required
                if tool_info and tool_info.requires_approval and not self.auto_approve:
                    approval = input(f"🔐 Execute {tool_name} with args {args}? (y/n): ")
                    if approval.lower() != 'y':
                        return f"❌ Tool execution denied by user: {tool_name}"

                # Execute the tool
                result = self.local_tools[tool_name]["function"](**args)
                return str(result)

            elif tool_name in self.mcp_tools:
                # Execute MCP tool (implementation depends on your MCP setup)
                return await self._execute_mcp_tool(tool_name, args)
            else:
                return f"❌ Unknown tool: {tool_name}"

        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def _execute_mcp_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute an MCP tool (placeholder for MCP implementation)"""
        # This would contain your MCP tool execution logic
        return f"MCP tool {tool_name} executed with args: {args}"

    def _extract_tool_calls(self, text: str) -> List[Dict]:
        """Extract tool calls from LLM response - supports multiple JSON formats"""
        tool_calls = []

        # Patterns for finding JSON tool calls in various formats
        patterns = [
            r'```json\s*(.*?)\s*```',  # Standard code blocks
            r'```(?:json)?\s*(.*?)```',  # Code blocks without newlines
            r'```(.*?)```',  # Any code blocks
        ]

        for pattern in patterns:
            json_blocks = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for block in json_blocks:
                try:
                    cleaned = block.strip()

                    # FIX: Convert Python booleans to JSON booleans
                    cleaned = cleaned.replace('True', 'true').replace('False', 'false').replace('None', 'null')

                    parsed = json.loads(cleaned)
                    if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                        if isinstance(parsed.get("arguments"), dict):
                            if parsed not in tool_calls:  # Avoid duplicates
                                tool_calls.append(parsed)
                except json.JSONDecodeError as e:
                    # Debug output to see what's failing
                    logger.debug(f"JSON decode error for block: {block[:100]}... Error: {e}")
                    continue

        return tool_calls

    async def run(self, user_query: str, max_iterations: int = 10, verbose: bool = True) -> str:
        """Main execution loop for processing user queries"""
        self._refusal_stats["total_queries"] += 1

        # Build system message with tools
        tools_description = self._format_tools_for_prompt()
        enhanced_system_prompt = f"{self.system_prompt}\n\n{tools_description}"

        # Create working conversation
        if not self.conversation_history:
            working_conversation = [
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": user_query}
            ]
            self.conversation_history = [
                {"role": "system", "content": self.system_prompt}
            ]
        else:
            working_conversation = self.conversation_history.copy()
            working_conversation[0] = {"role": "system", "content": enhanced_system_prompt}
            working_conversation.append({"role": "user", "content": user_query})
            self.conversation_history[0] = {"role": "system", "content": self.system_prompt}

        iteration = 0
        last_response = ""

        while iteration < max_iterations:
            iteration += 1
            if verbose:
                logger.info(f"--- Iteration {iteration} ---")

            try:
                # Generate response
                response = self.generate_chat_completion(working_conversation)
                last_response = str(response)

                # Add assistant response to working conversation
                working_conversation.append({"role": "assistant", "content": response})

                # Parse for tool calls
                tool_calls = self._extract_tool_calls(response)

                if tool_calls:
                    # Execute each tool call
                    for tool_call in tool_calls:
                        tool_name = tool_call.get("name", "")
                        tool_args = tool_call.get("arguments", {})

                        if verbose:
                            print(f"🔧 Using tool: {tool_name}")

                        result = await self._execute_tool(tool_name, tool_args)

                        # Add tool result to conversation
                        tool_response = f"Tool {tool_name}: {result}"
                        working_conversation.append({"role": "user", "content": tool_response})

                        if verbose:
                            print(f"🔧 {tool_response}")
                else:
                    # No more tool calls - execution complete
                    break

            except Exception as e:
                error_msg = f"Error during iteration {iteration}: {str(e)}"
                logger.error(error_msg)
                if verbose:
                    print(error_msg)
                break

        # Update permanent conversation history
        self.conversation_history.append({"role": "user", "content": user_query})

        # Add assistant responses from working conversation
        for msg in working_conversation[2:]:  # Skip system and initial user message
            if msg["role"] == "assistant":
                self.conversation_history.append(msg)
            elif msg["role"] == "user" and msg["content"].startswith("Tool "):
                self.conversation_history.append(msg)

        return last_response

    def load_plugin(self, plugin_path: str) -> bool:
        """Load a plugin from file"""
        return self.plugin_manager.load_plugin_from_file(plugin_path)

    def list_tools(self) -> Dict:
        """List all available tools"""
        tools = {}

        for name, info in self.tool_info.items():
            tools[name] = {
                "category": info.category.value,
                "description": info.description,
                "requires_approval": info.requires_approval,
                "usage_count": info.usage_count
            }

        return tools


# ===== PLUGIN MANAGER =====
class PluginManager:
    def __init__(self, sam_agent: SAMAgent):
        self.agent = sam_agent
        self.plugins = {}
        self.plugin_directories = ['./plugins', './sam_plugins']

    def load_plugin_from_file(self, plugin_path: str) -> bool:
        """Load a plugin from a Python file"""
        try:
            plugin_path = Path(plugin_path)
            if not plugin_path.exists():
                logger.error(f"Plugin file not found: {plugin_path}")
                return False

            # Load the module
            spec = importlib.util.spec_from_file_location(plugin_path.stem, plugin_path)
            if spec is None or spec.loader is None:
                logger.error(f"Could not load plugin spec: {plugin_path}")
                return False

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find plugin classes
            plugin_classes = []
            for name in dir(module):
                obj = getattr(module, name)
                if (inspect.isclass(obj) and
                        obj != SAMPlugin and
                        obj.__name__ != 'SAMPlugin' and  # ← Add this line
                        hasattr(obj, '__bases__') and
                        any(base.__name__ == 'SAMPlugin' for base in obj.__mro__)):
                    plugin_classes.append(obj)

            if not plugin_classes:
                logger.error(f"No plugin classes found in {plugin_path}")
                return False

            # Initialize and register plugins
            for plugin_class in plugin_classes:
                # SAMPlugin requires a name parameter, but plugin classes handle it in their own __init__
                try:
                    plugin = plugin_class()  # This should work with your CoreToolsPlugin
                    self.plugins[plugin.name] = plugin

                    # Register tools
                    plugin.register_tools(self.agent)
                    plugin.on_load(self.agent)

                    logger.info(f"Successfully loaded plugin: {plugin.name} v{plugin.version}")
                except Exception as e:
                    logger.error(f"Error initializing plugin {plugin_class}: {e}")
                    continue

            return True

        except Exception as e:
            logger.error(f"Error loading plugin {plugin_path}: {str(e)}")
            return False


# ===== API SERVER =====
class SAMAPIServer:
    def __init__(self, sam_agent: SAMAgent):
        self.sam_agent = sam_agent
        self.app = FastAPI(
            title="SAM AI Agent API",
            description="HTTP API for SAM AI Agent",
            version="2.0.0"
        )

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.start_time = time.time()
        self.websocket_connections = set()
        self.setup_routes()

    def setup_routes(self):
        @self.app.post("/query", response_model=QueryResponse)
        async def process_query(request: QueryRequest):
            try:
                logger.info(f"API Query from session {request.session_id}: {request.message[:100]}...")

                # Store original auto_approve setting
                original_auto_approve = self.sam_agent.auto_approve
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
                    session_id=request.session_id,
                    status="success",
                    timestamp=datetime.now().isoformat()
                )

            except Exception as e:
                logger.error(f"API Error: {str(e)}")
                return QueryResponse(
                    response="",
                    session_id=request.session_id,
                    status="error",
                    timestamp=datetime.now().isoformat(),
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


async def run_api_server(sam_agent: SAMAgent, host: str = "127.0.0.1", port: int = 8888):
    """Run SAM API server"""
    api_server = SAMAPIServer(sam_agent)
    sam_agent._api_server = api_server

    config = uvicorn.Config(
        api_server.app,
        host=host,
        port=port,
        log_level="info",
        access_log=False
    )

    server = uvicorn.Server(config)
    logger.info(f"🌐 SAM API server starting at http://{host}:{port}")
    logger.info(f"🔗 Health check: http://{host}:{port}/health")
    logger.info(f"📋 API docs: http://{host}:{port}/docs")

    await server.serve()


# ===== MAIN FUNCTION =====
async def main():
    print("🤖 Starting SAM initialization...")

    parser = argparse.ArgumentParser(description="SAM: Semi-Autonomous Model AI Agent")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--api", action="store_true", help="Run as HTTP API server")
    parser.add_argument("--api-host", type=str, default="127.0.0.1", help="API server host")
    parser.add_argument("--api-port", type=int, default=8888, help="API server port")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize SAM agent
        sam = SAMAgent()

        # Load core tools plugin
        try:
            if os.path.exists("plugins/core_tools.py"):
                result = sam.load_plugin("plugins/core_tools.py")
                if result:
                    print("✅ Core tools plugin loaded successfully!")
                else:
                    print("❌ Core tools plugin loading failed!")
            else:
                print("❌ plugins/core_tools.py file not found!")
        except Exception as e:
            logger.error(f"Exception loading core tools: {str(e)}")
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")

        # Test API connection
        try:
            sam.generate_chat_completion([
                {"role": "system", "content": "You are SAM, a helpful AI assistant."},
                {"role": "user", "content": "Say hello!"}
            ])
            print("✅ API connection test successful!")
        except Exception as e:
            print(f"❌ API connection test failed: {str(e)}")

        # Show capabilities
        print("\n=== 🤖 SAM CAPABILITIES ===")
        print(f"🤖 Model: {sam.model_name}")
        print(f"🧠 Context: {sam.context_limit:,} tokens")
        print(f"🔧 Local tools: {len(sam.local_tools)}")
        print(f"🌐 MCP tools: {len(sam.mcp_tools)}")
        print(f"🔌 Plugins: {len(sam.plugin_manager.plugins)}")

        # Check if we should run in API mode
        if args.api:
            try:
                await run_api_server(sam, args.api_host, args.api_port)
            except KeyboardInterrupt:
                print("\n🌐 API server stopped by user")
                return

        # Interactive mode
        try:
            print("\n=== 🤖 SAM Agent Interactive Mode ===")
            print("Type 'exit' to quit, 'tools' to list available tools")
            print("Commands: 'debug' (toggle debug), 'reset' (clear history), 'tools' (list tools)")

            debug_mode = args.debug

            while True:
                user_input = input("\n💬 You: ")

                if user_input.lower() in ('exit', 'quit'):
                    print("👋 Goodbye!")
                    break

                if user_input.lower() == 'debug':
                    debug_mode = not debug_mode
                    print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
                    continue

                if user_input.lower() == 'reset':
                    sam.conversation_history = []
                    print("🔄 Conversation history cleared")
                    continue

                if user_input.lower() == 'tools':
                    tools = sam.list_tools()
                    print("\n🔧 Available Tools:")
                    for name, info in tools.items():
                        print(f"  - {name} ({info['category']}): {info['description']}")
                    continue

                # Process user query
                print("\n🤖 SAM is thinking...")
                result = await sam.run(user_input, verbose=args.verbose or debug_mode)

                # Don't double-print if verbose is on
                if not (args.verbose or debug_mode):
                    print(f"\n🤖 SAM: {result}")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        logger.exception("Fatal error in main")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())