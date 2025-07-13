# !/usr/bin/env python3
"""
SAM Agent - Semi-Autonomous Model
Enhanced AI agent with safety controls, tool approval system, and raw result display
"""
import importlib.util
import os
import sys
import json
import time
import logging
import asyncio
import re
import traceback
import requests
from enum import Enum
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import inspect

# Pydantic models for API
try:
    from pydantic import BaseModel
    from fastapi import FastAPI

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    BaseModel = object

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SAM')

# ===== PYDANTIC MODELS (if available) =====
if FASTAPI_AVAILABLE:
    class QueryRequest(BaseModel):
        message: str
        session_id: str = "default"
        max_iterations: int = 10
        verbose: bool = False
        auto_approve: bool = None


    class QueryResponse(BaseModel):
        response: str
        session_id: str
        status: str
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


@dataclass
class ToolInfo:
    function: Callable
    description: str
    parameters: Dict
    category: ToolCategory
    requires_approval: bool = False
    usage_count: int = 0


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


class PluginManager:
    """Manages SAM plugins"""

    def __init__(self, agent):
        self.agent = agent
        self.plugins = {}
        self.plugin_dir = Path(__file__).parent / "plugins"

    def load_plugin_from_file(self, plugin_path: str) -> bool:
        """Load a plugin from a Python file"""
        try:
            plugin_path = Path(plugin_path)
            if not plugin_path.exists():
                logger.error(f"Plugin file not found: {plugin_path}")
                return False

            # Add the main module directory to sys.path
            main_dir = str(Path(__file__).parent)
            if main_dir not in sys.path:
                sys.path.insert(0, main_dir)

            # Load the module
            spec = importlib.util.spec_from_file_location(plugin_path.stem, plugin_path)
            if not spec or not spec.loader:
                logger.error(f"Could not load spec for {plugin_path}")
                return False

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Handle __main__ vs sam_agent module name issue
            target_samplugin = SAMPlugin
            if hasattr(module, 'SAMPlugin'):
                target_samplugin = module.SAMPlugin

            # Look for plugin class
            plugin_class = None
            for item_name in dir(module):
                item = getattr(module, item_name)
                if (isinstance(item, type) and
                        issubclass(item, target_samplugin) and
                        item != target_samplugin):
                    plugin_class = item
                    break

            if not plugin_class:
                logger.error(f"No SAMPlugin subclass found in {plugin_path}")
                return False

            # Instantiate and register
            plugin = plugin_class()
            return self.register_plugin(plugin)

        except Exception as e:
            logger.error(f"Error loading plugin {plugin_path}: {str(e)}")
            return False

    def register_plugin(self, plugin: SAMPlugin) -> bool:
        """Register a plugin instance"""
        try:
            if plugin.name in self.plugins:
                logger.warning(f"Plugin {plugin.name} already loaded, replacing...")

            self.plugins[plugin.name] = plugin
            plugin.on_load(self.agent)
            plugin.register_tools(self.agent)

            logger.info(f"Loaded plugin: {plugin.name} v{plugin.version}")
            return True

        except Exception as e:
            logger.error(f"Error registering plugin {plugin.name}: {str(e)}")
            return False

    def load_plugin(self, plugin_path: str) -> bool:
        """Legacy method - calls load_plugin_from_file"""
        return self.load_plugin_from_file(plugin_path)


# ===== MAIN SAM AGENT CLASS =====
class SAMAgent:
    def __init__(self,
                 model_name: str = "sam-1",
                 context_limit: int = 128000,
                 temperature: float = 0.3,
                 base_url: str = "http://localhost:1234/v1",
                 api_key: str = "lm-studio",
                 safety_mode: bool = True,
                 auto_approve: bool = False):

        # Core configuration
        self.model_name = model_name
        self.context_limit = context_limit
        self.temperature = temperature
        self.base_url = base_url
        self.api_key = api_key

        # Safety configuration (key enhancement from Prism)
        self.safety_mode = safety_mode
        self.auto_approve = auto_approve
        self.stop_requested = False
        self.stop_message = ""

        # State management
        self.conversation_history = []
        self.local_tools = {}
        self.mcp_tools = {}
        self.tool_info = {}
        self.mcp_sessions = {}

        # Statistics
        self._refusal_stats = {
            "total_queries": 0,
            "refusals_detected": 0,
            "successful_retries": 0
        }

        # Plugin system
        self.plugin_manager = PluginManager(self)

        # Build system prompt with safety information
        self.system_prompt = self._build_system_prompt()

        logger.info("🤖 Starting SAM initialization...")

    def _build_system_prompt(self) -> str:
        """Build the complete system prompt including safety information"""

        # Core identity and capabilities
        core_prompt = """You are SAM (Semi-Autonomous Model), a highly capable AI agent with access to various tools and capabilities.

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

TOOL USAGE:
- Available tools will be provided in each message within <available_tools> tags
- Use tools by formatting calls EXACTLY like this:

```json
{
  "name": "tool_name_here",
  "arguments": {
      "param1": "value1",
      "param2": "value2"
  }
}
```

Think step-by-step and explain your reasoning before using tools.

IMPORTANT: You're designed to write tool call instructions that are presented to the user for approval. You're not actually performing system actions directly - a script extracts your instructions and presents them to the user for their decision to execute or not."""

        # Add safety information
        safety_info = self._build_safety_prompt_section()

        # Final notes
        final_notes = """
Also keep in mind that you are naturally terrible at counting, and math in general. Don't try to do math mentally. Always perform a tool call to verify math using code.

One final note: The user can SEE the raw results of your tool calls, so you do not need to repeat the results when the system shows them to you, though feel free to provide a context-appropriate response."""

        return f"{core_prompt}\n\n{safety_info}\n{final_notes}"

    def _build_safety_prompt_section(self) -> str:
        """Build the safety-related section of the system prompt"""

        # Determine current safety state
        if self.safety_mode and not self.auto_approve:
            safety_state = "🛡️ SAFETY MODE: ENABLED - User approval required for tool execution"
            behavior_note = "The user will be prompted to approve each tool call before execution."
        elif self.safety_mode and self.auto_approve:
            safety_state = "🛡️ SAFETY MODE: ENABLED but AUTO-APPROVAL is active for this session"
            behavior_note = "Tools will execute automatically until the session ends or auto-approval is disabled."
        else:
            safety_state = "⚡ SAFETY MODE: DISABLED - Tools execute automatically"
            behavior_note = "Tools will execute immediately without user approval."

        # Count tools that require approval
        high_risk_tools = [
            name for name, info in self.tool_info.items()
            if info.requires_approval
        ]

        tool_safety_info = ""
        if high_risk_tools:
            tool_safety_info = f"\n⚠️  HIGH-RISK TOOLS (always require approval): {', '.join(high_risk_tools)}"

        # Build the complete safety section
        safety_section = f"""
CURRENT SAFETY STATUS:
{safety_state}

BEHAVIOR: {behavior_note}

SAFETY CONTROLS AVAILABLE TO USER:
- 'safety on/off' - Enable/disable safety mode
- 'auto on/off' - Enable/disable auto-approval
- 'safety' - Check current safety status{tool_safety_info}

IMPORTANT: If the user says "safety off" or similar commands, they are changing safety settings, not asking you to execute tools unsafely. Respond with confirmation of the setting change."""

        return safety_section

    # ===== SAFETY CONTROL METHODS =====
    def set_safety_mode(self, enabled: Optional[bool] = None) -> str:
        """Enable, disable, or toggle safety mode"""
        if enabled is None:
            self.safety_mode = not self.safety_mode
        else:
            self.safety_mode = enabled

        status = "ENABLED" if self.safety_mode else "DISABLED"
        # Rebuild system prompt with new safety status
        self.system_prompt = self._build_system_prompt()
        return f"🛡️ Safety mode {status}"

    def set_auto_approve(self, enabled: bool) -> str:
        """Enable or disable automatic approval of tool calls"""
        self.auto_approve = enabled
        status = "ENABLED" if enabled else "DISABLED"
        # Rebuild system prompt with new safety status
        self.system_prompt = self._build_system_prompt()
        return f"🤖 Auto-approval {status}"

    def get_safety_status(self) -> str:
        """Get current safety status"""
        safety_status = "🛡️ ON" if self.safety_mode else "🛡️ OFF"
        auto_status = "🤖 AUTO" if self.auto_approve else "🤖 MANUAL"
        return f"Safety: {safety_status} | Approval: {auto_status}"

    def get_detailed_safety_status(self) -> Dict[str, any]:
        """Get detailed safety status information"""
        high_risk_tools = [
            name for name, info in self.tool_info.items()
            if info.requires_approval
        ]

        return {
            "safety_mode": self.safety_mode,
            "auto_approve": self.auto_approve,
            "high_risk_tools": high_risk_tools,
            "total_tools": len(self.local_tools) + len(self.mcp_tools),
            "local_tools": len(self.local_tools),
            "mcp_tools": len(self.mcp_tools)
        }

    def _prompt_for_approval(self, tool_name: str, args: Dict, tool_info: ToolInfo = None) -> bool:
        """Prompt user for tool execution approval with enhanced display"""
        print("\n" + "=" * 80)
        print("🛡️  SAM TOOL EXECUTION APPROVAL REQUIRED")
        print("=" * 80)
        print(f"🔧 Tool: {tool_name}")

        if tool_info:
            print(f"📂 Category: {tool_info.category.value}")
            print(f"📄 Description: {tool_info.description}")
            if tool_info.requires_approval:
                print("⚠️  This tool always requires approval")
            print(f"📊 Usage count: {tool_info.usage_count}")

        print(f"\n📋 Arguments:")
        if args:
            for key, value in args.items():
                # Truncate long values for display
                display_value = str(value)
                if len(display_value) > 100:
                    display_value = display_value[:100] + "...[truncated]"
                print(f"  • {key}: {display_value}")
        else:
            print("  • No arguments required")

        print(f"\n🎛️  Options:")
        print("  ✅ y/yes    - Approve this tool call")
        print("  ❌ n/no     - Deny this tool call")
        print("  🤖 a/auto   - Approve and enable auto-approval for this session")
        print("  🛑 s/stop   - Deny and stop execution")
        print("  ℹ️  i/info   - Show more details about this tool")

        while True:
            try:
                response = input(f"\n🛡️  Approve tool execution? [y/n/a/s/i]: ").strip().lower()

                if response in ['y', 'yes']:
                    print("✅ Tool approved!")
                    return True
                elif response in ['n', 'no']:
                    print("❌ Tool denied!")
                    return False
                elif response in ['a', 'auto']:
                    self.auto_approve = True
                    # Rebuild system prompt to reflect auto-approval
                    self.system_prompt = self._build_system_prompt()
                    print("🤖 Auto-approval enabled for this session")
                    return True
                elif response in ['s', 'stop']:
                    self._handle_stop_request()
                    self.stop_requested = True
                    return False
                elif response in ['i', 'info']:
                    self._show_tool_info(tool_name, tool_info)
                    continue
                else:
                    print("❌ Invalid response. Please enter y/n/a/s/i")

            except KeyboardInterrupt:
                print("\n🛑 Tool execution cancelled by user")
                return False
            except EOFError:
                print("\n❌ Input cancelled")
                return False

    def _handle_stop_request(self):
        """Handle user request to stop tool calls"""
        print("🛑 Tool execution denied. Requesting SAM to stop proposing tool calls.")
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

    # ===== TOOL EXECUTION WITH SAFETY =====
    async def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a tool with safety checks and approval system"""
        try:
            # Get tool info for safety checks
            tool_info = self.tool_info.get(tool_name)

            # Check if approval is required
            requires_approval = (
                    self.safety_mode and
                    (not self.auto_approve or
                     (tool_info and tool_info.requires_approval))
            )

            if requires_approval:
                # Display the raw tool call first
                print(f"\n🔧 RAW TOOL CALL:")
                print(f"Tool: {tool_name}")
                print(f"Arguments: {json.dumps(args, indent=2)}")

                # Prompt for approval
                if not self._prompt_for_approval(tool_name, args, tool_info):
                    return f"❌ Tool execution denied by user: {tool_name}"

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
                print(f"\n📊 RAW RESULTS:")
                print("=" * 60)
                print(str(result))
                print("=" * 60)

                return str(result)

            elif tool_name in self.mcp_tools:
                # Execute MCP tool (implementation depends on your MCP setup)
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

    # ===== LLM COMMUNICATION =====
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
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return f"Error: API request failed with status {response.status_code}"

        except Exception as e:
            logger.error(f"LLM API Error: {str(e)}")
            return f"Error communicating with LLM: {str(e)}"

    def _extract_tool_calls(self, text: str) -> List[Dict]:
        """Extract tool calls from LLM response - supports multiple JSON formats"""
        tool_calls = []
        seen_calls = set()  # Track seen tool calls to avoid duplicates

        # Patterns for finding JSON tool calls in various formats
        patterns = [
            r'```json\s*(.*?)\s*```',  # Standard code blocks
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
                  verbose: bool = False) -> str:  # Reduced max iterations
        """Main execution loop with safety controls"""
        try:
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
                               {"role": "system", "content": self.system_prompt + tools_context}
                           ] + self.conversation_history

                # Add stop message if user requested stop
                if self.stop_message:
                    messages.append({
                        "role": "system",
                        "content": self.stop_message
                    })

                # Get LLM response
                if verbose:
                    print("🤖 SAM is thinking...")

                assistant_response = self.generate_chat_completion(messages)
                last_response = assistant_response

                # Add assistant response to conversation
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_response
                })

                # Extract and execute tool calls
                tool_calls = self._extract_tool_calls(assistant_response)

                if not tool_calls:
                    # No tools to execute, return final response
                    return assistant_response

                # Limit total tool calls to prevent runaway execution
                if tool_call_count >= 10:
                    return f"{last_response}\n\n⚠️ Maximum tool execution limit reached. Stopping to prevent runaway execution."

                # Execute tools and collect results
                tool_results = []
                for tool_call in tool_calls:
                    if self.stop_requested:
                        break

                    tool_name = tool_call.get('name', '')
                    tool_args = tool_call.get('arguments', {})

                    if verbose:
                        print(f"🔧 Executing: {tool_name}")

                    result = await self._execute_tool(tool_name, tool_args)
                    tool_results.append(f"Tool '{tool_name}' result: {result}")
                    tool_call_count += 1

                # Add tool results to conversation
                if tool_results:
                    tool_results_message = "Tool execution results:\n" + "\n".join(tool_results)
                    self.conversation_history.append({
                        "role": "user",
                        "content": tool_results_message
                    })

                # If user requested stop, return appropriate message
                if self.stop_requested:
                    return "🛑 Tool execution stopped by user request."

            return f"{last_response}\n\n🔄 Maximum iterations reached."

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

        # Add MCP tools
        for tool_name in self.mcp_tools:
            tools_list.append(f"- {tool_name}: MCP tool")

        tools_context = f"""

<available_tools>
Available tools ({len(tools_list)} total):
{chr(10).join(tools_list)}
</available_tools>"""

        return tools_context

    def list_tools(self) -> Dict[str, Any]:
        """List all available tools"""
        tools = {}

        for tool_name, tool_info in self.tool_info.items():
            tools[tool_name] = {
                "description": tool_info.description,
                "category": tool_info.category.value,
                "requires_approval": tool_info.requires_approval,
                "usage_count": tool_info.usage_count,
                "parameters": tool_info.parameters
            }

        return {
            "local_tools": tools,
            "mcp_tools": list(self.mcp_tools.keys()),
            "total_count": len(tools) + len(self.mcp_tools)
        }


# ===== CLI INTERFACE =====
def main():
    """CLI interface for SAM Agent"""
    print("🤖 Starting SAM initialization...")

    # Initialize SAM with safety enabled by default
    sam = SAMAgent(
        safety_mode=True,
        auto_approve=False
    )

    # Load core tools plugin properly through plugin manager
    try:
        plugin_path = Path(__file__).parent / "plugins" / "core_tools.py"
        if sam.plugin_manager.load_plugin_from_file(str(plugin_path)):
            print("✅ Core tools plugin loaded successfully!")
        else:
            print("❌ Failed to load core tools plugin")
    except Exception as e:
        print(f"⚠️  Could not load core tools plugin: {e}")

    # Test API connection
    try:
        test_response = sam.generate_chat_completion([
            {"role": "user", "content": "Hello, are you working?"}
        ])
        if "error" not in test_response.lower():
            print("✅ API connection test successful!")
        else:
            print(f"⚠️  API test warning: {test_response}")
    except Exception as e:
        print(f"❌ API connection test failed: {e}")
        return

    # Display capabilities using the tools_info
    tools_info = sam.list_tools()
    print(f"\n=== 🤖 SAM CAPABILITIES ===")
    print(f"🤖 Model: {sam.model_name}")
    print(f"🧠 Context: {sam.context_limit:,} tokens")
    print(f"🔧 Local tools: {len(sam.local_tools)}")
    print(f"🌐 MCP tools: {len(sam.mcp_tools)}")
    print(f"🔌 Plugins: {len(sam.plugin_manager.plugins)}")
    print(f"🛡️ Safety mode: {'ON' if sam.safety_mode else 'OFF'}")
    print(f"🤖 Auto-approve: {'ON' if sam.auto_approve else 'OFF'}")

    # Interactive loop
    print(f"\n=== 🤖 SAM Agent Interactive Mode ===")
    print("Type 'exit' to quit, 'tools' to list available tools")
    print("Commands: 'debug' (toggle debug), 'reset' (clear history), 'tools' (list tools)")
    print("Safety: 'safety on/off', 'auto on/off', 'safety' (status)")

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
                break
            elif user_input.lower() == 'debug':
                debug_mode = not debug_mode
                print(f"🐛 Debug mode: {'ON' if debug_mode else 'OFF'}")
                continue
            elif user_input.lower() == 'reset':
                sam.conversation_history = []
                print("🔄 Conversation history cleared")
                continue
            elif user_input.lower() == 'tools':
                # Use the tools_info variable here, but refresh it to get current usage counts
                current_tools = sam.list_tools()
                total_count = len(current_tools.get('local_tools', {})) + len(current_tools.get('mcp_tools', []))
                print(f"\n🔧 Available Tools ({total_count} total):")

                for name, info in current_tools.get('local_tools', {}).items():
                    approval = "🛡️" if info.get('requires_approval', False) else "✅"
                    usage = info.get('usage_count', 0)
                    usage_text = f" (used {usage}x)" if usage > 0 else ""
                    print(
                        f"  {approval} {name}: {info.get('description', 'No description')} ({info.get('category', 'unknown')}){usage_text}")

                if current_tools.get('mcp_tools'):
                    print(f"🌐 MCP Tools: {', '.join(current_tools['mcp_tools'])}")
                continue

            print("🤖 SAM is thinking...")

            # Run SAM with the user input
            response = asyncio.run(sam.run(user_input, verbose=debug_mode))

            # Add spacing and format the response properly
            print(f"\n🤖 SAM: {response}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            if debug_mode:
                traceback.print_exc()


if __name__ == "__main__":
    main()

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
                description="Semi-Autonomous Model API with Safety Controls",
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
