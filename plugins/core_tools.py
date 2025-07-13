#!/usr/bin/env python3
"""
Core Tools Plugin for SAM Agent
Essential tools for code execution, documentation, and basic utilities
"""

import os
import sys
import json
import math
import re
import time
import random
import subprocess
import platform
import signal
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional
import inspect
import ast
import traceback

# Simple, clean import - no complex fallback logic needed
from sam_agent import SAMPlugin, ToolCategory


class CoreToolsPlugin(SAMPlugin):
    """Core tools for SAM Agent"""

    def __init__(self):
        super().__init__(
            name="Core Tools",
            version="1.0.0",
            description="Essential tools including code execution, search, and calculations"
        )
        self.execution_history = []

    def register_tools(self, agent):
        """Register all core tools with the agent"""

        agent.register_local_tool(
            self.execute_code,
            category=ToolCategory.DEVELOPMENT,
            requires_approval=False
        )

        agent.register_local_tool(
            self.calculate,
            category=ToolCategory.UTILITY,
            requires_approval=False
        )

        agent.register_local_tool(
            self.get_current_time,
            category=ToolCategory.UTILITY,
            requires_approval=False
        )

        agent.register_local_tool(
            self.list_files,
            category=ToolCategory.FILESYSTEM,
            requires_approval=False
        )

        agent.register_local_tool(
            self.read_file,
            category=ToolCategory.FILESYSTEM,
            requires_approval=False
        )

        agent.register_local_tool(
            self.write_file,
            category=ToolCategory.FILESYSTEM,
            requires_approval=True
        )

        agent.register_local_tool(
            self.get_system_info,
            category=ToolCategory.SYSTEM,
            requires_approval=False
        )

    def execute_code(self, code: str, language: str = "python", timeout: int = 30) -> str:
        """
        Execute code safely in a controlled environment.

        Args:
            code: Code to execute
            language: Programming language (python, javascript, bash, powershell)
            timeout: Maximum execution time in seconds

        Returns:
            Execution result or error message
        """
        if language.lower() not in ["python", "javascript", "bash", "shell", "powershell", "pwsh"]:
            return f"❌ Unsupported language: {language}"

        try:
            if language.lower() == "python":
                return self._execute_python_code(code, timeout)
            elif language.lower() == "javascript":
                return self._execute_javascript_code(code)
            elif language.lower() in ["bash", "shell"]:
                return self._execute_shell_code(code)
            elif language.lower() in ["powershell", "pwsh"]:
                return self._execute_powershell_code(code)
        except Exception as e:
            return f"❌ Execution error: {str(e)}"

    def _execute_python_code(self, code: str, timeout: int = 30) -> str:
        """Execute Python code with proper import support"""
        # Capture output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            result = None
            error = None
            start_time = time.time()

            try:
                # Create execution environment with proper imports
                exec_globals = {
                    '__builtins__': __builtins__,
                    # Pre-import commonly needed modules
                    'os': os,
                    'sys': sys,
                    'json': json,
                    'math': math,
                    'random': random,
                    'datetime': datetime,
                    'time': time,
                    're': re,
                    'pathlib': Path,
                    'platform': platform,
                    'subprocess': subprocess,
                }

                # Parse and execute code
                tree = ast.parse(code)

                # If last node is an expression, evaluate it for return value
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    # Execute all but last statement
                    if len(tree.body) > 1:
                        statements = ast.Module(body=tree.body[:-1], type_ignores=[])
                        exec(compile(statements, '<string>', 'exec'), exec_globals)

                    # Evaluate final expression
                    expr = ast.Expression(body=tree.body[-1].value)
                    result = eval(compile(expr, '<string>', 'eval'), exec_globals)
                else:
                    # Execute as statements
                    exec(code, exec_globals)

            except Exception as e:
                error = f"{type(e).__name__}: {str(e)}"

            execution_time = time.time() - start_time

            # Collect output
            stdout_text = stdout_capture.getvalue()
            stderr_text = stderr_capture.getvalue()

            # Format response
            output_parts = [
                f"🐍 Python Code Executed ({execution_time:.3f}s)"
            ]

            if stdout_text.strip():
                output_parts.append(f"\n📤 Output:\n{stdout_text}")

            if result is not None:
                output_parts.append(f"\n🔢 Return Value: {repr(result)}")

            if stderr_text.strip():
                output_parts.append(f"\n⚠️  Stderr:\n{stderr_text}")

            if error:
                output_parts.append(f"\n❌ Error: {error}")

            # Show subprocess results if no output
            if not stdout_text.strip() and result is None and not stderr_text.strip() and not error:
                output_parts.append("\n⚠️  No output captured - check if subprocess commands are working correctly")

            return "".join(output_parts)

        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _execute_javascript_code(self, code: str) -> str:
        """Execute JavaScript code using Node.js"""
        try:
            # Write code to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                # Execute with Node.js
                result = subprocess.run(
                    ['node', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                output = ""
                if result.stdout:
                    output += f"📤 Output:\n{result.stdout}"
                if result.stderr:
                    output += f"\n⚠️  Errors:\n{result.stderr}"

                if result.returncode != 0:
                    output += f"\n❌ Exit code: {result.returncode}"

                return output or "✅ JavaScript executed successfully (no output)"

            finally:
                os.unlink(temp_file)

        except subprocess.TimeoutExpired:
            return "❌ JavaScript execution timed out"
        except FileNotFoundError:
            return "❌ Node.js not found. Please install Node.js to execute JavaScript."
        except Exception as e:
            return f"❌ JavaScript execution error: {str(e)}"

    def _execute_shell_code(self, code: str) -> str:
        """Execute shell commands safely"""
        # Basic safety checks
        dangerous_commands = ['rm -rf', 'sudo', 'passwd', 'chmod 777', 'dd if=', 'format', 'del /']
        if any(cmd in code.lower() for cmd in dangerous_commands):
            return "❌ Potentially dangerous command detected. Execution blocked for safety."

        try:
            result = subprocess.run(
                code,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )

            output = ""
            if result.stdout:
                output += f"📤 Output:\n{result.stdout}"
            if result.stderr:
                output += f"\n⚠️  Errors:\n{result.stderr}"

            if result.returncode != 0:
                output += f"\n❌ Exit code: {result.returncode}"

            return output or "✅ Shell command executed successfully (no output)"

        except subprocess.TimeoutExpired:
            return "❌ Shell command timed out"
        except Exception as e:
            return f"❌ Shell execution error: {str(e)}"

    def _execute_powershell_code(self, code: str) -> str:
        """Execute PowerShell commands"""
        try:
            # Use PowerShell to execute the command
            result = subprocess.run(
                ["powershell", "-Command", code],
                capture_output=True,
                text=True,
                timeout=30
            )

            output = ""
            if result.stdout:
                output += f"📤 Output:\n{result.stdout}"
            if result.stderr:
                output += f"\n⚠️  Errors:\n{result.stderr}"

            if result.returncode != 0:
                output += f"\n❌ Exit code: {result.returncode}"

            return output or "✅ PowerShell command executed successfully (no output)"

        except subprocess.TimeoutExpired:
            return "❌ PowerShell command timed out"
        except FileNotFoundError:
            return "❌ PowerShell not found. Please ensure PowerShell is installed and in PATH."
        except Exception as e:
            return f"❌ PowerShell execution error: {str(e)}"

    def calculate(self, expression: str) -> str:
        """
        Safely evaluate mathematical expressions.

        Args:
            expression: Mathematical expression to evaluate

        Returns:
            Result of the calculation
        """
        try:
            # Create a safe environment for math
            safe_globals = {
                '__builtins__': {},
                'abs': abs,
                'max': max,
                'min': min,
                'round': round,
                'pow': pow,
                'sqrt': math.sqrt,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'log': math.log,
                'log10': math.log10,
                'exp': math.exp,
                'pi': math.pi,
                'e': math.e,
                'degrees': math.degrees,
                'radians': math.radians,
                'factorial': math.factorial,
            }

            # Clean and validate the expression
            # Allow numbers, operators, parentheses, and math functions
            if re.search(r'[^0-9+\-*/().%\s\w]', expression):
                return "❌ Expression contains invalid characters"

            result = eval(expression, safe_globals)
            return f"🧮 Calculation Result: {result}"

        except Exception as e:
            return f"❌ Calculation error: {str(e)}"

    def get_current_time(self, format: str = "%m-%d-%Y %H:%M:%S", timezone: str = "local") -> str:
        """
        Get the current date and time.

        Args:
            format: Time format string (default: "%m-%d-%Y %H:%M:%S")
            timezone: Timezone (currently only supports "local")

        Returns:
            Formatted current time
        """
        try:
            now = datetime.now()
            formatted_time = now.strftime(format)
            return f"🕒 Current time: {formatted_time}"
        except Exception as e:
            return f"❌ Time format error: {str(e)}"

    def list_files(self, directory: str = ".", pattern: str = "*", max_files: int = 50) -> str:
        """
        List files in a directory.

        Args:
            directory: Directory to list (default: current directory)
            pattern: File pattern to match (default: all files)
            max_files: Maximum number of files to return

        Returns:
            List of files matching the pattern
        """
        try:
            dir_path = Path(directory)
            if not dir_path.exists():
                return f"❌ Directory does not exist: {directory}"

            if not dir_path.is_dir():
                return f"❌ Path is not a directory: {directory}"

            # Get files matching pattern
            files = list(dir_path.glob(pattern))[:max_files]

            if not files:
                return f"📂 No files found matching pattern '{pattern}' in {directory}"

            # Format file list
            file_list = []
            for file_path in files:
                if file_path.is_file():
                    size = file_path.stat().st_size
                    file_list.append(f"📄 {file_path.name} ({size} bytes)")
                elif file_path.is_dir():
                    file_list.append(f"📁 {file_path.name}/")

            result = f"📂 Files in {directory}:\n"
            result += "\n".join(file_list)

            if len(files) == max_files:
                result += f"\n... (truncated to {max_files} files)"

            return result

        except Exception as e:
            return f"❌ Error listing files: {str(e)}"

    def read_file(self, file_path: str, max_size: int = 100000) -> str:
        """
        Read contents of a text file.

        Args:
            file_path: Path to the file to read
            max_size: Maximum file size to read (bytes)

        Returns:
            File contents or error message
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                return f"❌ File does not exist: {file_path}"

            if not file_path.is_file():
                return f"❌ Path is not a file: {file_path}"

            # Check file size
            file_size = file_path.stat().st_size
            if file_size > max_size:
                return f"❌ File too large ({file_size} bytes, max: {max_size} bytes)"

            # Read file content
            content = file_path.read_text(encoding='utf-8', errors='replace')

            return f"📄 File: {file_path}\n" + "=" * 50 + f"\n{content}\n" + "=" * 50

        except Exception as e:
            return f"❌ Error reading file: {str(e)}"

    def write_file(self, file_path: str, content: str, mode: str = "w") -> str:
        """
        Write content to a file.

        Args:
            file_path: Path to the file to write
            content: Content to write to the file
            mode: Write mode ('w' for overwrite, 'a' for append)

        Returns:
            Success message or error
        """
        try:
            file_path = Path(file_path)

            # Create directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if mode == "a":
                with open(file_path, mode, encoding='utf-8') as f:
                    f.write(content)
                action = "appended to"
            else:
                with open(file_path, mode, encoding='utf-8') as f:
                    f.write(content)
                action = "written to"

            size = len(content.encode('utf-8'))
            return f"✅ Content {action} {file_path} ({size} bytes)"

        except Exception as e:
            return f"❌ Error writing file: {str(e)}"

    def get_system_info(self) -> str:
        """
        Get system information.

        Returns:
            System information summary
        """
        try:
            info = {
                "platform": platform.platform(),
                "system": platform.system(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
                "python_version": platform.python_version(),
                "hostname": platform.node(),
            }

            result = "💻 System Information:\n"
            result += f"🖥️  Platform: {info['platform']}\n"
            result += f"⚙️  System: {info['system']}\n"
            result += f"🔧 Processor: {info['processor']}\n"
            result += f"🏗️  Architecture: {info['architecture'][0]}\n"
            result += f"🐍 Python: {info['python_version']}\n"
            result += f"🌐 Hostname: {info['hostname']}"

            return result

        except Exception as e:
            return f"❌ Error getting system info: {str(e)}"


# Create the plugin instance
def get_plugin():
    """Plugin entry point"""
    return CoreToolsPlugin()


# For direct execution testing
if __name__ == "__main__":
    plugin = CoreToolsPlugin()

    # Test the execute_code function
    test_code = """
from datetime import datetime
current_time = datetime.now()
print(current_time.strftime('%Y-%m-%d %H:%M:%S'))
"""

    print("Testing execute_code function:")
    result = plugin.execute_code(test_code, "python")
    print(result)