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
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional
import inspect
import ast
import traceback

# Robust import handling for dynamic plugin loading
try:
    # First try normal import
    from sam_agent import SAMPlugin, ToolCategory
except ImportError:
    # If that fails, try adding the parent directory to sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # Try import again
    try:
        from sam_agent import SAMPlugin, ToolCategory
    except ImportError:
        # Last resort: try the project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
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

    def execute_code(self, code: str, language: str = "python") -> str:
        """
        Execute code safely in a controlled environment.

        Args:
            code: Code to execute
            language: Programming language (python, javascript, bash)

        Returns:
            Execution result or error message
        """
        if language.lower() not in ["python", "javascript", "bash", "shell"]:
            return f"❌ Unsupported language: {language}"

        try:
            if language.lower() == "python":
                return self._execute_python_code(code)
            elif language.lower() == "javascript":
                return self._execute_javascript_code(code)
            elif language.lower() in ["bash", "shell"]:
                return self._execute_shell_code(code)
        except Exception as e:
            return f"❌ Execution error: {str(e)}"

    def _execute_python_code(self, code: str) -> str:
        """Execute Python code safely"""
        # Create a restricted environment
        restricted_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'abs': abs,
                'max': max,
                'min': min,
                'sum': sum,
                'sorted': sorted,
                'enumerate': enumerate,
                'zip': zip,
                'any': any,
                'all': all,
            },
            'math': math,
            'json': json,
            'datetime': datetime,
            'time': time,
        }

        # Capture output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # Execute the code
            exec(code, restricted_globals)

            # Get output
            output = stdout_capture.getvalue()
            errors = stderr_capture.getvalue()

            result = ""
            if output:
                result += f"Output:\n{output}"
            if errors:
                result += f"\nErrors:\n{errors}"

            if not result:
                result = "✅ Code executed successfully (no output)"

            return result

        except Exception as e:
            return f"❌ Python execution error: {str(e)}\n{traceback.format_exc()}"
        finally:
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
                    output += f"Output:\n{result.stdout}"
                if result.stderr:
                    output += f"\nErrors:\n{result.stderr}"

                if result.returncode != 0:
                    output += f"\nExit code: {result.returncode}"

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
        dangerous_commands = ['rm -rf', 'sudo', 'passwd', 'chmod 777', 'dd if=']
        if any(cmd in code.lower() for cmd in dangerous_commands):
            return "❌ Potentially dangerous command detected. Execution blocked."

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
                output += f"Output:\n{result.stdout}"
            if result.stderr:
                output += f"\nErrors:\n{result.stderr}"

            if result.returncode != 0:
                output += f"\nExit code: {result.returncode}"

            return output or "✅ Shell command executed successfully (no output)"

        except subprocess.TimeoutExpired:
            return "❌ Shell command timed out"
        except Exception as e:
            return f"❌ Shell execution error: {str(e)}"

    def calculate(self, expression: str) -> str:
        """
        Safely evaluate mathematical expressions.

        Args:
            expression: Mathematical expression to evaluate

        Returns:
            Result of the calculation
        """
        try:
            # Remove any potentially dangerous functions
            safe_expression = re.sub(r'[^0-9+\-*/().%\s]', '', expression)

            # Use eval with restricted globals for safety
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
                'exp': math.exp,
                'pi': math.pi,
                'e': math.e,
            }

            result = eval(safe_expression, safe_globals)
            return f"📊 Result: {result}"

        except Exception as e:
            return f"❌ Calculation error: {str(e)}"

    def get_current_time(self, format: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        Get the current date and time.

        Args:
            format: Time format string (default: "%Y-%m-%d %H:%M:%S")

        Returns:
            Formatted current time
        """
        try:
            now = datetime.now()
            return f"🕐 Current time: {now.strftime(format)}"
        except Exception as e:
            return f"❌ Time format error: {str(e)}"

    def list_files(self, directory: str = ".", pattern: str = "*") -> str:
        """
        List files in a directory.

        Args:
            directory: Directory path to list (default: current directory)
            pattern: File pattern to match (default: all files)

        Returns:
            List of files in the directory
        """
        try:
            import glob

            path = Path(directory).resolve()
            if not path.exists():
                return f"❌ Directory not found: {directory}"

            if not path.is_dir():
                return f"❌ Not a directory: {directory}"

            search_pattern = str(path / pattern)
            files = glob.glob(search_pattern)

            if not files:
                return f"📁 No files found matching pattern '{pattern}' in {directory}"

            # Sort and format file list
            files.sort()
            file_list = []

            for file_path in files:
                file_obj = Path(file_path)
                if file_obj.is_file():
                    size = file_obj.stat().st_size
                    size_str = self._format_file_size(size)
                    file_list.append(f"📄 {file_obj.name} ({size_str})")
                elif file_obj.is_dir():
                    file_list.append(f"📁 {file_obj.name}/")

            return f"📁 Files in {directory}:\n" + "\n".join(file_list)

        except Exception as e:
            return f"❌ Error listing files: {str(e)}"

    def read_file(self, file_path: str, max_lines: int = 100) -> str:
        """
        Read contents of a text file.

        Args:
            file_path: Path to the file to read
            max_lines: Maximum number of lines to read (default: 100)

        Returns:
            File contents or error message
        """
        try:
            path = Path(file_path).resolve()
            if not path.exists():
                return f"❌ File not found: {file_path}"

            if not path.is_file():
                return f"❌ Not a file: {file_path}"

            # Check file size
            size = path.stat().st_size
            if size > 1024 * 1024:  # 1MB limit
                return f"❌ File too large to read: {self._format_file_size(size)}"

            # Read file content
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            if len(lines) > max_lines:
                content = ''.join(lines[:max_lines])
                content += f"\n... (truncated, showing first {max_lines} lines of {len(lines)})"
            else:
                content = ''.join(lines)

            return f"📄 Contents of {file_path}:\n```\n{content}\n```"

        except Exception as e:
            return f"❌ Error reading file: {str(e)}"

    def write_file(self, file_path: str, content: str, append: bool = False) -> str:
        """
        Write content to a file.

        Args:
            file_path: Path to the file to write
            content: Content to write to the file
            append: Whether to append to existing file (default: False)

        Returns:
            Success message or error
        """
        try:
            path = Path(file_path).resolve()

            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)

            mode = 'a' if append else 'w'
            with open(path, mode, encoding='utf-8') as f:
                f.write(content)

            action = "appended to" if append else "written to"
            size = len(content.encode('utf-8'))
            return f"✅ Content {action} {file_path} ({self._format_file_size(size)})"

        except Exception as e:
            return f"❌ Error writing file: {str(e)}"

    def get_system_info(self) -> str:
        """
        Get system information.

        Returns:
            System information including OS, CPU, memory, etc.
        """
        try:
            info = []

            # Basic system info
            info.append(f"🖥️ Platform: {platform.platform()}")
            info.append(f"💻 System: {platform.system()} {platform.release()}")
            info.append(f"🏗️ Architecture: {platform.architecture()[0]}")
            info.append(f"🔧 Machine: {platform.machine()}")
            info.append(f"📍 Node: {platform.node()}")

            # Python info
            info.append(f"🐍 Python: {platform.python_version()}")

            # CPU info
            try:
                import psutil
                cpu_count = psutil.cpu_count()
                cpu_percent = psutil.cpu_percent(interval=1)
                info.append(f"⚡ CPU: {cpu_count} cores, {cpu_percent:.1f}% usage")

                # Memory info
                memory = psutil.virtual_memory()
                total_gb = memory.total / (1024 ** 3)
                used_gb = memory.used / (1024 ** 3)
                info.append(f"🧠 Memory: {used_gb:.1f}GB / {total_gb:.1f}GB ({memory.percent:.1f}% used)")

            except ImportError:
                info.append("📊 Install 'psutil' for detailed system metrics")

            return "🖥️ System Information:\n" + "\n".join(info)

        except Exception as e:
            return f"❌ Error getting system info: {str(e)}"

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 ** 2:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 ** 3:
            return f"{size_bytes / (1024 ** 2):.1f} MB"
        else:
            return f"{size_bytes / (1024 ** 3):.1f} GB"