#!/usr/bin/env python3
"""
Computer Control Plugin for SAM Agent
Simple keyboard and mouse control functionality
"""

import os
import sys
import time
import platform
from typing import Any, Dict, List, Optional, Tuple
import traceback
# Try to import PIL with fallback
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

import base64
from io import BytesIO

# Try to import required libraries with fallbacks
try:
    import pyautogui

    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    pyautogui = None

try:
    import keyboard

    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    keyboard = None

try:
    import mouse

    MOUSE_AVAILABLE = True
except ImportError:
    MOUSE_AVAILABLE = False
    mouse = None

# Simple, clean import - no complex fallback logic needed
from sam_agent import SAMPlugin, ToolCategory


class ComputerControlPlugin(SAMPlugin):
    """Computer control plugin for keyboard and mouse automation"""

    def __init__(self):
        super().__init__(
            name="Computer Control",
            version="1.0.0",
            description="Simple keyboard and mouse control functionality"
        )

        # Configure pyautogui safety settings
        if PYAUTOGUI_AVAILABLE:
            pyautogui.FAILSAFE = True  # Move mouse to corner to abort
            pyautogui.PAUSE = 0.1  # Small pause between actions

        self.action_history = []

    def register_tools(self, agent):
        """Register all computer control tools with the agent"""

        # Store reference to agent for dynamic provider checking
        self._agent = agent

        # Store current provider for provider-specific behavior
        self._current_provider = agent.raw_config.get('provider', 'unknown')

        # Mouse control tools
        agent.register_local_tool(
            self.move_mouse,
            category=ToolCategory.SYSTEM,
            requires_approval=True
        )

        agent.register_local_tool(
            self.click_mouse,
            category=ToolCategory.SYSTEM,
            requires_approval=True
        )

        agent.register_local_tool(
            self.scroll_mouse,
            category=ToolCategory.SYSTEM,
            requires_approval=True
        )

        agent.register_local_tool(
            self.get_mouse_position,
            category=ToolCategory.SYSTEM,
            requires_approval=False
        )

        # Keyboard control tools
        agent.register_local_tool(
            self.type_text,
            category=ToolCategory.SYSTEM,
            requires_approval=True
        )

        agent.register_local_tool(
            self.press_key,
            category=ToolCategory.SYSTEM,
            requires_approval=True
        )

        agent.register_local_tool(
            self.key_combination,
            category=ToolCategory.SYSTEM,
            requires_approval=True
        )

        # Screen capture tools
        agent.register_local_tool(
            self.get_screen_size,
            category=ToolCategory.SYSTEM,
            requires_approval=False
        )

        # Utility tools
        agent.register_local_tool(
            self.check_dependencies,
            category=ToolCategory.UTILITY,
            requires_approval=False
        )

        # Add this to the register_tools method, after the existing screenshot tool
        agent.register_local_tool(
            self.take_screenshot_with_ai_integration,
            category=ToolCategory.SYSTEM,
            requires_approval=True
        )

    def _check_pyautogui(self) -> str:
        """Check if pyautogui is available"""
        if not PYAUTOGUI_AVAILABLE:
            return "❌ PyAutoGUI not available. Install with: pip install pyautogui"
        return None

    def _check_keyboard(self) -> str:
        """Check if keyboard library is available"""
        if not KEYBOARD_AVAILABLE:
            return "❌ Keyboard library not available. Install with: pip install keyboard"
        return None

    def _check_mouse(self) -> str:
        """Check if mouse library is available"""
        if not MOUSE_AVAILABLE:
            return "❌ Mouse library not available. Install with: pip install mouse"
        return None

    def _log_action(self, action: str, details: str = ""):
        """Log an action for history tracking"""
        timestamp = time.strftime("%H:%M:%S")
        self.action_history.append(f"[{timestamp}] {action} {details}")
        # Keep only last 50 actions
        if len(self.action_history) > 50:
            self.action_history.pop(0)

    def move_mouse(self, x: int, y: int, duration: float = 0.5) -> str:
        """
        Move mouse to specified coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            duration: Duration of movement in seconds

        Returns:
            Success message or error
        """
        error = self._check_pyautogui()
        if error:
            return error

        try:
            pyautogui.moveTo(x, y, duration=duration)
            self._log_action("MOVE_MOUSE", f"to ({x}, {y})")
            return f"🖱️ Mouse moved to ({x}, {y}) in {duration}s"
        except Exception as e:
            return f"❌ Error moving mouse: {str(e)}"

    def click_mouse(self, x: Optional[int] = None, y: Optional[int] = None,
                    button: str = "left", clicks: int = 1, interval: float = 0.1) -> str:
        """
        Click mouse at specified coordinates or current position.

        Args:
            x: X coordinate (optional, uses current position if None)
            y: Y coordinate (optional, uses current position if None)
            button: Mouse button ('left', 'right', 'middle')
            clicks: Number of clicks
            interval: Interval between clicks

        Returns:
            Success message or error
        """
        error = self._check_pyautogui()
        if error:
            return error

        try:
            if button not in ['left', 'right', 'middle']:
                return "❌ Invalid button. Use 'left', 'right', or 'middle'"

            if x is not None and y is not None:
                pyautogui.click(x, y, clicks=clicks, interval=interval, button=button)
                location = f"at ({x}, {y})"
            else:
                pyautogui.click(clicks=clicks, interval=interval, button=button)
                location = "at current position"

            self._log_action("CLICK_MOUSE", f"{button} button {clicks}x {location}")
            return f"🖱️ {button.title()} clicked {clicks}x {location}"
        except Exception as e:
            return f"❌ Error clicking mouse: {str(e)}"

    def scroll_mouse(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> str:
        """
        Scroll mouse wheel.

        Args:
            clicks: Number of scroll clicks (positive = up, negative = down)
            x: X coordinate (optional)
            y: Y coordinate (optional)

        Returns:
            Success message or error
        """
        error = self._check_pyautogui()
        if error:
            return error

        try:
            if x is not None and y is not None:
                pyautogui.scroll(clicks, x=x, y=y)
                location = f"at ({x}, {y})"
            else:
                pyautogui.scroll(clicks)
                location = "at current position"

            direction = "up" if clicks > 0 else "down"
            self._log_action("SCROLL_MOUSE", f"{abs(clicks)} clicks {direction} {location}")
            return f"🖱️ Scrolled {abs(clicks)} clicks {direction} {location}"
        except Exception as e:
            return f"❌ Error scrolling mouse: {str(e)}"

    def get_mouse_position(self) -> str:
        """
        Get current mouse position.

        Returns:
            Current mouse coordinates
        """
        error = self._check_pyautogui()
        if error:
            return error

        try:
            x, y = pyautogui.position()
            return f"🖱️ Mouse position: ({x}, {y})"
        except Exception as e:
            return f"❌ Error getting mouse position: {str(e)}"

    def type_text(self, text: str, interval: float = 0.05) -> str:
        """
        Type text at current cursor position.

        Args:
            text: Text to type
            interval: Interval between keystrokes

        Returns:
            Success message or error
        """
        error = self._check_pyautogui()
        if error:
            return error

        try:
            pyautogui.typewrite(text, interval=interval)
            self._log_action("TYPE_TEXT", f"'{text[:50]}{'...' if len(text) > 50 else ''}'")
            return f"⌨️ Typed text: '{text[:50]}{'...' if len(text) > 50 else ''}'"
        except Exception as e:
            return f"❌ Error typing text: {str(e)}"

    def press_key(self, key: str, presses: int = 1, interval: float = 0.1) -> str:
        """
        Press a key or key combination.

        Args:
            key: Key to press (e.g., 'enter', 'space', 'ctrl', 'alt', 'tab')
            presses: Number of times to press the key
            interval: Interval between presses

        Returns:
            Success message or error
        """
        error = self._check_pyautogui()
        if error:
            return error

        try:
            pyautogui.press(key, presses=presses, interval=interval)
            self._log_action("PRESS_KEY", f"'{key}' {presses}x")
            return f"⌨️ Pressed '{key}' {presses}x"
        except Exception as e:
            return f"❌ Error pressing key: {str(e)}"

    def key_combination(self, keys: str) -> str:
        """
        Press a key combination (e.g., 'ctrl+c', 'alt+tab').

        Args:
            keys: Key combination separated by '+' (e.g., 'ctrl+c', 'shift+tab')

        Returns:
            Success message or error
        """
        error = self._check_pyautogui()
        if error:
            return error

        try:
            key_list = [k.strip() for k in keys.split('+')]
            pyautogui.hotkey(*key_list)
            self._log_action("KEY_COMBINATION", f"'{keys}'")
            return f"⌨️ Pressed key combination: '{keys}'"
        except Exception as e:
            return f"❌ Error pressing key combination: {str(e)}"

    def take_screenshot_for_ai(self, region: Optional[Tuple[int, int, int, int]] = None,
                               compress_quality: int = 50, max_size_kb: int = 100) -> Dict[str, Any]:
        """
        Take a screenshot of the primary monitor optimized for AI consumption.

        Args:
            region: Optional region tuple (left, top, width, height) relative to primary monitor
            compress_quality: JPEG compression quality (1-100, lower = more compression)
            max_size_kb: Maximum size in KB for the compressed image

        Returns:
            Dict with 'success', 'file_path', 'base64_data', 'width', 'height'
        """
        error = self._check_pyautogui()
        if error:
            return {"success": False, "error": error}

        if not PIL_AVAILABLE:
            return {"success": False, "error": "❌ PIL/Pillow not available. Install with: pip install Pillow"}

        try:
            # Get primary monitor dimensions
            import tkinter as tk
            root = tk.Tk()
            primary_width = root.winfo_screenwidth()
            primary_height = root.winfo_screenheight()
            root.destroy()

            # Take screenshot of primary monitor only
            if region:
                # Region is relative to primary monitor
                screenshot = pyautogui.screenshot(region=region)
            else:
                # Screenshot just the primary monitor (0,0 to primary_width, primary_height)
                screenshot = pyautogui.screenshot(region=(0, 0, primary_width, primary_height))

            # Save original
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_primary_{timestamp}.png"
            screenshot.save(filename)

            # Convert to RGB if needed (removes transparency)
            if screenshot.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', screenshot.size, (255, 255, 255))
                background.paste(screenshot, mask=screenshot.split()[-1] if screenshot.mode == 'RGBA' else None)
                screenshot = background

            # Resize if still too large (reduce resolution to save space)
            max_dimension = 1920  # Max width or height
            if screenshot.width > max_dimension or screenshot.height > max_dimension:
                ratio = min(max_dimension / screenshot.width, max_dimension / screenshot.height)
                new_size = (int(screenshot.width * ratio), int(screenshot.height * ratio))
                screenshot = screenshot.resize(new_size, Image.Resampling.LANCZOS)

            # Compress with multiple attempts to stay under size limit
            max_size_bytes = max_size_kb * 1024
            quality = compress_quality

            while quality > 10:  # Don't go below quality 10
                compressed_io = BytesIO()
                screenshot.save(compressed_io, format='JPEG', quality=quality, optimize=True)
                compressed_data = compressed_io.getvalue()

                if len(compressed_data) <= max_size_bytes:
                    break

                quality -= 10  # Reduce quality more aggressively

            # If still too big, resize further
            if len(compressed_data) > max_size_bytes and quality <= 10:
                # Try smaller resolution
                smaller_size = (screenshot.width // 2, screenshot.height // 2)
                screenshot = screenshot.resize(smaller_size, Image.Resampling.LANCZOS)

                compressed_io = BytesIO()
                screenshot.save(compressed_io, format='JPEG', quality=30, optimize=True)
                compressed_data = compressed_io.getvalue()

            # Base64 encode
            base64_data = base64.b64encode(compressed_data).decode('utf-8')

            # Estimate token count (rough: 1 token ≈ 1.5 base64 chars for images)
            estimated_tokens = len(base64_data) // 1.5

            self._log_action("SCREENSHOT_AI",
                             f"primary monitor saved as '{filename}', {len(compressed_data):,} bytes, "
                             f"~{estimated_tokens:,.0f} tokens, quality={quality}")

            return {
                "success": True,
                "file_path": filename,
                "base64_data": base64_data,
                "width": screenshot.width,
                "height": screenshot.height,
                "compressed_size": len(compressed_data),
                "base64_size": len(base64_data),
                "estimated_tokens": int(estimated_tokens),
                "final_quality": quality,
                "monitor": "primary"
            }

        except Exception as e:
            return {"success": False, "error": f"Error taking primary monitor screenshot: {str(e)}"}

    def take_screenshot_with_ai_integration(self, region: Optional[Tuple[int, int, int, int]] = None,
                                            include_in_next_message: bool = True) -> str:
        """
        Take screenshot and optionally prepare it for AI consumption based on current provider.
        """
        screenshot_result = self.take_screenshot_for_ai(region=region)

        if not screenshot_result["success"]:
            return screenshot_result["error"]

        # Get current provider dynamically from agent
        current_provider = getattr(self, '_agent', None)
        if current_provider:
            current_provider = current_provider.raw_config.get('provider', 'unknown')
        else:
            current_provider = 'unknown'

        if include_in_next_message and current_provider == 'claude':
            # Store the base64 data for the next message
            self._pending_image_data = screenshot_result["base64_data"]
            return (f"📸 Screenshot taken and prepared for Claude! "
                    f"File: {screenshot_result['file_path']} "
                    f"(Compressed: {screenshot_result['compressed_size']:,} bytes)")
        else:
            provider_msg = f" (Provider: {current_provider} - image data not sent)" if current_provider != 'claude' else ""
            return f"📸 Screenshot saved: {screenshot_result['file_path']}{provider_msg}"


    def get_screen_size(self) -> str:
        """
        Get screen dimensions.

        Returns:
            Screen width and height
        """
        error = self._check_pyautogui()
        if error:
            return error

        try:
            width, height = pyautogui.size()
            return f"🖥️ Screen size: {width} x {height} pixels"
        except Exception as e:
            return f"❌ Error getting screen size: {str(e)}"

    def check_dependencies(self) -> str:
        """
        Check which dependencies are available.

        Returns:
            Status of all dependencies
        """
        status = "🔧 Computer Control Dependencies:\n"

        # Check PyAutoGUI
        if PYAUTOGUI_AVAILABLE:
            status += "✅ PyAutoGUI: Available\n"
        else:
            status += "❌ PyAutoGUI: Not installed (pip install pyautogui)\n"

        # Check keyboard library
        if KEYBOARD_AVAILABLE:
            status += "✅ Keyboard: Available\n"
        else:
            status += "❌ Keyboard: Not installed (pip install keyboard)\n"

        # Check mouse library
        if MOUSE_AVAILABLE:
            status += "✅ Mouse: Available\n"
        else:
            status += "❌ Mouse: Not installed (pip install mouse)\n"

        # Platform info
        status += f"🖥️ Platform: {platform.system()}\n"

        # Safety info
        if PYAUTOGUI_AVAILABLE:
            status += "🛡️ Safety: Failsafe enabled (move mouse to corner to abort)"

        return status


# Create the plugin instance
def create_plugin():
    """Plugin entry point"""
    return ComputerControlPlugin()


# For direct execution testing
if __name__ == "__main__":
    plugin = ComputerControlPlugin()

    # Test dependency check
    print("Testing dependency check:")
    result = plugin.check_dependencies()
    print(result)