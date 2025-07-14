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
            self.take_screenshot,
            category=ToolCategory.SYSTEM,
            requires_approval=True
        )

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

    def take_screenshot(self, filename: Optional[str] = None,
                        region: Optional[Tuple[int, int, int, int]] = None) -> str:
        """
        Take a screenshot of the screen.

        Args:
            filename: Optional filename to save screenshot
            region: Optional region tuple (left, top, width, height)

        Returns:
            Success message or error
        """
        error = self._check_pyautogui()
        if error:
            return error

        try:
            if filename is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"

            if region:
                screenshot = pyautogui.screenshot(region=region)
            else:
                screenshot = pyautogui.screenshot()

            screenshot.save(filename)
            self._log_action("SCREENSHOT", f"saved as '{filename}'")
            return f"📸 Screenshot saved as '{filename}'"
        except Exception as e:
            return f"❌ Error taking screenshot: {str(e)}"

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