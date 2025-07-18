#!/usr/bin/env python3
"""
Enhanced Computer Control Plugin for SAM Agent
Intelligent area selection and context-aware computer vision
"""

import os
import sys
import time
import platform
import re
from typing import Any, Dict, List, Optional, Tuple
import traceback
import json
import base64
from io import BytesIO

# Try to import PIL with fallback
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

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
    """Intelligent computer control plugin with context-aware vision"""

    def __init__(self):
        super().__init__(
            name="Computer Control",
            version="3.0.0",
            description="Intelligent computer control with context-aware vision and area selection"
        )

        # Configure pyautogui safety settings
        if PYAUTOGUI_AVAILABLE:
            pyautogui.FAILSAFE = True  # Move mouse to corner to abort
            pyautogui.PAUSE = 0.1  # Small pause between actions

        self.action_history = []
        self._pending_image_data = None
        self._agent = None

        # Context-aware keyword mappings
        self.area_keywords = {
            "taskbar": ["start", "search", "taskbar", "menu", "windows button", "system tray", "notification", "clock"],
            "desktop": ["desktop", "icon", "wallpaper", "background", "shortcut", "folder"],
            "center": ["window", "application", "app", "dialog", "popup", "form", "button"],
            "top_half": ["title bar", "menu bar", "toolbar", "tab", "close", "minimize", "maximize"],
            "bottom_half": ["status bar", "footer", "bottom"],
            "left_half": ["sidebar", "left panel", "navigation"],
            "right_half": ["right panel", "properties"]
        }

    def register_tools(self, agent):
        """Register all computer control tools with the agent"""

        # Store reference to agent for dynamic provider checking
        self._agent = agent

        # Basic control tools
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

        # Intelligent vision tools
        agent.register_local_tool(
            self.smart_screenshot_and_analyze,
            category=ToolCategory.SYSTEM,
            requires_approval=True
        )

        agent.register_local_tool(
            self.find_and_click_element,
            category=ToolCategory.SYSTEM,
            requires_approval=True
        )

        agent.register_local_tool(
            self.complete_task_with_vision,
            category=ToolCategory.SYSTEM,
            requires_approval=True
        )

        # Utility tools
        agent.register_local_tool(
            self.get_screen_size,
            category=ToolCategory.SYSTEM,
            requires_approval=False
        )

        agent.register_local_tool(
            self.list_available_areas,
            category=ToolCategory.UTILITY,
            requires_approval=False
        )

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

    def _check_pil(self) -> str:
        """Check if PIL is available"""
        if not PIL_AVAILABLE:
            return "❌ PIL/Pillow not available. Install with: pip install Pillow"
        return None

    def _log_action(self, action: str, details: str = ""):
        """Log an action for history tracking"""
        timestamp = time.strftime("%H:%M:%S")
        self.action_history.append(f"[{timestamp}] {action} {details}")
        # Keep only last 50 actions
        if len(self.action_history) > 50:
            self.action_history.pop(0)

    def _get_current_provider(self) -> str:
        """Get the current LLM provider"""
        if self._agent and hasattr(self._agent, 'raw_config'):
            return self._agent.raw_config.get('provider', 'unknown')
        return 'unknown'

    def get_screen_regions(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Get smart screen regions based on current screen size"""
        try:
            width, height = pyautogui.size()

            return {
                "full": (0, 0, width, height),
                "taskbar": (0, height - 100, width, 100),
                "start_menu_area": (0, height - 250, 450, 250),
                "system_tray": (width - 350, height - 100, 350, 100),
                "desktop": (0, 0, width, height - 100),
                "top_half": (0, 0, width, height // 2),
                "bottom_half": (0, height // 2, width, height // 2),
                "left_half": (0, 0, width // 2, height),
                "right_half": (width // 2, 0, width // 2, height),
                "center": (width // 4, height // 4, width // 2, height // 2),
                "top_bar": (0, 0, width, 150),
                "search_area": (width // 4, height // 4, width // 2, height // 4)
            }
        except Exception as e:
            return {"full": (0, 0, 1920, 1080)}  # Fallback

    def _detect_best_area(self, goal: str) -> str:
        """Intelligently detect the best screen area based on goal description"""
        goal_lower = goal.lower()

        # Score each area based on keyword matches
        area_scores = {}

        for area, keywords in self.area_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in goal_lower:
                    score += len(keyword)  # Longer matches get higher scores
            area_scores[area] = score

        # Find the area with the highest score
        best_area = max(area_scores, key=area_scores.get)

        # If no good matches, use fallback logic
        if area_scores[best_area] == 0:
            if any(word in goal_lower for word in ["click", "find", "locate", "see"]):
                return "center"
            else:
                return "full"

        return best_area

    def _optimize_screenshot(self, screenshot: Image.Image,
                             max_width: int = 1280,
                             max_height: int = 720,
                             quality: int = 60,
                             max_file_size: int = 80_000) -> bytes:
        """Aggressively optimize screenshot for rate limits"""

        # Convert to RGB if needed
        if screenshot.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', screenshot.size, (255, 255, 255))
            if screenshot.mode == 'RGBA':
                background.paste(screenshot, mask=screenshot.split()[-1])
            else:
                background.paste(screenshot)
            screenshot = background

        # Resize if too large
        if screenshot.width > max_width or screenshot.height > max_height:
            ratio = min(max_width / screenshot.width, max_height / screenshot.height)
            new_size = (int(screenshot.width * ratio), int(screenshot.height * ratio))
            screenshot = screenshot.resize(new_size, Image.Resampling.LANCZOS)

        # Aggressive compression
        current_quality = quality

        while current_quality > 15:
            compressed_io = BytesIO()
            screenshot.save(compressed_io, format='JPEG', quality=current_quality, optimize=True)
            compressed_data = compressed_io.getvalue()

            if len(compressed_data) <= max_file_size:
                return compressed_data

            current_quality -= 10

        # Final fallback - smaller size
        if len(compressed_data) > max_file_size:
            smaller_size = (screenshot.width * 2 // 3, screenshot.height * 2 // 3)
            screenshot = screenshot.resize(smaller_size, Image.Resampling.LANCZOS)

            compressed_io = BytesIO()
            screenshot.save(compressed_io, format='JPEG', quality=20, optimize=True)
            compressed_data = compressed_io.getvalue()

        return compressed_data

    def _create_vision_message(self, text_prompt: str, image_data: bytes) -> Dict:
        """Create properly structured message for Claude with images"""
        base64_image = base64.b64encode(image_data).decode('utf-8')

        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text_prompt
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image
                    }
                }
            ]
        }

    def _claude_api_call_with_retry(self, vision_message: Dict, max_retries: int = 2) -> str:
        """Make Claude API call with retry (synchronous)"""

        for attempt in range(max_retries):
            try:
                if self._agent:
                    response = self._agent.generate_chat_completion([vision_message])
                    return response
                else:
                    return "❌ Error: No agent reference available"

            except Exception as e:
                error_str = str(e)
                if "rate_limit_error" in error_str and attempt < max_retries - 1:
                    delay = 5 + (attempt * 5)  # 5s, 10s delays
                    print(f"⏳ Rate limited, retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    return f"❌ Claude API error: {error_str}"

    def _parse_coordinates_from_response(self, response: str) -> Optional[Tuple[int, int]]:
        """Extract coordinates from Claude's response"""
        try:
            # Look for coordinate patterns like (123, 456) or 123,456 or click at 123, 456
            patterns = [
                r'\((\d+),\s*(\d+)\)',  # (123, 456)
                r'(\d+),\s*(\d+)',  # 123, 456
                r'x:\s*(\d+).*?y:\s*(\d+)',  # x: 123 y: 456
                r'(\d+)\s*,\s*(\d+)'  # 123,456
            ]

            for pattern in patterns:
                match = re.search(pattern, response)
                if match:
                    x, y = int(match.group(1)), int(match.group(2))
                    # Sanity check coordinates
                    if 0 <= x <= 3000 and 0 <= y <= 2000:
                        return (x, y)

            return None
        except:
            return None

    # ===== BASIC CONTROL TOOLS =====

    def move_mouse(self, x: int, y: int, duration: float = 0.5) -> str:
        """Move mouse to specified coordinates"""
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
        """Click mouse at specified coordinates or current position"""
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
        """Scroll mouse wheel"""
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
        """Get current mouse position"""
        error = self._check_pyautogui()
        if error:
            return error

        try:
            x, y = pyautogui.position()
            return f"🖱️ Mouse position: ({x}, {y})"
        except Exception as e:
            return f"❌ Error getting mouse position: {str(e)}"

    def type_text(self, text: str, interval: float = 0.05) -> str:
        """Type text at current cursor position"""
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
        """Press a key or key combination"""
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
        """Press a key combination (e.g., 'ctrl+c', 'alt+tab')"""
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

    # ===== INTELLIGENT VISION TOOLS =====

    def smart_screenshot_and_analyze(self, goal: str) -> str:
        """Take an intelligent screenshot based on the goal and analyze it"""
        current_provider = self._get_current_provider()

        if current_provider != 'claude':
            return f"❌ Vision analysis requires Claude provider (current: {current_provider})"

        error = self._check_pyautogui()
        if error:
            return error

        error = self._check_pil()
        if error:
            return error

        try:
            # Intelligently select the best area
            best_area = self._detect_best_area(goal)
            regions = self.get_screen_regions()
            region = regions.get(best_area, regions["full"])

            print(f"🎯 Auto-selected '{best_area}' area for goal: {goal}")

            # Take screenshot of selected region
            screenshot = pyautogui.screenshot(region=region)

            # Optimize aggressively for rate limits
            optimized_data = self._optimize_screenshot(screenshot, max_file_size=60_000)

            # Create concise vision prompt
            vision_prompt = (
                f"GOAL: {goal}\n"
                f"AREA: {best_area} region of screen\n\n"
                "Analyze this screenshot and provide specific guidance. "
                "If clickable elements are visible, provide exact coordinates like (x, y)."
            )

            vision_message = self._create_vision_message(vision_prompt, optimized_data)

            # Save screenshot for reference
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"smart_screenshot_{best_area}_{timestamp}.jpg"
            with open(filename, 'wb') as f:
                f.write(optimized_data)

            # Make vision analysis call
            analysis_result = self._claude_api_call_with_retry(vision_message)

            self._log_action("SMART_VISION", f"area: {best_area}, goal: '{goal[:30]}...'")

            return (f"🎯 Smart Vision Analysis\n"
                    f"Goal: {goal}\n"
                    f"Area: {best_area} ({len(optimized_data):,} bytes)\n"
                    f"Screenshot: {filename}\n\n"
                    f"Analysis: {analysis_result}")

        except Exception as e:
            return f"❌ Error with smart vision: {str(e)}"

    def find_and_click_element(self, description: str) -> str:
        """Find an element using progressive vision and click it"""
        current_provider = self._get_current_provider()

        if current_provider != 'claude':
            return f"❌ Vision required for element finding (current: {current_provider})"

        error = self._check_pyautogui()
        if error:
            return error

        try:
            # Determine search strategy based on description
            if any(word in description.lower() for word in ["start", "search", "taskbar"]):
                areas_to_try = ["taskbar", "start_menu_area"]
            elif any(word in description.lower() for word in ["desktop", "icon"]):
                areas_to_try = ["desktop"]
            else:
                areas_to_try = ["center", "full"]

            regions = self.get_screen_regions()

            for attempt, area in enumerate(areas_to_try):
                print(f"🔍 Searching in {area} area...")

                # Take targeted screenshot
                region = regions.get(area, regions["full"])
                screenshot = pyautogui.screenshot(region=region)
                optimized_data = self._optimize_screenshot(screenshot, max_file_size=50_000)

                # Create focused vision prompt
                vision_prompt = (
                    f"Find: {description}\n"
                    f"Area: {area}\n\n"
                    "Look for this element in the screenshot. If found, respond with:\n"
                    "FOUND: (x, y) coordinates\n"
                    "If not found, respond with: NOT FOUND"
                )

                vision_message = self._create_vision_message(vision_prompt, optimized_data)
                analysis = self._claude_api_call_with_retry(vision_message)

                # Check if element was found
                if "FOUND:" in analysis or "found" in analysis.lower():
                    coords = self._parse_coordinates_from_response(analysis)
                    if coords:
                        x, y = coords
                        # Adjust coordinates if we used a region
                        if area != "full":
                            x += region[0]
                            y += region[1]

                        # Click the element
                        click_result = self.click_mouse(x, y)

                        self._log_action("FIND_AND_CLICK", f"'{description}' at ({x}, {y})")

                        return (f"✅ Found and clicked: {description}\n"
                                f"Location: ({x}, {y}) in {area} area\n"
                                f"Click result: {click_result}\n"
                                f"Analysis: {analysis}")

                if attempt < len(areas_to_try) - 1:
                    print(f"⏭️ Not found in {area}, trying next area...")
                    time.sleep(0.5)

            return f"❌ Could not find element: {description}"

        except Exception as e:
            return f"❌ Error finding element: {str(e)}"

    def complete_task_with_vision(self, task: str, max_steps: int = 3) -> str:
        """Complete a multi-step task using vision guidance"""
        current_provider = self._get_current_provider()

        if current_provider != 'claude':
            return f"❌ Vision required for task completion (current: {current_provider})"

        try:
            results = [f"🎯 Starting task: {task}"]

            for step in range(max_steps):
                results.append(f"\n--- Step {step + 1}/{max_steps} ---")

                # Get vision analysis of current state
                analysis = self.smart_screenshot_and_analyze(
                    f"Step {step + 1} of task: {task}. What should I do next?"
                )
                results.append(f"Vision: {analysis}")

                # Extract action from analysis
                if "click" in analysis.lower():
                    coords = self._parse_coordinates_from_response(analysis)
                    if coords:
                        x, y = coords
                        action_result = self.click_mouse(x, y)
                        results.append(f"Action: Clicked ({x}, {y}) - {action_result}")

                elif "type" in analysis.lower() and ("notepad" in task.lower() or "text" in analysis.lower()):
                    # Extract text to type (simplified)
                    if "notepad" in task.lower():
                        action_result = self.type_text("notepad")
                        results.append(f"Action: Typed 'notepad' - {action_result}")

                elif "press" in analysis.lower() and "enter" in analysis.lower():
                    action_result = self.press_key("enter")
                    results.append(f"Action: Pressed Enter - {action_result}")

                # Check if task appears complete
                if any(word in analysis.lower() for word in ["complete", "done", "success", "opened"]):
                    results.append("✅ Task appears complete!")
                    break

                time.sleep(1)  # Brief pause between steps

            self._log_action("COMPLETE_TASK", f"'{task}', {step + 1} steps")

            return "\n".join(results)

        except Exception as e:
            return f"❌ Error completing task: {str(e)}"

    # ===== UTILITY TOOLS =====

    def get_screen_size(self) -> str:
        """Get screen dimensions"""
        error = self._check_pyautogui()
        if error:
            return error

        try:
            width, height = pyautogui.size()
            return f"🖥️ Screen size: {width} x {height} pixels"
        except Exception as e:
            return f"❌ Error getting screen size: {str(e)}"

    def list_available_areas(self) -> str:
        """List all available screen areas and their purposes"""
        regions = self.get_screen_regions()

        area_descriptions = {
            "full": "Entire screen",
            "taskbar": "Bottom taskbar (start menu, system tray)",
            "start_menu_area": "Start menu and search area",
            "system_tray": "System tray and notifications",
            "desktop": "Desktop area (excluding taskbar)",
            "top_half": "Upper half of screen",
            "bottom_half": "Lower half of screen",
            "left_half": "Left half of screen",
            "right_half": "Right half of screen",
            "center": "Center portion of screen",
            "top_bar": "Top portion (title bars, menus)",
            "search_area": "Central search/dialog area"
        }

        result = "📍 Available Screen Areas:\n"
        for area, (x, y, w, h) in regions.items():
            desc = area_descriptions.get(area, "Custom area")
            result += f"• {area}: {desc} - ({x}, {y}, {w}x{h})\n"

        result += "\n🎯 Context Keywords:\n"
        for area, keywords in self.area_keywords.items():
            result += f"• {area}: {', '.join(keywords[:5])}...\n"

        return result

    def check_dependencies(self) -> str:
        """Check which dependencies are available"""
        status = "🔧 Intelligent Computer Control Dependencies:\n"

        # Check core dependencies
        if PYAUTOGUI_AVAILABLE:
            status += "✅ PyAutoGUI: Available\n"
        else:
            status += "❌ PyAutoGUI: Not installed (pip install pyautogui)\n"

        if PIL_AVAILABLE:
            status += "✅ PIL/Pillow: Available\n"
        else:
            status += "❌ PIL/Pillow: Not installed (pip install Pillow)\n"

        # Platform and provider info
        status += f"🖥️ Platform: {platform.system()}\n"
        status += f"🤖 Current Provider: {self._get_current_provider()}\n"

        # Screen info
        try:
            width, height = pyautogui.size() if PYAUTOGUI_AVAILABLE else (0, 0)
            status += f"📱 Screen: {width}x{height}\n"
        except:
            status += "📱 Screen: Unable to detect\n"

        # Capability status
        if PIL_AVAILABLE and self._get_current_provider() == 'claude':
            status += "🎯 Smart Vision: Available\n"
            status += "🔍 Element Finding: Available\n"
            status += "🤖 Task Automation: Available"
        elif PIL_AVAILABLE:
            status += "🎯 Smart Vision: Needs Claude provider\n"
            status += "🔍 Element Finding: Needs Claude provider\n"
            status += "🤖 Task Automation: Needs Claude provider"
        else:
            status += "🎯 Smart Vision: Needs PIL + Claude\n"
            status += "🔍 Element Finding: Needs PIL + Claude\n"
            status += "🤖 Task Automation: Needs PIL + Claude"

        return status


# Create the plugin instance
def create_plugin():
    """Plugin entry point"""
    return ComputerControlPlugin()


# For direct execution testing
if __name__ == "__main__":
    plugin = ComputerControlPlugin()
    print("Testing intelligent computer control plugin:")
    result = plugin.check_dependencies()
    print(result)