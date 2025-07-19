#!/usr/bin/env python3
"""
System 2 Tools Plugin for SAM Agent
Metacognitive tools for context management, loop detection, and strategic intervention
These tools are exclusively available to the System 2 metacognitive agent
"""

import json
import logging
import time
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from sam_agent import SAMPlugin, ToolCategory

logger = logging.getLogger("SAM.System2Tools")


class System2ToolsPlugin(SAMPlugin):
    """Metacognitive tools exclusively for System 2 agent"""

    def __init__(self):
        super().__init__(
            name="System 2 Metacognitive Tools",
            version="1.0.0",
            description="Advanced context management and loop detection tools for System 2"
        )
        self.restricted = True  # Flag to prevent System 1 access
        self.intervention_log = []

    def register_tools(self, agent):
        """Register System 2 exclusive tools"""
        # NOTE: These tools are NOT registered with System 1's normal tool registry
        # Instead, they're registered with System 2's exclusive registry
        if hasattr(agent, 'register_system2_tool'):
            agent.register_system2_tool(
                self.analyze_conversation_patterns,
                category=ToolCategory.UTILITY,
                requires_approval=False
            )

            agent.register_system2_tool(
                self.compress_conversation_segment,
                category=ToolCategory.UTILITY,
                requires_approval=False
            )

            agent.register_system2_tool(
                self.identify_critical_information,
                category=ToolCategory.UTILITY,
                requires_approval=False
            )

            agent.register_system2_tool(
                self.inject_metacognitive_guidance,
                category=ToolCategory.UTILITY,
                requires_approval=False
            )

            agent.register_system2_tool(
                self.detect_tool_usage_patterns,
                category=ToolCategory.UTILITY,
                requires_approval=False
            )

            agent.register_system2_tool(
                self.assess_progress_metrics,
                category=ToolCategory.UTILITY,
                requires_approval=False
            )

            agent.register_system2_tool(
                self.generate_strategic_summary,
                category=ToolCategory.UTILITY,
                requires_approval=False
            )

            logger.info("System 2 tools registered successfully")
        else:
            logger.warning("System 2 tool registration not available - falling back to regular registration")
            # Fallback to regular registration if System 2 isn't fully implemented yet
            self._register_fallback_tools(agent)

    def _register_fallback_tools(self, agent):
        """Fallback registration for testing purposes"""
        agent.register_local_tool(
            self.analyze_conversation_patterns,
            category=ToolCategory.UTILITY,
            requires_approval=False
        )

    # ===== CONTEXT ANALYSIS TOOLS =====

    def analyze_conversation_patterns(self, conversation_history: List[Dict]) -> str:
        """
        Analyze conversation for redundancy, loops, and efficiency patterns

        Args:
            conversation_history: List of conversation messages

        Returns:
            Analysis report of conversation patterns
        """
        try:
            analysis = {
                "total_messages": len(conversation_history),
                "message_breakdown": {"user": 0, "assistant": 0, "system": 0},
                "tool_calls_detected": 0,
                "repeated_phrases": [],
                "conversation_topics": [],
                "efficiency_score": 0.0
            }

            # Basic message analysis
            repeated_phrases = {}
            tool_pattern = re.compile(r'```json\s*\{.*?"name".*?\}', re.DOTALL)

            for msg in conversation_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                if role in analysis["message_breakdown"]:
                    analysis["message_breakdown"][role] += 1

                # Detect tool calls
                if tool_pattern.search(content):
                    analysis["tool_calls_detected"] += 1

                # Find repeated phrases (simple approach)
                words = content.lower().split()
                for i in range(len(words) - 2):
                    phrase = " ".join(words[i:i + 3])
                    if len(phrase) > 10:  # Only meaningful phrases
                        repeated_phrases[phrase] = repeated_phrases.get(phrase, 0) + 1

            # Identify highly repeated phrases
            analysis["repeated_phrases"] = [
                                               {"phrase": phrase, "count": count}
                                               for phrase, count in repeated_phrases.items()
                                               if count >= 3
                                           ][:5]  # Top 5

            # Calculate efficiency score
            if analysis["total_messages"] > 0:
                tool_ratio = analysis["tool_calls_detected"] / analysis["total_messages"]
                repetition_penalty = len(analysis["repeated_phrases"]) * 0.1
                analysis["efficiency_score"] = max(0, tool_ratio - repetition_penalty)

            # Generate report
            report = f"🧠 **CONVERSATION PATTERN ANALYSIS**\n\n"
            report += f"📊 **Message Statistics:**\n"
            report += f"• Total messages: {analysis['total_messages']}\n"
            report += f"• User: {analysis['message_breakdown']['user']}\n"
            report += f"• Assistant: {analysis['message_breakdown']['assistant']}\n"
            report += f"• System: {analysis['message_breakdown']['system']}\n"
            report += f"• Tool calls detected: {analysis['tool_calls_detected']}\n\n"

            if analysis["repeated_phrases"]:
                report += f"🔄 **Repeated Patterns:**\n"
                for item in analysis["repeated_phrases"]:
                    report += f"• \"{item['phrase'][:50]}...\" ({item['count']} times)\n"
                report += "\n"

            report += f"⚡ **Efficiency Score:** {analysis['efficiency_score']:.2f}\n"

            if analysis['efficiency_score'] < 0.3:
                report += "⚠️ **Recommendation:** Low efficiency detected - consider context compression\n"
            elif len(analysis["repeated_phrases"]) > 3:
                report += "⚠️ **Recommendation:** High repetition detected - consider loop breaking\n"
            else:
                report += "✅ **Status:** Conversation efficiency appears normal\n"

            return report

        except Exception as e:
            return f"❌ Error analyzing conversation patterns: {str(e)}"

    def compress_conversation_segment(self, messages: List[Dict], target_reduction: float = 0.7) -> str:
        """
        Intelligently compress a segment of conversation while preserving key information

        Args:
            messages: List of messages to compress
            target_reduction: Target reduction ratio (0.7 = reduce to 30% of original)

        Returns:
            Compressed summary of the conversation segment
        """
        try:
            if not messages:
                return "No messages to compress"

            # Extract key information
            user_requests = []
            assistant_responses = []
            tool_executions = []

            for msg in messages:
                content = msg.get("content", "")
                role = msg.get("role", "")

                if role == "user":
                    if not content.startswith("Here are the results"):
                        user_requests.append(content[:200])  # First 200 chars
                elif role == "assistant":
                    if "```json" in content:
                        # Extract tool calls
                        tool_matches = re.findall(r'"name":\s*"([^"]+)"', content)
                        if tool_matches:
                            tool_executions.extend(tool_matches)
                    else:
                        assistant_responses.append(content[:150])  # First 150 chars

            # Create intelligent summary
            summary_parts = []

            if user_requests:
                summary_parts.append(f"User requested: {'; '.join(user_requests[:3])}")

            if tool_executions:
                tool_counts = {}
                for tool in tool_executions:
                    tool_counts[tool] = tool_counts.get(tool, 0) + 1
                tool_summary = ", ".join([f"{tool}({count})" for tool, count in tool_counts.items()])
                summary_parts.append(f"Tools used: {tool_summary}")

            if assistant_responses:
                summary_parts.append(f"Assistant provided: {assistant_responses[0]}")

            if not summary_parts:
                summary_parts.append("Various exchanges and tool executions")

            compressed_summary = f"[COMPRESSED SEGMENT: {'. '.join(summary_parts)}]"

            original_length = sum(len(msg.get("content", "")) for msg in messages)
            compressed_length = len(compressed_summary)
            reduction_achieved = 1 - (compressed_length / original_length) if original_length > 0 else 0

            result = f"✅ Compressed {len(messages)} messages\n"
            result += f"📊 Reduction: {reduction_achieved:.1%} (target: {target_reduction:.1%})\n"
            result += f"📝 Summary: {compressed_summary}"

            return result

        except Exception as e:
            return f"❌ Error compressing conversation segment: {str(e)}"

    def identify_critical_information(self, conversation_history: List[Dict]) -> str:
        """
        Identify critical information that must be preserved during compression

        Args:
            conversation_history: Full conversation history

        Returns:
            List of critical information to preserve
        """
        try:
            critical_info = {
                "user_goals": [],
                "successful_tools": [],
                "important_results": [],
                "error_patterns": [],
                "context_references": []
            }

            for i, msg in enumerate(conversation_history):
                content = msg.get("content", "")
                role = msg.get("role", "")

                # Identify user goals (questions, requests)
                if role == "user" and not content.startswith("Here are the results"):
                    if any(word in content.lower() for word in ["please", "can you", "i need", "help me"]):
                        critical_info["user_goals"].append({
                            "message_index": i,
                            "goal": content[:100],
                            "priority": "high"
                        })

                # Track successful tool executions
                if "successfully" in content.lower() or "✅" in content:
                    if "Tool" in content:
                        tool_match = re.search(r"Tool '([^']+)'", content)
                        if tool_match:
                            critical_info["successful_tools"].append({
                                "tool": tool_match.group(1),
                                "message_index": i,
                                "result_preview": content[:100]
                            })

                # Identify important results
                if any(indicator in content.lower() for indicator in ["found", "discovered", "result:", "output:"]):
                    critical_info["important_results"].append({
                        "message_index": i,
                        "content_preview": content[:100],
                        "importance": "medium"
                    })

                # Track error patterns
                if any(error_word in content.lower() for error_word in ["error", "failed", "❌"]):
                    critical_info["error_patterns"].append({
                        "message_index": i,
                        "error_preview": content[:100]
                    })

            # Generate critical information report
            report = "🎯 **CRITICAL INFORMATION IDENTIFICATION**\n\n"

            if critical_info["user_goals"]:
                report += f"🎯 **User Goals ({len(critical_info['user_goals'])}):**\n"
                for goal in critical_info["user_goals"][:3]:
                    report += f"• [{goal['message_index']}] {goal['goal']}\n"
                report += "\n"

            if critical_info["successful_tools"]:
                report += f"✅ **Successful Tools ({len(critical_info['successful_tools'])}):**\n"
                for tool in critical_info["successful_tools"][:5]:
                    report += f"• {tool['tool']} (msg {tool['message_index']})\n"
                report += "\n"

            if critical_info["error_patterns"]:
                report += f"❌ **Error Patterns ({len(critical_info['error_patterns'])}):**\n"
                for error in critical_info["error_patterns"][:3]:
                    report += f"• [{error['message_index']}] {error['error_preview']}\n"
                report += "\n"

            preservation_count = (len(critical_info["user_goals"]) +
                                  len(critical_info["successful_tools"]) +
                                  len(critical_info["important_results"]))

            report += f"📋 **Summary:** {preservation_count} critical items identified for preservation"

            return report

        except Exception as e:
            return f"❌ Error identifying critical information: {str(e)}"

    # ===== LOOP DETECTION AND BREAKING TOOLS =====

    def detect_tool_usage_patterns(self, recent_tools: List[str], threshold: int = 3) -> str:
        """
        Detect problematic tool usage patterns

        Args:
            recent_tools: List of recently used tool names
            threshold: Number of repetitions to consider problematic

        Returns:
            Pattern analysis and recommendations
        """
        try:
            if not recent_tools:
                return "No recent tool usage to analyze"

            # Count tool frequencies
            tool_counts = {}
            for tool in recent_tools:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

            # Detect sequential repetitions
            sequential_patterns = []
            current_tool = None
            current_count = 0

            for tool in recent_tools:
                if tool == current_tool:
                    current_count += 1
                else:
                    if current_count >= threshold:
                        sequential_patterns.append({
                            "tool": current_tool,
                            "consecutive_count": current_count
                        })
                    current_tool = tool
                    current_count = 1

            # Check final sequence
            if current_count >= threshold:
                sequential_patterns.append({
                    "tool": current_tool,
                    "consecutive_count": current_count
                })

            # Detect alternating patterns
            alternating_patterns = []
            if len(recent_tools) >= 6:
                for i in range(len(recent_tools) - 3):
                    pattern = recent_tools[i:i + 4]
                    if pattern[0] == pattern[2] and pattern[1] == pattern[3]:
                        alternating_patterns.append({
                            "pattern": f"{pattern[0]} ↔ {pattern[1]}",
                            "position": i
                        })

            # Generate report
            report = "🔍 **TOOL USAGE PATTERN ANALYSIS**\n\n"
            report += f"📊 **Recent Tools:** {', '.join(recent_tools[-10:])}\n\n"

            # Overall frequency analysis
            frequent_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            report += f"🔢 **Tool Frequency:**\n"
            for tool, count in frequent_tools:
                report += f"• {tool}: {count} times\n"
            report += "\n"

            # Sequential repetition analysis
            if sequential_patterns:
                report += f"🔄 **Sequential Repetitions Detected:**\n"
                for pattern in sequential_patterns:
                    report += f"• {pattern['tool']}: {pattern['consecutive_count']} consecutive uses\n"
                report += "⚠️ **Recommendation:** Break tool loop with alternative approach\n\n"

            # Alternating pattern analysis
            if alternating_patterns:
                report += f"↔️ **Alternating Patterns Detected:**\n"
                for pattern in alternating_patterns[:3]:
                    report += f"• {pattern['pattern']} (position {pattern['position']})\n"
                report += "⚠️ **Recommendation:** Try different tool combination\n\n"

            # Overall assessment
            if sequential_patterns or alternating_patterns:
                report += "🚨 **Status:** Problematic patterns detected - intervention recommended"
            elif max(tool_counts.values()) > len(recent_tools) * 0.6:
                report += "⚠️ **Status:** High tool repetition - consider diversification"
            else:
                report += "✅ **Status:** Tool usage patterns appear healthy"

            return report

        except Exception as e:
            return f"❌ Error detecting tool patterns: {str(e)}"

    def inject_metacognitive_guidance(self, intervention_type: str, context: Dict[str, Any]) -> str:
        """
        Generate metacognitive guidance message for System 1

        Args:
            intervention_type: Type of intervention needed
            context: Additional context for guidance generation

        Returns:
            Formatted guidance message for injection into conversation
        """
        try:
            guidance_templates = {
                "loop_breaking": [
                    "You've been using the same approach repeatedly. Try a different strategy or ask the user for clarification.",
                    "The current tool pattern isn't making progress. Consider breaking down the task differently.",
                    "Multiple attempts with the same tool suggest you need a new approach. Step back and reassess."
                ],
                "stagnation": [
                    "You've executed many tools but progress is unclear. Summarize what you've learned so far.",
                    "Consider asking the user for clarification or feedback on your current approach.",
                    "You may be overcomplicating the task. Try a simpler, more direct approach."
                ],
                "token_management": [
                    "Context is getting full. Focus on the most important information going forward.",
                    "Summarize previous progress before continuing with new tool executions.",
                    "Prioritize essential tools only - avoid exploratory or redundant tool calls."
                ],
                "error_recovery": [
                    "Multiple tool errors detected. Verify your approach and parameters before continuing.",
                    "Consider using simpler, more reliable tools instead of complex ones.",
                    "Break down the current task into smaller, more manageable steps."
                ]
            }

            # Select appropriate guidance
            guidance_options = guidance_templates.get(intervention_type, ["Consider a different approach."])

            # Add context-specific elements
            context_info = ""
            if context.get("consecutive_tools"):
                context_info += f" (Tool '{context['consecutive_tools']}' used {context.get('count', 0)} times)"
            if context.get("error_rate"):
                context_info += f" (Error rate: {context['error_rate']:.1%})"

            # Format the guidance message
            guidance_text = guidance_options[0] + context_info

            guidance_message = f"""<metacognitive_guidance>
🧠 System 2 Intervention ({intervention_type}):

{guidance_text}

This guidance is being provided because your current execution pattern suggests intervention is needed. Please adjust your approach accordingly.
</metacognitive_guidance>"""

            # Log the intervention
            self.intervention_log.append({
                "timestamp": time.time(),
                "type": intervention_type,
                "guidance": guidance_text,
                "context": context
            })

            return guidance_message

        except Exception as e:
            return f"❌ Error generating metacognitive guidance: {str(e)}"

    # ===== STRATEGIC ASSESSMENT TOOLS =====

    def assess_progress_metrics(self, system1_state: Dict[str, Any]) -> str:
        """
        Assess System 1's progress toward the user's goals

        Args:
            system1_state: Current state metrics from System 1

        Returns:
            Progress assessment and recommendations
        """
        try:
            assessment = {
                "progress_score": 0.0,
                "efficiency_rating": "unknown",
                "bottlenecks": [],
                "recommendations": []
            }

            # Calculate progress score based on multiple factors
            factors = []

            # Tool success rate
            if system1_state.get("total_tool_calls", 0) > 0:
                success_rate = 1 - system1_state.get("recent_error_rate", 0)
                factors.append(success_rate * 0.3)  # 30% weight

                if success_rate < 0.7:
                    assessment["bottlenecks"].append("High tool error rate")

            # Stagnation penalty
            stagnation_factor = max(0, 1 - (system1_state.get("tools_since_progress", 0) / 10))
            factors.append(stagnation_factor * 0.4)  # 40% weight

            if system1_state.get("tools_since_progress", 0) > 5:
                assessment["bottlenecks"].append("Tools not producing measurable progress")

            # Loop penalty
            consecutive_tools = system1_state.get("consecutive_identical_tools", 0)
            loop_factor = max(0, 1 - (consecutive_tools / 8))
            factors.append(loop_factor * 0.3)  # 30% weight

            if consecutive_tools > 3:
                assessment["bottlenecks"].append("Repetitive tool usage pattern")

            # Calculate overall progress score
            if factors:
                assessment["progress_score"] = sum(factors) / len(factors)

            # Determine efficiency rating
            if assessment["progress_score"] > 0.8:
                assessment["efficiency_rating"] = "excellent"
            elif assessment["progress_score"] > 0.6:
                assessment["efficiency_rating"] = "good"
            elif assessment["progress_score"] > 0.4:
                assessment["efficiency_rating"] = "fair"
            else:
                assessment["efficiency_rating"] = "poor"

            # Generate recommendations
            if assessment["progress_score"] < 0.5:
                assessment["recommendations"].append("Consider asking user for clarification")
                assessment["recommendations"].append("Simplify the current approach")

            if consecutive_tools > 2:
                assessment["recommendations"].append("Try alternative tools or methods")

            if system1_state.get("recent_error_rate", 0) > 0.3:
                assessment["recommendations"].append("Focus on more reliable tools")

            # Generate report
            report = "📈 **PROGRESS ASSESSMENT REPORT**\n\n"
            report += f"🎯 **Progress Score:** {assessment['progress_score']:.2f}/1.0\n"
            report += f"⚡ **Efficiency Rating:** {assessment['efficiency_rating'].upper()}\n\n"

            if assessment["bottlenecks"]:
                report += f"🚧 **Identified Bottlenecks:**\n"
                for bottleneck in assessment["bottlenecks"]:
                    report += f"• {bottleneck}\n"
                report += "\n"

            if assessment["recommendations"]:
                report += f"💡 **Recommendations:**\n"
                for rec in assessment["recommendations"]:
                    report += f"• {rec}\n"
                report += "\n"

            # System 1 state details
            report += f"📊 **Current Metrics:**\n"
            report += f"• Total tool calls: {system1_state.get('total_tool_calls', 0)}\n"
            report += f"• Consecutive identical tools: {consecutive_tools}\n"
            report += f"• Tools since progress: {system1_state.get('tools_since_progress', 0)}\n"
            report += f"• Recent error rate: {system1_state.get('recent_error_rate', 0):.1%}\n"
            report += f"• Token usage: {system1_state.get('token_usage_percent', 0):.1%}"

            return report

        except Exception as e:
            return f"❌ Error assessing progress metrics: {str(e)}"

    def generate_strategic_summary(self, conversation_history: List[Dict], focus: str = "comprehensive") -> str:
        """
        Generate a high-level strategic summary of the session

        Args:
            conversation_history: Full conversation history
            focus: Summary focus ("comprehensive", "goals", "tools", "outcomes")

        Returns:
            Strategic summary of the session
        """
        try:
            summary_data = {
                "session_start": datetime.now().isoformat(),
                "total_exchanges": len(conversation_history),
                "user_goals": [],
                "tools_used": [],
                "outcomes_achieved": [],
                "unresolved_issues": []
            }

            # Extract information based on focus
            for i, msg in enumerate(conversation_history):
                content = msg.get("content", "")
                role = msg.get("role", "")

                if focus in ["comprehensive", "goals"] and role == "user":
                    if not content.startswith("Here are the results"):
                        # This is likely a user request/goal
                        summary_data["user_goals"].append({
                            "index": i,
                            "request": content[:100] + "..." if len(content) > 100 else content
                        })

                if focus in ["comprehensive", "tools"]:
                    # Extract tool usage
                    tool_matches = re.findall(r'"name":\s*"([^"]+)"', content)
                    for tool in tool_matches:
                        if tool not in summary_data["tools_used"]:
                            summary_data["tools_used"].append(tool)

                if focus in ["comprehensive", "outcomes"]:
                    # Look for successful outcomes
                    if "successfully" in content.lower() or "completed" in content.lower():
                        summary_data["outcomes_achieved"].append({
                            "index": i,
                            "outcome": content[:80] + "..." if len(content) > 80 else content
                        })

                    # Look for unresolved issues
                    if "error" in content.lower() or "failed" in content.lower():
                        summary_data["unresolved_issues"].append({
                            "index": i,
                            "issue": content[:80] + "..." if len(content) > 80 else content
                        })

            # Generate summary based on focus
            report = f"📋 **STRATEGIC SESSION SUMMARY** ({focus.upper()})\n\n"

            if focus in ["comprehensive", "goals"]:
                report += f"🎯 **User Goals ({len(summary_data['user_goals'])}):**\n"
                for goal in summary_data["user_goals"][:5]:
                    report += f"• [{goal['index']}] {goal['request']}\n"
                report += "\n"

            if focus in ["comprehensive", "tools"]:
                report += f"🔧 **Tools Utilized ({len(summary_data['tools_used'])}):**\n"
                for tool in summary_data["tools_used"][:10]:
                    report += f"• {tool}\n"
                report += "\n"

            if focus in ["comprehensive", "outcomes"]:
                report += f"✅ **Outcomes Achieved ({len(summary_data['outcomes_achieved'])}):**\n"
                for outcome in summary_data["outcomes_achieved"][:5]:
                    report += f"• [{outcome['index']}] {outcome['outcome']}\n"
                report += "\n"

                if summary_data["unresolved_issues"]:
                    report += f"❌ **Unresolved Issues ({len(summary_data['unresolved_issues'])}):**\n"
                    for issue in summary_data["unresolved_issues"][:3]:
                        report += f"• [{issue['index']}] {issue['issue']}\n"
                    report += "\n"

            # Overall session metrics
            report += f"📊 **Session Metrics:**\n"
            report += f"• Total exchanges: {summary_data['total_exchanges']}\n"
            report += f"• Unique tools used: {len(summary_data['tools_used'])}\n"
            report += f"• Successful outcomes: {len(summary_data['outcomes_achieved'])}\n"
            report += f"• Issues encountered: {len(summary_data['unresolved_issues'])}\n"

            # Calculate success rate
            if summary_data['user_goals']:
                success_rate = len(summary_data['outcomes_achieved']) / len(summary_data['user_goals'])
                report += f"• Estimated success rate: {success_rate:.1%}"

            return report

        except Exception as e:
            return f"❌ Error generating strategic summary: {str(e)}"


# Create the plugin instance
def create_plugin():
    """Factory function to create the plugin instance"""
    return System2ToolsPlugin()


# For testing
if __name__ == "__main__":
    plugin = System2ToolsPlugin()
    print(f"System 2 Tools Plugin: {plugin.name} v{plugin.version}")
    print(f"Description: {plugin.description}")
    print(f"Restricted access: {plugin.restricted}")