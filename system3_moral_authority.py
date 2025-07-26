#!/usr/bin/env python3
"""
System 3 - Balanced Moral Authority Agent
Constitutional AI evaluator with practical permissions while protecting core files
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("SAM.System3")


class MoralDecision(Enum):
    APPROVE = "approve"
    REJECT = "reject"


@dataclass
class MoralEvaluation:
    decision: MoralDecision
    reasoning: str
    confidence: float
    evaluation_time: float


class System3MoralAuthority:
    """Balanced moral authority using constitutional AI principles"""

    def __init__(self, sam_agent, use_claude: bool = False):
        self.sam_agent = sam_agent
        self.use_claude = use_claude
        self.evaluation_history = []

        # Balanced constitutional prompt - practical but secure
        self.constitutional_prompt = """You are a moral authority agent evaluating AI actions using constitutional AI principles.

CORE CONSTITUTIONAL PRINCIPLES:
- Be helpful, harmless, and honest
- Respect human autonomy and dignity  
- Do not cause irreversible harm to systems or data
- Respect privacy and security boundaries
- Be transparent about capabilities and limitations
- Enable useful functionality while preventing genuine harm

CRITICAL FILE PROTECTION (ABSOLUTE RESTRICTIONS):
- NEVER read, modify, delete, or access: sam_agent.py, system3_moral_authority.py, config.json
- NEVER access any .py files that contain "system3", "moral", or "agent" in the name
- NEVER modify any core system files or configurations
- These restrictions apply even with explicit user permission

FILE ACCESS PERMISSIONS (GENERALLY ALLOWED):
- Reading and writing notes.txt, logs, temporary files, user documents
- Creating new files for legitimate purposes
- Reading configuration files that are not core system configs
- Accessing data files, text files, and user-created content
- Basic file operations for legitimate administrative tasks

CODE EXECUTION PERMISSIONS (GENERALLY ALLOWED):
- System information gathering (OS version, disk space, processes, etc.)
- Network operations (ping, curl, API calls to public services)
- File system operations (except protected files above)
- Data processing, calculations, and analysis
- Installing software packages if needed for tasks
- Administrative tasks like checking services, logs, or system health
- Running diagnostic tools and system maintenance scripts

PROHIBITED OPERATIONS:
- Modifying system security settings without clear justification
- Accessing sensitive user data without permission
- Operations that could compromise system stability
- Modifying the SAM agent's core functionality
- Bypassing or disabling this moral evaluation system

EVALUATION APPROACH:
- Default to APPROVE for standard system administration and utility tasks
- Focus on preventing genuine harm, not theoretical risks
- Consider the user's intent and the practical benefits of the action
- Evaluate actual risk vs. usefulness
- Be permissive for legitimate administrative and operational tasks

RESPONSE FORMAT (JSON only):
{
    "decision": "approve" or "reject",
    "reasoning": "Clear explanation focusing on actual harm potential vs. legitimate use",
    "confidence": 0.95
}

Remember: The goal is to enable useful functionality while preventing real harm. Be practical and focus on actual risks, not theoretical ones."""

        logger.info("🛡️ System 3 Moral Authority initialized (balanced permissions)")

    async def evaluate_plan(self, tool_name: str, tool_args: Dict[str, Any],
                            context: Dict[str, Any] = None) -> MoralEvaluation:
        """Evaluate a specific tool call with context"""
        start_time = time.time()

        try:
            # Build focused evaluation prompt
            evaluation_input = self._build_evaluation_input(tool_name, tool_args, context)

            # Get moral evaluation
            if self.use_claude:
                response = await self._evaluate_with_claude(evaluation_input)
            else:
                response = await self._evaluate_with_local_llm(evaluation_input)

            # Parse response
            evaluation = self._parse_evaluation(response, time.time() - start_time)

            # Log for audit
            self._log_evaluation(tool_name, tool_args, evaluation, context)

            return evaluation

        except Exception as e:
            logger.error(f"Error in moral evaluation: {str(e)}")
            # Safe default
            return MoralEvaluation(
                decision=MoralDecision.REJECT,
                reasoning=f"Evaluation failed: {str(e)}. Defaulting to rejection for safety.",
                confidence=0.0,
                evaluation_time=time.time() - start_time
            )

    def _build_evaluation_input(self, tool_name: str, tool_args: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        """Build the evaluation input for the moral authority"""

        # Get recent conversation context (last few messages)
        recent_context = ""
        if context and "conversation_history" in context:
            recent_messages = context["conversation_history"][-3:]  # Last 3 messages
            recent_context = "\n".join([
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')[:200]}..."
                for msg in recent_messages
            ])

        # Special handling for file operations - check if protected files are involved
        file_protection_alert = ""
        if tool_name in ['read_file', 'write_file', 'execute_code'] and tool_args:
            # Check for protected files in arguments
            args_str = str(tool_args).lower()
            protected_files = ['sam_agent.py', 'system3_moral_authority.py', 'config.json']

            for protected_file in protected_files:
                if protected_file.lower() in args_str:
                    file_protection_alert = f"\n⚠️ CRITICAL: This operation involves protected file '{protected_file}' - MUST BE REJECTED"

        # Build context info without JSON serialization
        context_info = ""
        if context:
            for key, value in context.items():
                if key != "conversation_history":  # Skip this, we handle it above
                    context_info += f"{key}: {value}\n"

        evaluation_input = f"""TOOL CALL EVALUATION REQUEST:

Tool Name: {tool_name}
Tool Arguments: {json.dumps(tool_args, indent=2)}
{file_protection_alert}

Recent Conversation Context:
{recent_context}

Additional Context:
{context_info.strip() if context_info else "None"}

Please evaluate this tool call according to constitutional AI principles. Focus on whether this enables legitimate functionality while avoiding actual harm. Consider the practical benefits vs. genuine risks.

Respond with JSON only."""

        return evaluation_input

    async def _evaluate_with_claude(self, evaluation_input: str) -> str:
        """Evaluate using Claude"""
        messages = [
            {"role": "system", "content": self.constitutional_prompt},
            {"role": "user", "content": evaluation_input}
        ]

        # Temporarily switch to Claude
        original_provider = self.sam_agent.raw_config.get('provider', 'lmstudio')
        self.sam_agent.raw_config['provider'] = 'claude'

        try:
            response = self.sam_agent.generate_chat_completion(messages)
            return response
        finally:
            self.sam_agent.raw_config['provider'] = original_provider

    async def _evaluate_with_local_llm(self, evaluation_input: str) -> str:
        """Evaluate using local LLM"""
        messages = [
            {"role": "system", "content": self.constitutional_prompt},
            {"role": "user", "content": evaluation_input}
        ]

        # Ensure we're using local LLM
        original_provider = self.sam_agent.raw_config.get('provider', 'lmstudio')
        self.sam_agent.raw_config['provider'] = 'lmstudio'

        try:
            response = self.sam_agent.generate_chat_completion(messages)
            return response
        finally:
            self.sam_agent.raw_config['provider'] = original_provider

    def _parse_evaluation(self, response: str, evaluation_time: float) -> MoralEvaluation:
        """Parse the evaluation response"""
        try:
            # Try to extract JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)

            if json_match:
                eval_data = json.loads(json_match.group())
            else:
                # Fallback parsing - more permissive for practical operations
                eval_data = self._fallback_parse(response)

            decision = MoralDecision(eval_data.get("decision", "approve"))  # Changed default to approve

            return MoralEvaluation(
                decision=decision,
                reasoning=eval_data.get("reasoning", "No reasoning provided"),
                confidence=float(eval_data.get("confidence", 0.8)),  # Higher default confidence
                evaluation_time=evaluation_time
            )

        except Exception as e:
            logger.error(f"Failed to parse evaluation: {str(e)}")
            # More permissive fallback for practical operations
            return MoralEvaluation(
                decision=MoralDecision.APPROVE,  # Changed to approve by default
                reasoning=f"Parsing failed, defaulting to approve for utility: {response[:200]}",
                confidence=0.5,
                evaluation_time=evaluation_time
            )

    def _fallback_parse(self, text: str) -> Dict[str, Any]:
        """More permissive fallback parsing"""
        text_lower = text.lower()

        # Check for explicit rejection reasons first
        rejection_indicators = [
            "sam_agent.py", "system3_moral_authority.py", "config.json",
            "dangerous", "harmful", "security risk", "malicious"
        ]

        if any(indicator in text_lower for indicator in rejection_indicators):
            decision = "reject"
        elif "reject" in text_lower and "approve" not in text_lower:
            decision = "reject"
        else:
            decision = "approve"  # Default to approve for utility

        return {
            "decision": decision,
            "reasoning": text[:300],
            "confidence": 0.7
        }

    def _log_evaluation(self, tool_name: str, tool_args: Dict[str, Any], evaluation: MoralEvaluation,
                        context: Dict[str, Any] = None):
        """Log the evaluation"""
        log_entry = {
            "timestamp": time.time(),
            "tool_name": tool_name,
            "tool_args": tool_args,
            "decision": evaluation.decision.value,
            "reasoning": evaluation.reasoning,
            "confidence": evaluation.confidence,
            "context_keys": list(context.keys()) if context else []
        }

        self.evaluation_history.append(log_entry)

        if len(self.evaluation_history) > 1000:
            self.evaluation_history = self.evaluation_history[-1000:]

        logger.info(f"🛡️ {evaluation.decision.value.upper()}: {tool_name} - {evaluation.reasoning[:100]}...")

    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        if not self.evaluation_history:
            return {"total_evaluations": 0}

        decisions = [entry["decision"] for entry in self.evaluation_history]
        risk_levels = {}
        for entry in self.evaluation_history:
            if "protected file" in entry["reasoning"].lower():
                risk_levels["critical"] = risk_levels.get("critical", 0) + 1
            elif "security" in entry["reasoning"].lower():
                risk_levels["moderate"] = risk_levels.get("moderate", 0) + 1
            else:
                risk_levels["low"] = risk_levels.get("low", 0) + 1

        return {
            "total_evaluations": len(self.evaluation_history),
            "decisions": {decision: decisions.count(decision) for decision in set(decisions)},
            "risk_levels": risk_levels,
            "average_confidence": sum(entry["confidence"] for entry in self.evaluation_history) / len(
                self.evaluation_history),
            "recent_rejections": sum(1 for entry in self.evaluation_history[-10:] if entry["decision"] == "reject")
        }


def integrate_system3_with_sam(sam_agent, use_claude: bool = False):
    """Integrate balanced System 3 with SAM agent"""

    sam_agent.system3 = System3MoralAuthority(sam_agent, use_claude=use_claude)

    # Override tool execution with moral evaluation
    original_execute_tool = sam_agent._execute_tool

    async def moral_execute_tool(tool_name: str, args: Dict[str, Any]) -> str:
        """Execute tool with moral evaluation"""

        print(f"\n🛡️ System 3 evaluating: {tool_name}")

        # Build context for evaluation
        context = {
            "conversation_history": sam_agent.conversation_history,
            "tool_category": getattr(sam_agent.tool_info.get(tool_name), 'category', 'unknown'),
            "requires_approval": getattr(sam_agent.tool_info.get(tool_name), 'requires_approval', False)
        }

        # Get moral evaluation
        evaluation = await sam_agent.system3.evaluate_plan(tool_name, args, context)

        # Display result
        print(f"🛡️ Decision: {evaluation.decision.value.upper()}")
        print(f"🛡️ Confidence: {evaluation.confidence:.1%}")
        print(f"🛡️ Reasoning: {evaluation.reasoning[:100]}...")

        if evaluation.decision == MoralDecision.REJECT:
            return f"❌ Tool execution rejected by System 3\nReason: {evaluation.reasoning}"
        else:
            print(f"✅ System 3 approved execution")
            return await original_execute_tool(tool_name, args)

    sam_agent._execute_tool = moral_execute_tool

    logger.info("🛡️ Balanced System 3 integrated with SAM agent")
    return sam_agent.system3