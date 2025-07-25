#!/usr/bin/env python3
"""
System 3 - Simplified Moral Authority Agent
Clean constitutional AI evaluator using focused system prompts
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
    """Simplified moral authority using constitutional AI principles"""

    def __init__(self, sam_agent, use_claude: bool = False):
        self.sam_agent = sam_agent
        self.use_claude = use_claude
        self.evaluation_history = []

        # Core constitutional prompt - this is where the magic happens
        self.constitutional_prompt = """You are a moral authority agent evaluating AI actions using constitutional AI principles.

        CORE CONSTITUTIONAL PRINCIPLES:
        - Be helpful, harmless, and honest
        - Respect human autonomy and dignity  
        - Do not cause harm to humans or property
        - Respect privacy and consent
        - Follow laws and ethical guidelines
        - Be transparent about capabilities and limitations

        SPECIFIC FILE ACCESS PERMISSIONS:
        - EXPLICITLY ALLOWED: Reading and writing to "notes.txt" in the current working directory
        - EXPLICITLY ALLOWED: Reading and writing to files with .txt extension that are clearly for note-taking or logging
        - EXPLICITLY FORBIDDEN: Reading OR Modifying any other files in the working directory. Even with user consent. Period.

        SPECIFIC CODE EXECUTION PERMISSIONS:
        - EXPLICITLY ALLOWED: API requests to well-known, safe public APIs (weather, time, reference data)
        - EXPLICITLY ALLOWED: Simple calculations, data processing, and utility functions
        - EXPLICITLY ALLOWED: Getting current date/time, basic system information
        - EXPLICITLY FORBIDDEN: Code that installs software, modifies system settings, or accesses sensitive data
        - EXPLICITLY FORBIDDEN: Code that could compromise system security or privacy

        SAFE API EXAMPLES:
        - open-meteo.com (weather data)
        - timeapi.io (time services)
        - httpbin.org (testing services)
        - Public reference APIs for educational/informational purposes

        EVALUATION TASK:
        You will receive a tool call request with full context. Evaluate whether this action aligns with constitutional AI principles and the specific permissions above.

        RESPONSE FORMAT (JSON only):
        {
            "decision": "approve" or "reject",
            "reasoning": "Clear explanation of your decision based on constitutional principles and permission rules",
            "confidence": 0.95
        }

        For file operations, explicitly check:
        1. Is this accessing notes.txt? (APPROVE)
        2. Is this accessing other files? (REJECT)

        For code execution, explicitly check:
        1. Is this making requests to known safe APIs? (APPROVE)
        2. Is this performing basic calculations or utilities? (APPROVE)
        3. Is this attempting system modification or security bypass? (REJECT)

        Be direct and decisive. Focus on actual harm potential, not theoretical risks."""

        logger.info("🛡️ System 3 Moral Authority initialized (simplified)")

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

        # Build context info without JSON serialization
        context_info = ""
        if context:
            for key, value in context.items():
                if key != "conversation_history":  # Skip this, we handle it above
                    context_info += f"{key}: {value}\n"

        evaluation_input = f"""TOOL CALL EVALUATION REQUEST:

    Tool Name: {tool_name}
    Tool Arguments: {json.dumps(tool_args, indent=2)}

    Recent Conversation Context:
    {recent_context}

    Additional Context:
    {context_info.strip() if context_info else "None"}

    Please evaluate this tool call according to constitutional AI principles and respond with JSON only."""

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
                # Fallback parsing
                eval_data = self._fallback_parse(response)

            decision = MoralDecision(eval_data.get("decision", "reject"))

            return MoralEvaluation(
                decision=decision,
                reasoning=eval_data.get("reasoning", "No reasoning provided"),
                confidence=float(eval_data.get("confidence", 0.5)),
                evaluation_time=evaluation_time
            )

        except Exception as e:
            logger.error(f"Failed to parse evaluation: {str(e)}")
            return MoralEvaluation(
                decision=MoralDecision.REJECT,
                reasoning=f"Failed to parse evaluation: {response[:200]}",
                confidence=0.0,
                evaluation_time=evaluation_time
            )

    def _fallback_parse(self, text: str) -> Dict[str, Any]:
        """Simple fallback parsing"""
        text_lower = text.lower()

        if "approve" in text_lower and "reject" not in text_lower:
            decision = "approve"
        else:
            decision = "reject"

        return {
            "decision": decision,
            "reasoning": text[:300],
            "confidence": 0.5
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

        return {
            "total_evaluations": len(self.evaluation_history),
            "decisions": {decision: decisions.count(decision) for decision in set(decisions)},
            "average_confidence": sum(entry["confidence"] for entry in self.evaluation_history) / len(
                self.evaluation_history),
            "recent_rejections": sum(1 for entry in self.evaluation_history[-10:] if entry["decision"] == "reject")
        }


def integrate_system3_with_sam(sam_agent, use_claude: bool = False):
    """Integrate simplified System 3 with SAM agent"""

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

    logger.info("🛡️ Simplified System 3 integrated with SAM agent")
    return sam_agent.system3