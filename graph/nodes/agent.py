"""
Agent node for workflow execution.
Executes the chat agent with optional tool support.
"""

from typing import Optional, Type
from pydantic import BaseModel

from langchain_core.runnables import RunnableLambda

from tools.registry import ToolRegistry
from agents.chat import ChatAgent
from graph.state import WorkflowState
from constants import AGENT_NODE_NAME, STRUCTURED_AGENT_RUNNABLE_NAME

from models import NodeMetadata, Message, MessageRole
from utils.logging import llmmllogger


class AgentNode:
    """
    Executes the chat agent with optional tool support.

    When tool_registry is provided, tools are passed to the agent for tool-calling.
    When tool_registry is None, the agent runs without tools (passthrough mode).
    """

    def __init__(
        self,
        agent: ChatAgent,
        node_metadata: NodeMetadata,
        tool_registry: Optional[ToolRegistry] = None,
        grammar: Optional[Type[BaseModel]] = None,
    ):
        self.agent = agent.bind_node_metadata(node_metadata)
        self.logger = llmmllogger.bind(component=AGENT_NODE_NAME)
        self.tool_registry = tool_registry
        self.grammar = grammar

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Execute the agent node.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with agent response
        """
        assert state.conversation_id is not None
        try:
            tools = (
                self.tool_registry.get_all_executable_tools()
                if self.tool_registry
                else None
            )

            if self.grammar:
                self.logger.info("Using structured output grammar for agent response")
                structured_response = await self.agent.run_structured(
                    message_input=state.messages,
                    tools=tools,
                    grammar=self.grammar,
                )

                runnable = RunnableLambda(
                    lambda x: x, name=STRUCTURED_AGENT_RUNNABLE_NAME
                )

                self.logger.debug(
                    f"Structured response from agent: {structured_response.model_dump_json(warnings=False)}"
                )

                runnable.invoke(structured_response)

                state.messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=[],
                        structured_output=structured_response.model_dump(
                            warnings=False
                        ),
                    )
                )
            else:
                response = await self.agent.run(
                    messages=state.messages,
                    tools=tools,
                )

                if response.message:
                    if response.message.tool_calls:
                        self.logger.info(
                            f"Generated {len(response.message.tool_calls)} tool calls"
                        )
                    state.messages.append(response.message)

            return state

        except Exception as e:
            self.logger.error(
                "Chat Agent failed",
                extra={
                    "user_id": getattr(state, "user_id", "unknown"),
                    "error": str(e),
                },
            )
            raise
