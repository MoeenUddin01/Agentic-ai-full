# DeepAgent Architecture

## Overview

DeepAgent is an agent framework built on LangGraph that provides a structured approach to creating AI agents with tool-use capabilities. It uses a state-graph architecture where each node represents a stage in the agent's reasoning loop.

## Core Architecture

```
User Input
    │
    ▼
┌─────────────────┐
│   Call Model    │  ◄── LLM (e.g., Groq Llama 3.3 70B)
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│  Tool Execution  │  ◄── Tool Node (runs tools like web_search)
└────────┬─────────┘
         │
         ▼
   ┌──────────┐
   │  Decide   │  ◄── Should we continue or end?
   └────┬─────┘
        │
   ┌────┴────┐
   │  End    │  ◄── Returns final response
   └─────────┘
```

## State Graph

DeepAgent uses a cyclic state graph with three main nodes:

1. **call_model** — Invokes the LLM with current conversation state and tool definitions.
2. **tool_node** — Executes any tool calls returned by the LLM. Tools are registered at agent creation time.
3. **decide** — Conditional edge that checks if the LLM requested more tool calls or is ready to respond.

## Key Components

### LLM Integration
- Supports any LangChain-compatible chat model.
- In this project: Groq's `llama-3.3-70b-versatile` via `init_chat_model`.
- Model handles reasoning, tool selection, and response generation.

### Tool System
- Tools are decorated with `@tool` from `langchain_core.tools`.
- Each tool has a name, description, and typed parameters — the LLM uses these to decide when and how to call tools.
- Tools execute in a dedicated tool node with results fed back into the model.

### Agent State
- Maintains conversation history (messages).
- Tracks tool call results.
- Stores the current reasoning path.

## Agent Lifecycle

1. **Initialization**: `create_deep_agent(model, tools, system_prompt)` sets up the state graph.
2. **Invocation**: Agent is called with a user message.
3. **Reasoning Loop**:
   - Model generates a response (text or tool calls).
   - If tool calls: execute tools → feed results back → loop.
   - If final answer: return response.
4. **Completion**: Agent returns structured output with the final answer and trace.

## Comparison: Basic Agent vs DeepAgent

| Feature | Basic LangChain Agent | DeepAgent |
|---------|----------------------|-----------|
| Architecture | Simple LLM call | State graph with loops |
| Tool Use | Linear (call → respond) | Iterative (plan → act → observe → repeat) |
| Multi-step Reasoning | Limited | Full reasoning loop |
| State Management | Stateless | Stateful graph |
| Use Case | Simple Q&A | Complex multi-tool research tasks |

## Best Practices

- **System Prompt**: Be specific about the agent's role and capabilities.
- **Tools**: Keep tool descriptions clear and parameters well-typed for reliable LLM tool selection.
- **Model Choice**: Use models with strong tool-calling support (Llama 3.3, GPT-4, Claude).
- **Error Handling**: Tools should handle failures gracefully and return meaningful error messages to the LLM.

## References

- LangGraph: https://langchain-ai.github.io/langgraph/
- DeepAgent library: `deepagents` package on PyPI
- Groq: https://console.groq.com
- Tavily Search: https://tavily.com
