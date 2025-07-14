# Google ADK Multi-Agent Systems Framework

## Table of Contents

1. [Framework Overview](#framework-overview)
2. [Core Concepts](#core-concepts)
3. [Environment Setup](#environment-setup)
4. [Architecture Patterns](#architecture-patterns)
5. [Agent Types Reference](#agent-types-reference)
6. [State Management](#state-management)
7. [Tool Development](#tool-development)
8. [Workflow Patterns](#workflow-patterns)
9. [Best Practices](#best-practices)
10. [Troubleshooting Guide](#troubleshooting-guide)
11. [Complete Code Templates](#complete-code-templates)

---

## Framework Overview

The Google Agent Development Kit (ADK) enables the creation of sophisticated multi-agent systems that can collaborate on complex tasks. This framework provides reusable patterns for building hierarchical and workflow-driven agent systems.

### Key Benefits

- **Modular Design**: Break complex workflows into specialized agents
- **Reliable Performance**: Each agent focuses on specific tasks with clear examples
- **Easy Maintenance**: Update individual components without affecting the entire system
- **Reusable Components**: Transfer agents between different workflows
- **Structured Flow Control**: Predictable conversation routing through hierarchical trees

---

## Core Concepts

### Agent Hierarchy Types

```
Root Agent (Entry Point)
├── Sub-Agent 1 (Specialist)
│   ├── Sub-Sub-Agent A
│   └── Sub-Sub-Agent B
├── Sub-Agent 2 (Specialist)
└── Workflow Agent
    ├── Sequential Sub-Agents
    ├── Parallel Sub-Agents
    └── Loop Sub-Agents
```

### Conversation Flow Patterns

1. **Hierarchical Flow**: User interacts with specialists via parent routing
2. **Workflow Flow**: Agents execute automatically without user input
3. **Hybrid Flow**: Combination of user interaction and automated workflows

---

## Environment Setup
### For Google Cloud Shell Keep in mind
```bash
cloudshell workspace ~
```

### Prerequisites

```bash
# Required system components
- Python 3.8+
- Google Cloud Project with Vertex AI enabled
- Internet connectivity for package installation
```

### Installation Script

```bash
#!/bin/bash
# ADK Multi-Agent Setup Script

# Install ADK with specific version for consistency
sudo python3 -m pip install google-adk==1.4.2

# Install common dependencies
sudo python3 -m pip install python-dotenv google-cloud-logging

# Create project structure
mkdir -p my_multiagent_project/{agents,tools,config,outputs}
cd my_multiagent_project
```

### Environment Configuration

Create `.env` file in project root:

```ini
# .env template
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
MODEL=gemini-2.0-flash-001
LOG_LEVEL=INFO
```
### Google Shell Environment Configuration

```bash
# Go to your working directory
cd ~/agentic-fw
```

```bash
# Create a virtual environment in this folder
python3 -m venv .venv
```

```bash
# Activate the virtual environment
pip install -r requirements.txt

```

```bash
# Validate Environment
which python
which pip
pip list

```

### If You Want the Latest ADK Production Features

```bash
# Check Installed ADK Version
pip show google-adk
```

```bash
# Use the most stable version listed on PyPI:
pip install google-adk --upgrade
```

```bash

# Create .env File via Bash
cat <<EOF > .env
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_CLOUD_PROJECT=agentic-fw
GOOGLE_CLOUD_LOCATION=us-central1
MODEL=gemini-2.0-flash-001
LOG_LEVEL=INFO
EOF
```

```bash

# Confirm It Was Created
cat .env
```



### Project Structure Template

```
my_multiagent_project/
├── .env                    # Environment variables
├── requirements.txt        # Python dependencies
├── main.py                # Entry point
├── agents/
│   ├── __init__.py
│   ├── hierarchical/      # Parent-child agents
│   │   ├── __init__.py
│   │   └── agent.py
│   └── workflow/          # Sequential/Loop/Parallel agents
│       ├── __init__.py
│       └── agent.py
├── tools/
│   ├── __init__.py
│   ├── state_tools.py     # State management tools
│   ├── file_tools.py      # File operations
│   └── custom_tools.py    # Project-specific tools
├── config/
│   ├── __init__.py
│   └── logging_config.py
└── outputs/               # Generated files
```

---

## Architecture Patterns

### Pattern 1: Hierarchical Specialists

Use when you need user interaction with different specialist agents.

```python
# Example: Customer Service System
root_agent = Agent(
    name="customer_service_router",
    description="Routes customer inquiries to appropriate specialists",
    instruction="""
    Analyze the customer's request and route to:
    - 'billing_agent' for payment and invoice questions
    - 'technical_agent' for product support
    - 'sales_agent' for new purchases
    """,
    sub_agents=[billing_agent, technical_agent, sales_agent]
)
```

### Pattern 2: Sequential Workflow

Use for multi-step processes that must happen in order.

```python
# Example: Document Processing Pipeline
document_processor = SequentialAgent(
    name="document_processor",
    description="Process documents through validation, analysis, and summary",
    sub_agents=[
        document_validator,    # Step 1: Validate format
        content_analyzer,      # Step 2: Extract key information
        summary_generator,     # Step 3: Create summary
        file_saver            # Step 4: Save results
    ]
)
```

### Pattern 3: Iterative Refinement

Use for tasks requiring multiple improvement cycles.

```python
# Example: Content Creation Loop
content_refinement = LoopAgent(
    name="content_refinement",
    description="Iteratively improve content quality",
    sub_agents=[
        content_writer,        # Generate content
        content_reviewer,      # Review and provide feedback
        content_editor        # Apply improvements
    ],
    max_iterations=5
)
```

### Pattern 4: Parallel Processing

Use for independent tasks that can run simultaneously.

```python
# Example: Multi-source Research
research_team = ParallelAgent(
    name="research_team",
    description="Gather information from multiple sources",
    sub_agents=[
        web_researcher,        # Search web sources
        database_researcher,   # Query internal databases
        expert_interviewer    # Simulate expert consultation
    ]
)
```

---

## Agent Types Reference

### Base Agent Configuration

```python
from google.adk import Agent
from google.genai import types

base_agent = Agent(
    name="agent_name",                    # Unique identifier
    model="gemini-2.0-flash-001",        # Model specification
    description="Agent purpose",          # For parent routing decisions
    instruction="Detailed behavior",      # Core behavior definition
    tools=[],                            # Available tools list
    sub_agents=[],                       # Child agents
    generate_content_config=types.GenerateContentConfig(
        temperature=0.7,                  # Creativity level (0-1)
        top_p=0.95,                      # Nucleus sampling
        max_output_tokens=2048           # Response length limit
    ),
    output_key="agent_output",           # Store output in state
    disallow_transfer_to_peers=False     # Allow peer transfers
)
```

### Workflow Agent Configurations

#### SequentialAgent

```python
from google.adk.agents import SequentialAgent

sequential_workflow = SequentialAgent(
    name="sequential_process",
    description="Execute agents in defined order",
    sub_agents=[agent1, agent2, agent3]  # Executes in order
)
```

#### LoopAgent

```python
from google.adk.agents import LoopAgent
from google.adk.tools import exit_loop

iterative_workflow = LoopAgent(
    name="iterative_process",
    description="Repeat until condition met",
    sub_agents=[researcher, writer, critic],
    max_iterations=5,                     # Safety limit
    # Agents can call exit_loop tool to terminate early
)
```

#### ParallelAgent

```python
from google.adk.agents import ParallelAgent

parallel_workflow = ParallelAgent(
    name="parallel_process",
    description="Execute agents concurrently",
    sub_agents=[agent_a, agent_b, agent_c]  # Run simultaneously
)
```

---

## State Management

### Core State Operations

```python
from google.adk.tools.tool_context import ToolContext
from typing import List, Dict, Any

def append_to_state(
    tool_context: ToolContext, 
    field: str, 
    content: str
) -> Dict[str, str]:
    """Append content to a state field (creates list)"""
    existing = tool_context.state.get(field, [])
    tool_context.state[field] = existing + [content]
    return {"status": "success"}

def update_state(
    tool_context: ToolContext, 
    field: str, 
    content: Any
) -> Dict[str, str]:
    """Replace entire state field"""
    tool_context.state[field] = content
    return {"status": "success"}

def merge_state(
    tool_context: ToolContext, 
    field: str, 
    new_data: Dict
) -> Dict[str, str]:
    """Merge dictionary data into state field"""
    existing = tool_context.state.get(field, {})
    existing.update(new_data)
    tool_context.state[field] = existing
    return {"status": "success"}
```

### State Access in Instructions

```python
agent_with_state = Agent(
    name="state_aware_agent",
    instruction="""
    Current user preferences: {{ user_preferences? }}
    Processing queue: {{ processing_queue? }}
    
    Use the question mark syntax for optional fields that might not exist.
    Use regular syntax {{ field_name }} for required fields.
    """,
    tools=[append_to_state, update_state]
)
```

### Advanced State Patterns

```python
# Complex state structure example
def initialize_project_state(tool_context: ToolContext) -> Dict[str, str]:
    """Initialize a complex project structure in state"""
    project_state = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
            "status": "initialized"
        },
        "tasks": [],
        "results": {},
        "errors": []
    }
    tool_context.state["project"] = project_state
    return {"status": "project initialized"}

# Accessing nested state in instructions
nested_state_agent = Agent(
    instruction="""
    Project Status: {{ project.metadata.status? }}
    Task Count: {{ project.tasks|length }}
    
    Use Jinja2 syntax for complex state access.
    """
)
```

---

## Tool Development

### Tool Function Template

```python
from google.adk.tools.tool_context import ToolContext
from typing import List, Dict, Any, Optional
import logging

def template_tool(
    tool_context: ToolContext,
    param1: str,
    param2: Optional[int] = None,
    param3: List[str] = None
) -> Dict[str, Any]:
    """
    Tool description for LLM understanding.
    
    Args:
        param1 (str): Required parameter description
        param2 (int, optional): Optional parameter description
        param3 (List[str], optional): List parameter description
    
    Returns:
        Dict[str, Any]: Status and results
    """
    try:
        # Tool logic here
        result = process_parameters(param1, param2, param3)
        
        # Update state if needed
        tool_context.state["tool_results"] = result
        
        # Log for debugging
        logging.info(f"Tool executed successfully: {result}")
        
        return {
            "status": "success",
            "result": result,
            "message": "Tool completed successfully"
        }
    
    except Exception as e:
        logging.error(f"Tool error: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }
```

### Integrating External Tools

```python
# LangChain Integration
from google.adk.tools.langchain_tool import LangchainTool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wikipedia_tool = LangchainTool(
    tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
)

# CrewAI Integration
from google.adk.tools.crewai_tool import CrewaiTool
from crewai_tools import FileWriterTool

file_writer_tool = CrewaiTool(
    name="file_writer",
    description="Writes content to files",
    tool=FileWriterTool()
)

# Agent with external tools
research_agent = Agent(
    name="researcher",
    tools=[wikipedia_tool, file_writer_tool, custom_state_tool]
)
```

---

## Workflow Patterns

### Pattern: Research → Draft → Review → Finalize

```python
def create_content_pipeline():
    """Complete content creation workflow"""
    
    # Research Phase
    researcher = Agent(
        name="researcher",
        description="Gather information on the topic",
        instruction="""
        Research the topic: {{ topic }}
        Use Wikipedia and other sources.
        Save findings to 'research_data' state.
        """,
        tools=[wikipedia_tool, append_to_state]
    )
    
    # Draft Phase
    writer = Agent(
        name="content_writer",
        description="Create initial content draft",
        instruction="""
        Using research: {{ research_data? }}
        Create a comprehensive draft.
        Save to 'content_draft' state.
        """,
        tools=[append_to_state]
    )
    
    # Review Phase
    reviewer = Agent(
        name="content_reviewer",
        description="Review and suggest improvements",
        instruction="""
        Review draft: {{ content_draft? }}
        Provide specific feedback for improvement.
        If major issues, save feedback to 'review_feedback'.
        If ready, call exit_loop tool.
        """,
        tools=[append_to_state, exit_loop]
    )
    
    # Finalization Phase
    finalizer = Agent(
        name="content_finalizer",
        description="Apply final formatting and save",
        instruction="""
        Final content: {{ content_draft? }}
        Apply professional formatting and save to file.
        """,
        tools=[file_writer_tool]
    )
    
    # Workflow Assembly
    content_loop = LoopAgent(
        name="content_refinement",
        sub_agents=[writer, reviewer],
        max_iterations=3
    )
    
    content_pipeline = SequentialAgent(
        name="content_creation_pipeline",
        sub_agents=[researcher, content_loop, finalizer]
    )
    
    return content_pipeline
```

### Pattern: Multi-Source Analysis

```python
def create_analysis_system():
    """Parallel analysis from multiple sources"""
    
    # Parallel Research
    web_analyst = Agent(
        name="web_analyst",
        description="Analyze web-based information",
        tools=[web_search_tool],
        output_key="web_analysis"
    )
    
    database_analyst = Agent(
        name="database_analyst", 
        description="Query internal databases",
        tools=[database_tool],
        output_key="database_analysis"
    )
    
    document_analyst = Agent(
        name="document_analyst",
        description="Process uploaded documents",
        tools=[document_processor_tool],
        output_key="document_analysis"
    )
    
    # Synthesis Phase
    synthesizer = Agent(
        name="analysis_synthesizer",
        description="Combine all analysis results",
        instruction="""
        Web Analysis: {{ web_analysis? }}
        Database Analysis: {{ database_analysis? }}
        Document Analysis: {{ document_analysis? }}
        
        Create comprehensive synthesis report.
        """,
        tools=[report_generator_tool]
    )
    
    # Workflow Assembly
    parallel_analysis = ParallelAgent(
        name="multi_source_analysis",
        sub_agents=[web_analyst, database_analyst, document_analyst]
    )
    
    analysis_system = SequentialAgent(
        name="complete_analysis_system",
        sub_agents=[parallel_analysis, synthesizer]
    )
    
    return analysis_system
```

---

## Best Practices

### Agent Design Principles

1. **Single Responsibility**: Each agent should have one clear purpose
2. **Clear Instructions**: Use specific, actionable language in instructions
3. **State Management**: Use state for important data, conversation history for context
4. **Error Handling**: Include error handling in tools and instructions
5. **Temperature Control**: Lower temperature (0-0.3) for deterministic tasks, higher (0.7-1.0) for creative tasks

### Instruction Writing Guidelines

```python
# Good instruction example
good_instruction = """
CONTEXT: You are a technical documentation specialist.

TASK: Review the provided code and create user-friendly documentation.

PROCESS:
1. Analyze code structure and functionality
2. Identify key features and usage patterns
3. Create clear examples with explanations
4. Use markdown formatting for readability

INPUT: {{ code_to_document }}

OUTPUT: Save documentation to 'technical_docs' state using your tool.

QUALITY CRITERIA:
- Include practical examples
- Explain complex concepts simply
- Provide troubleshooting tips
"""

# Poor instruction example
poor_instruction = "Document the code"  # Too vague
```

### State Management Best Practices

```python
# Organized state structure
def setup_organized_state(tool_context: ToolContext):
    """Initialize well-organized state structure"""
    tool_context.state.update({
        # Metadata
        "session_id": generate_session_id(),
        "created_at": datetime.now().isoformat(),
        
        # User data
        "user_preferences": {},
        "user_inputs": [],
        
        # Processing data
        "current_task": None,
        "task_queue": [],
        "completed_tasks": [],
        
        # Results
        "intermediate_results": {},
        "final_results": {},
        
        # System status
        "errors": [],
        "warnings": [],
        "system_status": "initialized"
    })
```

### Error Handling Patterns

```python
def robust_tool_with_error_handling(
    tool_context: ToolContext,
    operation: str,
    data: Any
) -> Dict[str, Any]:
    """Tool with comprehensive error handling"""
    
    try:
        # Validate inputs
        if not operation:
            raise ValueError("Operation parameter is required")
        
        if operation not in ["process", "analyze", "generate"]:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Execute operation
        result = execute_operation(operation, data)
        
        # Validate results
        if not result:
            raise RuntimeError("Operation produced no results")
        
        # Update state
        tool_context.state["last_operation"] = {
            "operation": operation,
            "status": "success",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "status": "success",
            "result": result
        }
        
    except ValueError as e:
        error_msg = f"Input validation error: {str(e)}"
        logging.warning(error_msg)
        tool_context.state["errors"].append(error_msg)
        
        return {
            "status": "error",
            "error_type": "validation",
            "message": error_msg
        }
        
    except Exception as e:
        error_msg = f"Unexpected error in {operation}: {str(e)}"
        logging.error(error_msg)
        tool_context.state["errors"].append(error_msg)
        
        return {
            "status": "error", 
            "error_type": "system",
            "message": error_msg
        }
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: Agent Not Transferring to Sub-Agents

```python
# Problem: Missing sub_agents parameter
broken_agent = Agent(
    name="parent",
    # sub_agents parameter missing
)

# Solution: Add sub_agents parameter
fixed_agent = Agent(
    name="parent",
    sub_agents=[child_agent_1, child_agent_2]
)
```

#### Issue: State Not Persisting Between Agents

```python
# Problem: Not using ToolContext properly
def broken_state_tool(field: str, value: str):
    # This doesn't persist to session
    local_dict = {field: value}

# Solution: Use ToolContext
def working_state_tool(tool_context: ToolContext, field: str, value: str):
    tool_context.state[field] = value
    return {"status": "success"}
```

#### Issue: Loop Not Terminating

```python
# Problem: No exit condition
infinite_loop = LoopAgent(
    name="endless_loop",
    sub_agents=[agent1, agent2],
    # No max_iterations and no exit_loop tool
)

# Solution: Add exit conditions
safe_loop = LoopAgent(
    name="safe_loop",
    sub_agents=[agent1, agent2, critic_with_exit_loop],
    max_iterations=5  # Safety limit
)
```

### Debugging Techniques

```python
# Add logging to agents
import logging

debug_agent = Agent(
    name="debug_agent",
    instruction="""
    Log current state: {{ __STATE__ }}
    Process the request and log results.
    """,
    before_model_callback=lambda request: logging.info(f"Agent input: {request}"),
    after_model_callback=lambda response: logging.info(f"Agent output: {response}")
)

# Debug tool for state inspection
def debug_state_tool(tool_context: ToolContext) -> Dict[str, Any]:
    """Debug tool to inspect current state"""
    state_summary = {
        "state_keys": list(tool_context.state.keys()),
        "state_size": len(tool_context.state),
        "recent_events": len(tool_context.events[-5:]) if tool_context.events else 0
    }
    
    logging.info(f"State Debug: {state_summary}")
    return {"status": "debug_complete", "summary": state_summary}
```

---

## Complete Code Templates

### Basic Hierarchical System Template

```python
# basic_hierarchical_template.py
import os
from dotenv import load_dotenv
from google.adk import Agent
from google.genai import types

load_dotenv()

# Configuration
MODEL = os.getenv("MODEL", "gemini-2.0-flash-001")

# Specialist Agents
specialist_a = Agent(
    name="specialist_a",
    model=MODEL,
    description="Handles type A requests",
    instruction="Process type A requests with specific expertise.",
    generate_content_config=types.GenerateContentConfig(temperature=0.3)
)

specialist_b = Agent(
    name="specialist_b", 
    model=MODEL,
    description="Handles type B requests",
    instruction="Process type B requests with specific expertise.",
    generate_content_config=types.GenerateContentConfig(temperature=0.3)
)

# Root Agent
root_agent = Agent(
    name="router",
    model=MODEL,
    description="Routes user requests to appropriate specialists",
    instruction="""
    Analyze the user's request and route to:
    - 'specialist_a' for type A requests
    - 'specialist_b' for type B requests
    """,
    sub_agents=[specialist_a, specialist_b],
    generate_content_config=types.GenerateContentConfig(temperature=0.1)
)
```

### Complete Workflow System Template

```python
# complete_workflow_template.py
import os
from dotenv import load_dotenv
from google.adk import Agent
from google.adk.agents import SequentialAgent, LoopAgent, ParallelAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools import exit_loop
from typing import Dict, Any

load_dotenv()

# State Management Tools
def append_to_state(tool_context: ToolContext, field: str, content: str) -> Dict[str, str]:
    existing = tool_context.state.get(field, [])
    tool_context.state[field] = existing + [content]
    return {"status": "success"}

def save_final_result(tool_context: ToolContext, result: str) -> Dict[str, str]:
    tool_context.state["final_result"] = result
    return {"status": "saved"}

# Individual Agents
researcher = Agent(
    name="researcher",
    model=os.getenv("MODEL"),
    description="Gathers information on the topic",
    instruction="""
    Research topic: {{ topic }}
    Save findings to 'research_data' state.
    """,
    tools=[append_to_state]
)

analyzer = Agent(
    name="analyzer", 
    model=os.getenv("MODEL"),
    description="Analyzes research data",
    instruction="""
    Analyze: {{ research_data? }}
    Save analysis to 'analysis_data' state.
    """,
    tools=[append_to_state]
)

quality_checker = Agent(
    name="quality_checker",
    model=os.getenv("MODEL"),
    description="Checks work quality and decides if more iteration needed",
    instruction="""
    Review research: {{ research_data? }}
    Review analysis: {{ analysis_data? }}
    
    If quality is sufficient, call exit_loop.
    If improvements needed, provide feedback in 'quality_feedback'.
    """,
    tools=[append_to_state, exit_loop]
)

reporter_a = Agent(
    name="reporter_a",
    model=os.getenv("MODEL"), 
    description="Creates report type A",
    instruction="Create report A from: {{ analysis_data? }}",
    output_key="report_a"
)

reporter_b = Agent(
    name="reporter_b",
    model=os.getenv("MODEL"),
    description="Creates report type B", 
    instruction="Create report B from: {{ analysis_data? }}",
    output_key="report_b"
)

finalizer = Agent(
    name="finalizer",
    model=os.getenv("MODEL"),
    description="Combines all outputs into final result",
    instruction="""
    Combine outputs:
    Report A: {{ report_a? }}
    Report B: {{ report_b? }}
    Analysis: {{ analysis_data? }}
    
    Create comprehensive final report.
    """,
    tools=[save_final_result]
)

# Workflow Assembly
quality_loop = LoopAgent(
    name="quality_assurance_loop",
    description="Iteratively improve research and analysis",
    sub_agents=[researcher, analyzer, quality_checker],
    max_iterations=3
)

parallel_reporting = ParallelAgent(
    name="parallel_reports",
    description="Generate multiple report types simultaneously",
    sub_agents=[reporter_a, reporter_b]
)

complete_workflow = SequentialAgent(
    name="research_and_reporting_workflow",
    description="Complete research, analysis, and reporting system",
    sub_agents=[quality_loop, parallel_reporting, finalizer]
)

# Root Agent
root_agent = Agent(
    name="workflow_initiator",
    model=os.getenv("MODEL"),
    description="Initiates the complete workflow process",
    instruction="""
    Welcome user and ask for research topic.
    Save topic to 'topic' state and transfer to workflow.
    """,
    tools=[append_to_state],
    sub_agents=[complete_workflow]
)
```

### Usage Example

```python
# main.py - How to use the framework
from google.adk.cli import run_agent
from agents.workflow.agent import root_agent

if __name__ == "__main__":
    # Command line interface
    run_agent(root_agent)
    
    # Or programmatic usage
    # session = create_session()
    # response = root_agent.process(session, "Hello, I need help with...")
```

This framework provides a comprehensive foundation for building multi-agent systems with Google ADK. Each pattern and template can be adapted for specific use cases while maintaining the core architectural principles.
