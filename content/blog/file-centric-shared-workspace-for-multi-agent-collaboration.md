---
title: File-Centric Shared Workspace for Multi-Agent Collaboration
date: 2025-12-22T13:13:37.950Z
---

**Designing a Practical System for Dynamic Planning, Execution, and Memory for Multi-Agent Collaboration**

---

## Abstract

Large Language Model (LLM)-based multi-agent systems have shown promise in solving complex tasks through decomposition and collaboration. However, existing frameworks often suffer from rigid execution plans, static agent capabilities, and inefficient inter-agent communication that relies heavily on token-based message passing.

In this work, we present a **file-centric multi-agent system design** that treats a shared workspace as the primary medium for coordination, memory, and state synchronization. Instead of exchanging long conversational messages, agents communicate indirectly through persistent files that represent tasks, progress, intermediate artifacts, and final outputs.

Inspired by the Dynamic Planner–Actor architecture in [AIME](https://arxiv.org/pdf/2507.11988) and the [React](https://arxiv.org/pdf/2210.03629) reasoning–acting paradigm, our system externalizes agent state into a shared filesystem, enabling dynamic replanning, lossless information sharing, and robust recovery from failures. We demonstrate that this design significantly improves scalability, debuggability, and context efficiency in real-world multi-agent workflows.

> ![AIME 整体框架图.png](https://raw.githubusercontent.com/Paul33333/tinymind-blog/main/assets/images/2025-12-18/1766063375374.png)



---

## 1. Introduction

Multi-agent systems built on top of LLMs are increasingly used for complex tasks such as research synthesis, software development, and data analysis. Frameworks such as [MetaGPT](https://arxiv.org/pdf/2308.00352), AutoGen, and CrewAI decompose tasks into subtasks executed by specialized agents.

Despite their success, we observe three recurring limitations:

1. **Rigid plan execution**: once a plan is generated, it is executed mechanically even when new information invalidates subsequent steps.
2. **Static agent capabilities**: agents are predefined with fixed roles and cannot adapt dynamically to task requirements.
3. **Inefficient communication**: agents rely on token-based message passing, leading to information loss, context explosion, and lack of global visibility.

These issues are also identified in recent work such as [AIME](https://arxiv.org/pdf/2507.11988), which highlights fundamental flaws in static Plan-and-Execute paradigms.

We argue that the root cause lies not in agent reasoning itself, but in **how information is represented and shared** among agents.

---

## 2. Key Insight: Files as the Primitive of Collaboration

Our central insight is:

> **Files are a more reliable and scalable medium for inter-agent communication than conversational tokens.**

Token-based communication conflates *reasoning* with *state*. As task complexity grows, agents must repeatedly summarize, compress, and re-inject context into prompts, inevitably losing fidelity.

In contrast, files provide:

* **Persistent memory**: state survives agent crashes or restarts.
* **Natural encapsulation**: each file represents a coherent information unit.
* **Implicit compression**: a file path can reference arbitrarily large content.
* **Human inspectability**: artifacts can be debugged and audited directly.
* **Asynchronous access**: agents can read and write independently.

By shifting communication from messages to files, we decouple *agent cognition* from *system memory*.

---

## 3. System Overview

### 3.1 Architecture

The system consists of three primary components:

1. **Dynamic Planner (Main Agent)**
   A long-lived agent responsible for task decomposition, scheduling, and global decision-making.
2. **SubAgents (Actors)**
   Specialized agents instantiated on demand with task-specific roles, tools, and context.
3. **Shared Workspace**
   A sandboxed filesystem that serves as the single source of truth for all agents.

```
Planner ⇄ Shared Workspace ⇄ SubAgents
```

```
┌─────────────────────────────────────────────────────────────────┐
│             file-centric multi-agent system design              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────────────────────────────┐     │
│  │  Main agent │◄──►│      Shared Sandbox Workspace       │     │
│  │  (Planner)  │    │  ┌─────────────────────────────┐    │     │
│  └──────┬──────┘    │  │  TODO.md (任务进度日志)      │    │     │
│         │           │  │  FILE_INDEX.json (文件索引)  │    │    │
│         │           │  │  outputs/ (交付物目录)       │    │    │
│         │           │  │  research/ (研究文档)        │    │    │
│         ▼           │  │  data/ (数据文件)            │    │    │
│  ┌──────────────┐   │  └─────────────────────────────┘    │    │
│  │ SubAgent Pool│◄─►│                                     │    │
│  │              │   └─────────────────────────────────────┘    │
│  │ • deep_researcher                                           │
│  │ • data_analyst                                              │
│  │ • fact_checker                                              │
│  │ • code_developer                                            │
│  │ • summarizer                                                │
│  │ • general_assistant                                         │
│  │ • planner                                                   │
│  └──────────────┘                                              │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     Tool Stack                          │   │
│  │  [web_search]  [open_link]  [code_interpreter]          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

> the `sandbox` used [e2b -  Code Interpreter Sandbox](https://e2b.dev/) services, the `web_search`and`open_link` tool used [Tavily](https://www.tavily.com/) services



All agents have read/write access to the same workspace.

---

### 3.2 Shared Workspace Structure

```
/
├── TODO.md              # Global task journal (core state)
├── FILE_INDEX.json      # Metadata index of generated files
├── outputs/             # Final deliverables
├── research/            # Research documents
├── data/                # Datasets
└── temp/                # Temporary artifacts
```

The workspace externalizes system state and replaces conversational context sharing.

---

## 4. Core Workflow

The system follows a **workspace-centered ReAct loop**.

### Step 1: Workspace Initialization

Upon receiving a user request, the planner initializes the workspace and creates a `TODO.md` file containing:

* The original task
* Initial plan (possibly incomplete)
* Execution log template

This step is mandatory and ensures a consistent starting state.

---

### Step 2: Dynamic Planning

The planner writes or updates the task plan directly in `TODO.md`.
Unlike static planning, the plan is **mutable** and can be revised at any time.

---

### Step 3: Delegation and Execution

For each subtask:

* Simple tasks are handled directly by the planner using basic tools.
* Complex tasks are delegated to subagents.

Each subagent:

1. Reads `TODO.md` and workspace file listings.
2. Executes the assigned task using its tools.
3. Writes results to appropriate directories.
4. Updates `TODO.md` with progress and outputs.

---

### Step 4: Aggregation and Delivery

After all subtasks are completed, the planner reads workspace artifacts, synthesizes results, writes final outputs to `outputs/`, updates the task journal, and responds to the user.

---

## 5. SubAgent Design

### 5.1 SubAgent as Runtime Configuration

Subagents are not fixed classes but **runtime-configured entities** defined by:

* Role description
* System prompt
* Tool set
* Context injection policy

This enables dynamic capability composition.

Example roles include:

* Deep researcher
* Data analyst
* Fact checker
* Code developer
* Summarizer

---

### 5.2 Automatic Context Injection

When instantiated, a subagent automatically receives:

* A summary of `TODO.md`
* A list of existing workspace files

This ensures global awareness without message passing between agents.

---

## 6. TODO.md as Single Source of Truth

`TODO.md` functions as a **Task Journal**, not merely a checklist.

It records:

* Original task definition
* Evolving plan
* Execution status
* Generated artifacts
* Key findings

By centralizing state in a human-readable file, all agents operate on a consistent global view.

---

## 7. Engineering Considerations

### 7.1 Incremental File Writing

To avoid output truncation, long documents are generated incrementally across multiple tool calls and appended to files.

---

### 7.2 File Change Detection and Persistence

Before and after code execution, the system scans the workspace recursively to detect modified or newly created files. These artifacts can be uploaded to external storage for persistence and sharing.

---

### 7.3 Tool Invocation Format

We adopt [XML-based tool calls](https://docs.morphllm.com/guides/xml-tool-calls#xml-tool-calls%3A-beyond-json-constraints) instead of JSON to support complex, nested instructions and avoid structural limitations.

---

## 8. Comparison with AIME

| AIME Component      | Our System                |
| ------------------- | ------------------------- |
| Dynamic Planner     | Main Agent                |
| Actor Factory       | SubAgent Definitions      |
| Dynamic Actor       | ReAct-based SubAgents     |
| Progress Management | TODO.md + FILE_INDEX.json |
| Shared State        | Filesystem Workspace      |

Our work can be seen as a **concrete, file-based realization** of AIME’s abstract architecture.

---

## 9. Limitations and Future Work

Current limitations include:

* **Sequential execution**: independent tasks could be parallelized with proper concurrency control.
* **Conflict resolution**: concurrent writes to shared files require locking or transactional semantics.
* **Cross-session memory**: workspaces are ephemeral and reset per session.

Future directions include persistent workspaces, versioned task journals, and parallel execution with conflict detection.

---

## 10. Case Study

We validated the system by generating [a complete end-to-end instructional book on Retrieval-Augmented Generation (RAG)](https://deepresearch-agent.oss-cn-guangzhou.aliyuncs.com/files/Multi-Agent%E5%8D%8F%E4%BD%9C_%E6%B5%8B%E8%AF%95%E4%BB%BB%E5%8A%A1_%E7%AB%AF%E5%88%B0%E7%AB%AF%E8%BE%93%E5%87%BARAG%E6%95%99%E5%AD%A6%E4%B9%A6%E7%B1%8D.html), involving dozens of subtasks across research, writing, and synthesis. The entire process was coordinated through the shared workspace without long conversational context.

![Multi-Agent案例测试.png](https://raw.githubusercontent.com/Paul33333/tinymind-blog/main/assets/images/2025-12-18/1766064308333.png)

---

## 11. Conclusion

We present a practical multi-agent system design that rethinks collaboration by externalizing state into a shared workspace. By treating files—not tokens—as the primary coordination medium, our approach enables dynamic planning, robust information sharing, and scalable multi-agent collaboration.

We believe this file-centric paradigm provides a solid foundation for building reliable, production-grade LLM agent systems.

---

## References

1. [Aime: Towards Fully-Autonomous Multi-Agent Framework](https://arxiv.org/abs/2507.11988) - 字节跳动, 2025
2. [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/pdf/2210.03629) - Yao et al., 2023
3. [MetaGPT: Meta Programming for Multi-Agent Collaborative Framework](https://arxiv.org/pdf/2308.00352) - Hong et al., 2024
4. [Why do multi-agent LLM systems fail?](https://arxiv.org/abs/2503.13657) - Cemri et al., 2025
5. [Claude Code Interview](https://baoyu.io/blog/claude-code-best-practices-video-transcription)
6. [XML Tool Calls: Beyond JSON Constraints](https://docs.morphllm.com/guides/xml-tool-calls#xml-tool-calls%3A-beyond-json-constraints)
7. [e2b -  Code Interpreter Sandbox](https://e2b.dev/)
