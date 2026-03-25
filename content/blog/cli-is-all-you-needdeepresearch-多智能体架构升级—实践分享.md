---
title: CLI Is All You Need：DeepResearch 多智能体架构升级—实践分享
date: 2026-03-25T12:43:50.899Z
---

## 一、省流版结论：本次升级（V5版本）解决了什么？

> **[V3版本](https://github.com/Paul33333/tinymind-blog/blob/main/content/blog/%E5%9F%BA%E4%BA%8E%E5%85%B1%E4%BA%AB%E5%B7%A5%E4%BD%9C%E5%8C%BA%E7%9A%84Multi-Agent%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%A9%E6%96%87%E4%BB%B6%E6%88%90%E4%B8%BA%E6%99%BA%E8%83%BD%E4%BD%93%E5%8D%8F%E4%BD%9C%E7%9A%84%E6%A1%A5%E6%A2%81.md) 解决了「多智能体怎么共享上下文」—— 共享沙箱工作区；V5 进一步解决了「多智能体怎么以最低认知负担调用能力、原生并行、稳定落地」—— Cli is all you need**

2025 年 12 月的 [V3 版本]((https://github.com/Paul33333/tinymind-blog/blob/main/content/blog/%E5%9F%BA%E4%BA%8E%E5%85%B1%E4%BA%AB%E5%B7%A5%E4%BD%9C%E5%8C%BA%E7%9A%84Multi-Agent%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%A9%E6%96%87%E4%BB%B6%E6%88%90%E4%B8%BA%E6%99%BA%E8%83%BD%E4%BD%93%E5%8D%8F%E4%BD%9C%E7%9A%84%E6%A1%A5%E6%A2%81.md) )，我的核心实践是：

- 文件是智能体协作的桥梁；
- 共享工作区是全局状态容器；
- TODO.md 是跨智能体的「任务账本」。

V5 没有推翻这套范式。**共享工作区的内核完整保留**，变的是能力的暴露方式——**从 5 种自定义 XML 工具协议，收敛为 1 个统一入口：Bash**。

结果是：工具分发代码量减少约 80%，子智能体获得 Unix 原生并行能力，LLM 的工具学习成本显著降低。

![架构演进对比图.jpeg](https://raw.githubusercontent.com/Paul33333/tinymind-blog/main/assets/images/2026-03-25/1774432262217.jpeg)

---

## 二、V3 遗留的三个工程痛点

![工程痛点示意图.jpeg](https://raw.githubusercontent.com/Paul33333/tinymind-blog/main/assets/images/2026-03-25/1774431946852.jpeg)

### 痛点 1：工具调用的「翻译税」

V3 中，主智能体和子智能体通过 XML 格式的自定义协议调用工具：

```xml
<tool_call>
<function>web_search</function>
<query>搜索词</query>
<time_range>week</time_range>
</tool_call>
```

5 种工具（`web_search`、`open_link`、`code_interpreter`、`terminal_commands`、`subagent`）各有独立的正则解析规则和执行分支。后端主智能体和子智能体各维护一份几乎相同的 `if-elif` 分发链。

信息在 **LLM → XML → 正则解析 → Python 函数** 之间来回翻译，每一层都是潜在的出错点。V3 代码中甚至不得不写「双重解析」来兜底：

```python
# V3：担心正则解析规则导致代码提取失败，换个方式兜底
code = tool_call_content.split('<code>')[1].split('</code>')[0].strip()
```

### 痛点 2：工具数量膨胀的认知负担

V3 的系统提示词需要花大量篇幅描述 5 种工具各自的 XML 格式——参数标签名不同、可选参数不同、调用方式不同。LLM 在推理时需要做「工具选择 + 参数对齐 + 格式正确性」三重决策。

Manus 团队在 [Context Engineering](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) 中总结得很到位：

> 当你给 Agent 100 个工具，模型更容易选错工具、走低效路径。**你武装到牙齿的 Agent 反而变蠢了。**

### 痛点 3：串行执行无法原生并行

V3 的子智能体只能串行调用——在 Python 应用层同步执行 `execute_subagent()`。要并行执行两个独立的研究任务，需要引入 `asyncio.gather`、任务队列、依赖分析等复杂机制。中间版本 V4 曾短暂尝试异步 submit/poll 模式，但带来了大量状态管理复杂度，最终在 V5 中被删除。

---

## 三、CLI-First 也是当前行业趋势验证

![行业趋势验证图.jpeg](https://raw.githubusercontent.com/Paul33333/tinymind-blog/main/assets/images/2026-03-25/1774432198679.jpeg)

当前AI 社区出现了一个越来越清晰的共识信号：**CLI 正在成为 Agent 最强大的执行界面**。

### Bash = 通用工具适配器

一篇 [DEV Community 上的高赞文章](https://dev.to/uenyioha/writing-cli-tools-that-ai-agents-actually-want-to-use-39no) 对比了 MCP 工具和 CLI 工具：

> MCP 的工具定义（描述、参数、JSON Schema）必须**持久加载到 Agent 的系统提示词中，在工具被使用之前就已经消耗上下文窗口**。而 CLI 调用只是一条 bash 命令和它的 stdout——只在被实际使用时才消耗 token。

> 当我为 Gitea 构建了 MCP Server，然后意识到 Agent 可以直接运行 Gitea 的 CLI 工具 `tea` 时，MCP Server 就成了纯粹的多余开销。

### Unix 哲学天然适配 Agent

The New Stack 的《[Bash Is All You Need](https://thenewstack.io/the-key-to-agentic-success-let-unix-bash-lead-the-way/)》指出：每个工具做好一件事，**工具之间通过文本管道组合**——LLM 天生擅长处理文本流，stdout 从一个程序流向另一个程序的 stdin，正是 LLM 的母语。

Vercel 的 d0v2 团队做了一个实验：移除 80% 的辅助信息后，Agent 反而表现更好。团队反思：**grep 已经 50 岁了，它依然完美地做我们需要的事。**

### 渐进式披露（Progressive Disclosure）

![渐进式披露示意图.jpeg](https://raw.githubusercontent.com/Paul33333/tinymind-blog/main/assets/images/2026-03-25/1774432354501.jpeg)

CLI 工具天然支持渐进式披露：Agent 只需知道命令名，需要时再通过 `--help` 获取详细参数。传统方案把所有工具的全量 schema 一次性灌入提示词，CLI 方案则是按需展开——这正是 Anthropic Skills 背后的理念。

> **skill本质是提示词的渐进式披露（加载）， 那么工具tools的渐进式披露就是cli终端命令**

### Manus 的实践验证

Manus 联合创始人 Peak Ji 在一次 [Webinar](https://rlancemartin.github.io/2025/10/15/manus/) 中分享：

> Manus 不膨胀 function calling 层，而是把大部分操作下沉到沙箱层——Agent 通过 Bash 工具在沙箱中执行 CLI 命令（沙箱工具里面套工具）。

他们的分层动作空间设计：Level 1 约 20 个原子工具（其中 bash 是核心），Level 2 让模型用 bash 调沙箱 CLI，Level 3 让 Agent 写脚本组合多个 API。**工具定义不再污染上下文窗口。**

### 其他头部产品的验证

- OpenAI 将 Codex CLI 定义为本地终端中的开源命令行代理工具，可直接读写运行代码；
- Anthropic 将 Claude Code 定位为 terminal-based coding workflow，持续强化沙箱化 bash 执行与自主能力边界。

当前共识是：

> **CLI 正在成为 Agent 的「最小公共执行接口」（Minimum Common Execution Interface）。** 它未必是最终形态，但在工程落地阶段，是最稳妥、最可解释、生态兼容性最好的那层抽象。

---

## 四、V5 架构总览：CLI-First + 共享工作区

### 设计哲学

> **「保留共享工作区作为协作中枢的核心设计，将所有工具能力收敛为一个 Bash 入口」**

![V5架构总览图.jpeg](https://raw.githubusercontent.com/Paul33333/tinymind-blog/main/assets/images/2026-03-25/1774431711971.jpeg)

```
┌─────────────────────────────────────────────────────────────────┐
│              主智能体 (ReAct Loop + Bash-Only)                   │
│                                                                  │
│  ┌───────────────────────────────────────────────────────┐      │
│  │          e2b 沙箱（共享文件系统 + CLI 工具集）          │      │
│  │                                                        │      │
│  │  /usr/local/bin/                                       │      │
│  │  ├── dra-search      (网络搜索 CLI)                    │      │
│  │  ├── dra-open-url    (网页提取 CLI)                    │      │
│  │  ├── dra-subagent    (子智能体委托 CLI)                │      │
│  │  └── dra-tools       (工具列表 CLI)                    │      │
│  │                                                        │      │
│  │  /home/user/                                           │      │
│  │  ├── TODO.md         (任务进度日志)                    │      │
│  │  ├── FILE_INDEX.json (文件索引)                        │      │
│  │  ├── outputs/        (交付物)                          │      │
│  │  ├── research/       (研究文档)                        │      │
│  │  ├── data/           (数据文件)                        │      │
│  │  ├── agents/         (子智能体隔离目录) [NEW]          │      │
│  │  └── temp/           (临时文件)                        │      │
│  └───────────────────────────────────────────────────────┘      │
│                                                                  │
│  /internal/execute-subagent (子智能体 HTTP 回调端点)            │
└──────────────────────────────────────────────────────────────────┘
```

---

## 五、V3 → V5 核心变化深度解析

![工具调用流程对比图.jpeg](https://raw.githubusercontent.com/Paul33333/tinymind-blog/main/assets/images/2026-03-25/1774432008493.jpeg)

### 变化 1：从「多工具分发」到「单一 Bash 入口」

这是最根本的架构转变。

**V3 后端的工具分发**（主智能体 + 子智能体各维护一份，高度重复）：

```python
# V3: 每种工具有独立的解析和执行分支（这段逻辑在代码中出现了两次）
if function_name == "web_search":
    query_text = re.findall(config.query_text_regex, tool_call_content, re.DOTALL)[-1].strip()
    time_range = re.findall(config.time_range_regex, tool_call_content, re.DOTALL)[-1].strip() \
        if re.findall(config.time_range_regex, tool_call_content, re.DOTALL) != [] else ""
    search_result = await web_search_async(query_text, time_range)
elif function_name == "open_link":
    url = re.findall(config.URL_regex, tool_call_content, re.DOTALL)[-1].strip()
    search_result = await url_extract_async(url)
elif function_name == "code_interpreter":
    code = tool_call_content.split('<code>')[1].split('</code>')[0].strip()
    execute_result, sandbox_id, sbx = await code_interpreter_async(sbx, code)
elif function_name == "terminal_commands":
    commands = tool_call_content.split('<commands>')[1].split('</commands>')[0].strip()
    execute_result, sandbox_id, sbx = await terminal_commands_async(sbx, commands)
elif function_name == "subagent":
    agent_name = re.findall(config.subagent_agentname_regex, tool_call_content, re.DOTALL)[-1].strip()
    subagent_result = await execute_subagent(...)
```

**V5 后端的工具分发**（极度简化，且只出现一次）：

```python
# V5: 一个正则替代了五个
bash_match = re.search(config.BASH_CONTENT_REGEX, tool_call_content, re.DOTALL)
if bash_match:
    bash_cmd = bash_match.group(1).strip()
    exec_result = await execute_bash_async(sbx, bash_cmd, timeout=cmd_timeout, envs=sandbox_envs)
```

量化效果：

| 指标 | V3 | V5 | 变化 |
|------|----|----|------|
| 后端正则模式数量 | 7 个 | 1 个（BASH_CONTENT_REGEX） | -86% |
| 工具分发 if-elif 分支 | 主 5 + 子 5 = 10 个 | 主 1 + 子 1 = 2 个 | -80% |
| 系统提示词工具格式描述 | ~50 行 | ~20 行 | -60% |

### 变化 2：CLI 工具集设计

V5 在沙箱初始化时，将四个 CLI 工具安装到 `/usr/local/bin/`：

| CLI 命令 | 对应 V3 工具 | 实现方式 |
|---------|-------------|---------|
| `dra-search` | web_search | 独立 Python 脚本，调用 Tavily API |
| `dra-open-url` | open_link | 独立 Python 脚本，调用 Tavily Extract |
| `dra-subagent` | subagent | Python 脚本，HTTP POST 回调后端 |
| `dra-tools` | (无) | 列出所有可用工具 |

每个 CLI 遵循标准规范：`--help` 查看用法（渐进式披露）、`--output json|text` 切换格式、通过环境变量获取密钥（非硬编码）、stdout/stderr 返回结果（非自定义协议）。

**LLM 不再需要学习多种自定义 XML 格式——它只需要会写 bash 命令：**

```bash
# 搜索
dra-search "AI trends 2025" --time-range month

# 打开链接
dra-open-url "https://arxiv.org/abs/..." --max-chars 50000

# 执行 Python
python3 -c "import pandas as pd; df = pd.read_csv('data.csv'); print(df.head())"

# 调用子智能体
dra-subagent deep_researcher --task "Research quantum computing"
```

> 训练数据中有数十亿行 bash 命令，它早就原生学会了如何使用bash命令。让 LLM 写 `dra-search "quantum computing" --time-range week` 比写 `<tool_call><function>web_search</function><query>quantum computing</query></tool_call>` 自然得多。

### 变化 3：子智能体调用链的革命——从内部分发到 CLI 回调闭环

**V3**：主智能体在 Python 应用层直接调用 `execute_subagent()`，子智能体的 ReAct 循环在同一进程内同步执行。

**V5**：调用链变成了一个完整的闭环——

```
主智能体 → bash: dra-subagent deep_researcher --task "..."
                    ↓
              dra_subagent.py（沙箱内 CLI 脚本）
                    ↓ HTTP POST to /internal/execute-subagent
              FastAPI 后端（通过 sandbox_id 查找沙箱实例）
                    ↓
              execute_subagent()（子智能体 ReAct 循环，使用同一沙箱）
                    ↓ JSON response
              dra_subagent.py → stdout
                    ↓
              主智能体读取 stdout 作为 tool_response
```

为了让这条链路跑通，V5 引入了**全局沙箱注册表**：

```python
# V5 新增：全局沙箱注册表
_sandbox_registry: Dict[str, Sandbox] = {}

def register_sandbox(sbx: Sandbox):
    _sandbox_registry[sbx.sandbox_id] = sbx

def get_sandbox_by_id(sandbox_id: str) -> Optional[Sandbox]:
    return _sandbox_registry.get(sandbox_id)
```

当 `dra-subagent` CLI 通过 HTTP 回调后端时，后端通过 `sandbox_id` 查找同一沙箱实例来执行子智能体——子智能体依然在共享文件系统中工作，依然能读写 TODO.md 和 outputs/ 下的文件。

**这个设计最大的红利是：并行执行变成了 Unix 原生能力。** V3 需要在应用层引入异步任务队列和状态轮询才能并行，V5 只要：

```bash
# 三个子智能体并行执行，零额外代码
dra-subagent deep_researcher --task "Research topic A" > temp/r1.txt &
dra-subagent fact_checker --task "Verify claim B" > temp/r2.txt &
dra-subagent data_analyst --task "Analyze dataset C" > temp/r3.txt &
wait
cat temp/r1.txt temp/r2.txt temp/r3.txt
```

`&` + `wait` 是经过 50 年验证的并行方案。

### 变化 4：环境变量注入方式

V3 中 API 密钥在 Python 应用层硬编码管理。V5 通过 e2b 原生参数注入环境变量：

```python
# V5: 沙箱创建时注入
self.sbx = Sandbox(
    api_key=api_key,
    timeout=config.sandbox_timeout,
    envs={
        "TAVILY_API_KEYS": ",".join(config.tavily_api_keys),
        "DRA_BACKEND_URL": config.backend_public_url,
        "DRA_API_KEY": config.API_KEYS[0],
    }
)
```

CLI 工具通过 `os.environ` 读取，不需要额外配置传递。这也是 Unix 世界的标准做法。

### 变化 5：工作区初始化从 Python 代码迁移到 Bash 脚本

V3 生成 Python 代码在沙箱中执行初始化：

```python
# V3: 生成 Python 代码，再用 code_interpreter 执行
init_code = WorkspaceManager.generate_init_workspace_code(task_description)
result = await code_interpreter_async(sbx, init_code)
```

V5 直接生成 bash 脚本（here document + mkdir）：

```bash
#!/bin/bash
set -e
mkdir -p outputs research data temp agents
cat > TODO.md << 'TODOEOF'
# 任务进度日志
...
TODOEOF
```

更简洁，更符合 CLI-First 哲学，且 bash 初始化比 Python 初始化快。

---

## 六、不变的内核：共享工作区

![共享工作区示意图.jpeg](https://raw.githubusercontent.com/Paul33333/tinymind-blog/main/assets/images/2026-03-25/1774432153235.jpeg)

尽管工具层发生了根本性变化，**共享工作区的核心设计完全保留**：
> **多智能体协作的桥梁是「文件系统」，不是「消息历史长度」。**

| 设计要素 | V3 | V5 | 变化 |
|---------|----|----|------|
| TODO.md 作为全局状态 | ✅ | ✅ | 不变 |
| FILE_INDEX.json 文件索引 | ✅ | ✅ | 不变 |
| 目录分层（outputs/research/data/temp） | ✅ | ✅ | 不变，新增 agents/ |
| 子智能体自动读取 TODO.md | ✅ | ✅ | 不变 |
| 子智能体完成后更新 TODO.md | ✅ | ✅ | 不变 |
| 文件作为信息传递桥梁 | ✅ | ✅ | 不变 |
| 递归文件变化检测 + 云存储上传 | ✅ | ✅ | 不变 |

>CLI-First 只是改变了「怎么操作工作区」，没有改变「工作区作为协作中枢」的根本定位。

---

## 七、Trade-off

架构选择没有银弹。引入 CLI-First 架构也有边界和弊端：

**1. 子智能体回调的网络依赖。** `dra-subagent` 通过 HTTP 回调后端来触发执行，意味着沙箱需要访问后端公网地址。网络不稳定时子智能体调用会失败。
>一个可能方向是让子智能体直接在沙箱内运行（通过沙箱内的 LLM 客户端），但又引入 API 密钥安全性问题。

**2. 超时策略仍需细化。** V5 简化到两档超时（普通命令 10 分钟，子智能体 30 分钟），但一个简单的搜索和一个复杂的 Python 数据分析脚本不应共享同一超时。后续可以根据命令类型做更细粒度策略。

**3. 命令注入风险。** Bash 入口意味着必须依赖沙箱隔离兜底。当前实践依赖 e2b 沙箱的隔离能力，但如果未来扩展到非沙箱环境，需要额外的命令白名单或审批机制。

**4. 跨会话记忆。** 同 V3 一样，每个会话的沙箱是独立的，结束后销毁。持久化工作区、沙箱快照恢复仍待解决。

---

## 八、总结

### 1. LLM 天生就是 Shell 用户

训练数据中有数十亿行 bash 命令，但没有一行你自定义的工具协议。让 Agent 学会一种「动作语法」（shell 命令），比学会多种「平台私有 API」更稳妥。

### 2. 先统一执行面，再堆能力

很多团队容易反过来：先接十几个工具，再想怎么统一。更好的顺序是——先定义统一执行面（CLI 或等价抽象），再把工具逐步封装成子命令，最后做权限和可观测增强。

### 3. 把「文件协作协议」当一等公民

只要引入多智能体，尽早定义：哪些目录放研究、哪些放交付、TODO 的最小字段、子智能体回写约定等等。这比「提示词写得更聪明」更能提升稳定性。

### 4. 并行要「可重放」，不要「炫技」

用 `& + wait + 文件落盘` 的方式并行，优点是可重跑、可定位、可回滚。不需要在应用层重新发明进程调度——Unix 在这个问题上已经有了 50 年的解决方案。

![并行执行模式图.jpeg](https://raw.githubusercontent.com/Paul33333/tinymind-blog/main/assets/images/2026-03-25/1774431821422.jpeg)

### 5. 间接层越少越好

每一个间接层都是出错的机会。V3 中「LLM → XML → 正则 → Python 函数 → 外部 API」有四层间接。V5 中「LLM → bash → CLI 脚本 → 外部 API」同样有间接层，但每一层都是标准化的、被广泛理解的协议。

---

## 九、结语

> **让 CLI 成为智能体能力的统一入口，让文件成为智能体状态的统一事实源。**

前者解决「怎么做」（执行），后者解决「做到哪」（状态）。两件事叠在一起，才是下一阶段 Multi-Agent 工程化的关键：有自治但不失控，有并行但可追溯，有智能但尊重软件工程的基本约束。

**共享工作区不变，执行面 CLI 化，协作协议文件化。** 很多看起来像模型问题的稳定性瓶颈，最后都会变成架构问题。

---

## 📚 参考资料

1. [AIME: Towards Fully-Autonomous Multi-Agent Framework](https://arxiv.org/abs/2507.11988) - 字节跳动, 2025
2. [Context Engineering for AI Agents: Lessons from Building Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) - Manus, 2025
3. [The Key to Agentic Success? BASH Is All You Need](https://thenewstack.io/the-key-to-agentic-success-let-unix-bash-lead-the-way/) - The New Stack, 2026
4. [Writing CLI Tools That AI Agents Actually Want to Use](https://dev.to/uenyioha/writing-cli-tools-that-ai-agents-actually-want-to-use-39no) - DEV Community, 2026
5. [Progressive Disclosure of Agent Tools from the Perspective of CLI Tool Style](https://github.com/musistudio/claude-code-router/blob/main/blog/en/progressive-disclosure-of-agent-tools-from-the-perspective-of-cli-tool-style.md) - musistudio
6. [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/pdf/2210.03629) - Yao et al., 2023
7. [e2b - Code Interpreter Sandbox](https://e2b.dev/)
8. [基于共享工作区的Multi-Agent实践：让文件成为智能体协作的桥梁](https://github.com/Paul33333/tinymind-blog/blob/main/content/blog/%E5%9F%BA%E4%BA%8E%E5%85%B1%E4%BA%AB%E5%B7%A5%E4%BD%9C%E5%8C%BA%E7%9A%84Multi-Agent%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%A9%E6%96%87%E4%BB%B6%E6%88%90%E4%B8%BA%E6%99%BA%E8%83%BD%E4%BD%93%E5%8D%8F%E4%BD%9C%E7%9A%84%E6%A1%A5%E6%A2%81.md) - V3 版本博客
