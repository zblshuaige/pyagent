# LangGraph Agent

基于 LangChain `create_agent` 构建的 LangGraph 智能体。默认通过 Ollama 调用本机的 Qwen3 4B 模型，也可切换到 OpenAI 或 DeepSeek；支持命令行对话和 LangGraph 开发服务（可接入 LangSmith Studio 可视化调试）。

## 功能

- **本地大模型**：默认使用 Ollama 上的 `qwen3:4b`，推理在本机 GPU 完成。
- **多轮记忆**：使用 LangGraph 官方检查点（checkpointer），同一会话内可引用前文。
- **工具调用**：
  - 查询当前时间、安全四则运算；
  - 在 `outputs/` 目录中读取、新建、局部替换、整篇重写 UTF-8 文本文件（任意后缀，如 `.html`、`.json`、`.md`）；
  - 可选的 Tavily 网页搜索（配置 API Key 后启用）。

## 目录结构

| 文件 | 说明 |
|---|---|
| `app.py` | 智能体主程序：模型选择、工具定义、命令行对话入口 |
| `langgraph_app.py` | 向 LangGraph 开发服务导出的图 `graph` |
| `langgraph.json` | LangGraph CLI 配置，注册名为 `agent` 的图 |
| `test_file_tools.py` | 文件工具回归测试 |
| `requirements.txt` | Python 依赖 |
| `.env` | 本地配置与密钥（不纳入 Git） |
| `outputs/` | 智能体生成的文件（不纳入 Git） |

## 环境要求

- Windows 10/11，Python 3.11 及以上
- 使用本地模型时：已安装 [Ollama](https://ollama.com/) 并下载模型

```powershell
& 'F:\Ollama\ollama.exe' pull qwen3:4b
& 'F:\Ollama\ollama.exe' list
```

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 配置

在项目根目录的 `.env` 中配置。没有 `.env` 时默认使用本地 Ollama 模型。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | 模型来源：`ollama`、`openai` 或 `deepseek` |
| `OLLAMA_MODEL` | `qwen3:4b` | Ollama 模型名 |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama 服务地址 |
| `OPENAI_API_KEY` | — | 使用 OpenAI 时必填 |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI 模型名 |
| `DEEPSEEK_API_KEY` | — | 使用 DeepSeek 时必填 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | DeepSeek 模型名 |
| `TAVILY_API_KEY` | — | 可选；配置后启用网页搜索 |
| `LANGSMITH_TRACING` | — | 设为 `true` 开启 LangSmith 追踪；`false` 则不上传运行记录 |
| `LANGSMITH_API_KEY` | — | LangSmith API Key；开启追踪或使用 Studio 时需要 |
| `LANGSMITH_PROJECT` | `default` | 追踪记录在 LangSmith 中所属的项目名 |

本地模式示例：

```dotenv
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:4b
```

## 使用

### 命令行

```powershell
.\.venv\Scripts\python.exe app.py
```

| 输入 | 作用 |
|---|---|
| 任意问题 | 与智能体对话 |
| `new` | 开始新对话（清除上下文） |
| `exit` / `quit` | 退出 |

命令行记忆仅在本次运行期间有效，退出后清空。

回答会边生成边显示。使用支持思考的 Ollama 模型（如 `qwen3`）时，会先显示 `[思考]` 过程；调用工具时显示 `[调用工具]` 和 `[工具结果]`（结果只显示前 200 个字符）。

### LangGraph 开发服务与 Studio

```powershell
$env:PYTHONUTF8="1"
.\.venv\Scripts\langgraph.exe dev
```

- API：`http://127.0.0.1:2024`
- API 文档：`http://127.0.0.1:2024/docs`
- Studio：`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`（需要 LangSmith 账号和 API Key）

在 Studio 或 API 中，同一线程（thread）内连续提问会保留上下文，新建线程则开启独立对话；该记忆与命令行不共享。开发服务仅供本机使用，不要直接暴露到公网。

## 文件工具

示例指令：

- “在 outputs 中创建 birthday_cake.html，内容为……”
- “读取 birthday_cake.html，把标题改成……并保存”

| 工具 | 作用 |
|---|---|
| `read_text_file` | 读取文件 |
| `create_text_file` | 新建文件，已存在时拒绝，不会覆盖 |
| `replace_text_file` | 替换文件中**只出现一次**的指定片段，适合局部修改 |
| `rewrite_text_file` | 整篇重写，要求传入的原文与当前文件完全一致 |

安全限制：

- 只能操作 `outputs/` 目录，只接受文件名，拒绝子目录、绝对路径和 `../`；
- 单个文件及写入内容最多 10000 个字符；
- 只写 UTF-8 文本：改成 `.png`、`.pdf` 等后缀不会生成有效的二进制文件。

读写底层使用 `langchain-community` 的 `ReadFileTool`、`WriteFileTool`，外层由工程提供上述保护，不会把原生写入工具直接交给模型。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_file_tools
```

## 常见问题

**启动 `langgraph dev` 报 `UnicodeDecodeError: 'gbk'`**
Windows 默认编码导致，启动前执行 `$env:PYTHONUTF8="1"`。

**命令行运行报缺少模块**
请使用虚拟环境中的 Python：`.\.venv\Scripts\python.exe app.py`，不要用系统的 `py` 或 `python`。

**本地模型偶尔调用工具失败或参数错误**
4B 小模型的工具调用不够稳定，通常会自动重试修正；指令中写明文件名和要替换的原文片段更可靠。需要更高稳定性时可换用更大的模型或云端模型。

**可以换哪些本地模型**
RTX 2060（6 GB 显存）适合 1–4B 的量化模型，如 `qwen3:4b`、`qwen2.5-coder:3b`；7–8B 可以运行但可能较慢。修改 `.env` 中的 `OLLAMA_MODEL` 即可切换。带 `:cloud` 标签的模型在 Ollama 云端推理，不会下载到本机。