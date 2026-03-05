# astrbot持久化记忆插件

一个为 AstrBot 提供长期记忆能力的插件。插件将关键对话信息向量化后写入 LanceDB，在后续对话中自动检索相关记忆并注入上下文，从而实现跨会话、可持久化的记忆能力。

- 插件目录（内部 ID）：`astrbot_plugin_memory_lancedb`
- 展示名称：`astrbot持久化记忆插件`
- 当前版本：`v0.3.0`

## 简介

本插件是基于开源项目 **[win4r/memory-lancedb-pro](https://github.com/win4r/memory-lancedb-pro)** 的二次开发版本，针对 AstrBot 的事件机制、插件结构和工具系统进行了适配与重构。

你可以把它理解为：
1. 自动把“值得记住”的信息存入长期记忆库。
2. 在用户发新消息时自动召回相关记忆。
3. 把召回结果注入到模型系统上下文中，提升回复一致性和连续性。

## 核心功能

1. 记忆持久化：基于 LanceDB 落盘存储，重启不丢失。
2. 自动记忆捕获：在 `on_llm_response` 阶段自动筛选并写入记忆。
3. 自动记忆召回：在 `on_llm_request` 阶段检索并注入 `<relevant-memories>`。
4. 混合检索：向量检索 + BM25（FTS）融合。
5. 重排序：支持 Jina reranker（默认 `jina-reranker-v2-base-multilingual`）。
6. 多重打分策略：recency、importance、length normalization、time decay。
7. 作用域隔离：支持 `global`、`session`、`session+global`。
8. 管理工具：`memory_recall` / `memory_store` / `memory_forget` / `memory_list` / `memory_stats`。
9. 安全处理：记忆注入块使用 `UNTRUSTED DATA` 包裹，降低提示注入风险。

## 工作原理（详细）

### 1) 自动召回链路（请求前）

触发点：`@filter.on_llm_request()`

执行流程：
1. 读取用户问题（`req.prompt` 或 `event.message_str`）。
2. 预处理查询：去除可能的时间戳包装前缀。
3. 召回门控：
   - `auto_recall` 开关。
   - `auto_recall_min_length` 最小长度阈值。
   - `should_skip_retrieval` 跳过规则（问候语、命令、心跳等）。
4. 生成 query embedding（Jina embedding）。
5. 执行混合检索：
   - 向量检索（LanceDB ANN）。
   - BM25 检索（FTS）。
   - 融合打分与去噪去重。
6. 可选 rerank（Jina reranker）。
7. 将结果以 `<relevant-memories>` 格式注入到 `system_prompt`。

### 2) 自动捕获链路（响应后）

触发点：`@filter.on_llm_response()`

执行流程：
1. 收集候选文本（默认用户消息，可选助手消息）。
2. 触发词判断（偏好/决策/实体/事实等模式）。
3. 噪声过滤：过滤寒暄、无效短句、元问题等。
4. 元指令排除：过滤“删除记忆/清理记忆”这类管理语句，避免污染记忆库。
5. 生成 passage embedding。
6. 重复检查（向量相似度阈值）。
7. 写入 LanceDB（含 category、scope、importance、metadata）。

### 3) 检索与打分策略

混合召回后依次进行：
1. `recency boost`：新信息优先。
2. `importance weight`：高重要度优先。
3. `length normalization`：抑制超长文本“碾压”。
4. `time decay`：陈旧信息逐步衰减。
5. `hard_min_score`：硬阈值过滤。
6. `noise filter` 与去重。

### 4) 幽灵记忆防护

已实现对 BM25 失配场景的保护：
1. BM25-only 命中会校验 `id` 是否仍存在于数据表。
2. 去掉 BM25-only 固定 0.5 提升，降低误召回概率。

## 配置说明

配置文件：`/AstrBot/data/config/astrbot_plugin_memory_lancedb_config.json`

主要配置项（常用）：

1. `embedding_api_key`：主 Jina API Key（首次加载可自动转存到插件 KV）。
2. `embedding_api_keys`：额外 Jina API Key（逗号/换行分隔，支持轮转）。
3. `retry_on_rate_limit`：限流/节流时自动切换到下一个 Key 并重试。
4. `embedding_model`：嵌入模型，默认 `jina-embeddings-v5-text-small`。
5. `rerank_model`：重排模型，默认 `jina-reranker-v2-base-multilingual`。
6. `retrieval_mode`：`hybrid` 或 `vector`。
7. `scope_mode`：`global` / `session` / `session+global`。
8. `auto_recall`：是否自动召回注入。
9. `auto_recall_min_length`：自动召回最小触发长度。
10. `recall_cross_turn_dedup`：跨轮次去重，避免连续注入同一条记忆。
11. `recall_dedup_window_sec`：跨轮次去重窗口（秒）。
12. `access_boost_weight`：访问强化排序权重（建议 `0.0~0.3`，默认 `0.08`）。
13. `auto_capture`：是否自动捕获记忆。
14. `capture_assistant`：是否捕获助手回复。
15. `recall_limit`：每轮注入记忆条数上限。
16. `embedding_chunking`：嵌入超长文本时是否自动分块并做均值向量（默认开启）。

`db_path` 在初始化时会做预校验（目录自动创建、可写性检查、符号链接检查），失败时会给出可操作的错误信息，便于快速排障。

完整字段请参考：`_conf_schema.json`。

## 工具说明

1. `memory_recall(query, limit, scope)`：检索记忆。
2. `memory_store(text, importance, category, scope)`：写入记忆。
3. `memory_forget(memory_id, query, scope)`：删除记忆。
4. `memory_list(scope, limit)`：列出记忆。
5. `memory_stats(scope)`：统计记忆分布。

## 安装与启用

1. 将插件放入目录：`/AstrBot/data/plugins/astrbot_plugin_memory_lancedb`
2. 确认 `requirements.txt` 包含：
   - `aiohttp>=3.9.5`
   - `lancedb>=0.29.2`
3. 配置 `astrbot_plugin_memory_lancedb_config.json`。
4. 重启 AstrBot 容器。

## 与 AstrBot 内置功能的关系（避免冲突）

建议关闭 AstrBot 内置的“群聊上下文感知（原聊天记忆增强）”，否则会与本插件同时向模型注入额外上下文，导致提示词重复、长度膨胀与回复漂移。

建议关闭以下配置项：

1. `provider_ltm_settings.group_icl_enable = false`
2. `provider_ltm_settings.active_reply.enable = false`

原因说明：

1. 本插件会在 `on_llm_request` 中注入 `<relevant-memories>`，属于“长期记忆召回”。
2. AstrBot 内置功能也会在请求前注入群聊历史，属于“群聊上下文感知”。
3. 两者同时开启通常不会报错，但会出现“双重上下文叠加”，实际效果常见为：
   - token 成本上升；
   - 召回噪声变多；
   - 回复稳定性下降（模型更容易被冗余上下文干扰）。

如果你明确需要“两套能力同时开”，建议至少降低本插件 `recall_limit`，并优先关闭内置 `active_reply`。

## 与上游项目的关系

本插件明确基于以下开源项目思路进行二次开发与 AstrBot 适配：

- 项目：`win4r/memory-lancedb-pro`
- 链接：<https://github.com/win4r/memory-lancedb-pro>

主要差异：
1. 上游面向 OpenClaw（TypeScript SDK）；本插件面向 AstrBot（Python Star 插件）。
2. 生命周期钩子、配置系统、工具注册接口均按 AstrBot 重写。
3. 保留核心设计理念：LanceDB 持久化、混合检索、重排、自动注入/捕获。

## 致谢

感谢 **win4r/memory-lancedb-pro** 提供的优秀开源实现与设计思路。  
本项目为基于其架构思想的 AstrBot 二次开发适配版本。

## 开源协议

本项目采用 **MIT License**，详见仓库根目录 [LICENSE](./LICENSE)。

## Release（中文）

### v0.3.0

1. 合并上游 v1.0.21 思路：新增超长文本自动分块嵌入（`embedding_chunking`，默认开启）。
2. 合并上游 v1.0.22 思路：新增 `db_path` 预校验（目录创建、可写性、符号链接目标检查）。
3. 改进 LanceDB 初始化与写入报错，输出更可操作的排障信息。

### v0.2.0

1. 插件中文化整理：名称、简介、文档与发布说明中文化。
2. 补齐防污染规则：自动捕获排除“删除/清理记忆”等管理语句。
3. 提升召回控制：新增 `auto_recall_min_length` 配置。
4. 提升检索稳健性：加入 BM25 stale 命中校验，减少幽灵记忆召回。
5. 扩展中英触发词与分类词（含繁体中文场景）。

### v0.1.0

1. 完成 AstrBot 首版落地：
   - 自动记忆捕获
   - 自动记忆召回注入
   - LanceDB 持久化
   - Jina embedding + rerank
   - 5 个记忆管理工具
