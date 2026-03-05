# Release Notes（中文）

## v0.4.1（当前版本）

本版本对 v0.4.0 做“官方行为补齐”，重点收敛到上游真实提交的语义一致性：

### 补齐项

1. 对齐上游 `v1.0.26`（访问强化衰减）：
   - 新增 `reinforcement_factor`（默认 `0.5`）和 `max_half_life_multiplier`（默认 `3`）
   - `time decay` 读取记忆 metadata 中的 `accessCount/lastAccessedAt` 计算有效 half-life
   - `memory_recall`（手动检索）触发访问统计持久化写回（debounce flush）
2. 对齐上游 `v1.0.28`（autoRecall 去重）：
   - 新增 `auto_recall_min_repeated`（默认 `0`），按会话“轮次”控制同一记忆重复注入间隔
3. 对齐上游 `v1.0.29`（查询归一化）：
   - `normalize_retrieval_query` 补齐对 `Conversation info/Sender (untrusted metadata)` 与 `[cron:...]` 包装清洗

### 兼容性说明

1. 对已有 LanceDB 表结构保持兼容，不需要重建库。
2. 新增参数均提供默认值，旧配置可直接运行。

## v0.4.0

本版本对齐上游 `memory-lancedb-pro` 的 v1.0.24 / v1.0.25 / v1.0.26 / v1.0.28 / v1.0.29 / v1.0.30：

### 新增与优化

1. 多 Key 嵌入与限流重试（对齐 v1.0.24 + v1.0.25）：
   - 新增 `embedding_api_keys`（额外 Jina Key 池）
   - 新增 `retry_on_rate_limit`（触发限流时自动切换 Key 重试）
2. 召回跨轮次去重（对齐 v1.0.28）：
   - 新增 `recall_cross_turn_dedup` 与 `recall_dedup_window_sec`
   - 降低连续多轮重复注入同一记忆导致的 token 浪费
3. 查询归一化增强（对齐 v1.0.29）：
   - 进一步清洗角色前缀、引用标记和包装标签
4. 写入去重 fail-open（对齐 v1.0.30）：
   - 向量预检异常时不中断写入，仅告警并继续存储
5. 访问强化排序（对齐 v1.0.26）：
   - 对近期高频访问记忆进行轻量加权，提升稳定召回概率
   - 新增 `access_boost_weight` 可调权重（建议 `0.0~0.3`）

### 兼容性说明

1. 保持现有 LanceDB 存储结构兼容。
2. 新增配置均有默认值，不改配置也可运行。

## v0.3.0

本版本对齐上游 `memory-lancedb-pro` 的近期两项关键能力（v1.0.21 与 v1.0.22）：

### 新增与优化

1. 长文本嵌入增强（对齐 v1.0.21）：
   - 当嵌入请求因上下文过长失败时，自动进行语义分块（含重叠）并计算均值向量
   - 新增配置项：`embedding_chunking`（默认 `true`）
2. 存储路径健壮性增强（对齐 v1.0.22）：
   - 启动初始化时预校验 `db_path`
   - 支持目录自动创建、可写性检查、符号链接目标检查
3. 错误提示增强：
   - LanceDB 打开、建表、写入失败时，返回更可操作的排障信息

### 兼容性说明

1. 为增强型升级，不引入破坏性变更。
2. 保持与现有 LanceDB 表结构兼容。

## v0.2.0

本版本完成了 AstrBot 侧“持久化记忆插件”的中文化整理与能力增强，重点是让插件在生产场景下更稳定、更易理解。

### 新增与优化

1. 中文化整理完成：
   - 插件名称统一为 **astrbot持久化记忆插件**
   - `metadata.yaml`、`README.md`、变更说明均改为中文
2. 召回链路增强：
   - 新增 `auto_recall_min_length`，避免过短消息触发无效检索
3. 捕获链路防污染：
   - 自动过滤“删除/清理记忆”等管理意图，防止误写入长期记忆
4. 混合检索稳健性提升：
   - 增加 BM25 stale 命中校验，降低幽灵记忆风险
5. 中文语义支持增强：
   - 扩展中英（含繁体）触发词、分类词与召回跳过规则

### 兼容性说明

1. 仍使用 `jina-embeddings-v5-text-small` 作为默认嵌入模型。
2. 仍使用 `jina-reranker-v2-base-multilingual` 作为默认重排模型。
3. 持续兼容 LanceDB 持久化存储结构。

## v0.1.0

首个 AstrBot 可用版本，包含：

1. 自动记忆捕获与自动记忆召回注入
2. LanceDB 持久化存储
3. 混合检索与重排序
4. 记忆管理工具：
   - `memory_recall`
   - `memory_store`
   - `memory_forget`
   - `memory_list`
   - `memory_stats`
