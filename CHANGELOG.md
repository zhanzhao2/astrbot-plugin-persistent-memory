# 变更记录

## v0.4.2

1. 对齐上游 v1.0.32：`normalize_retrieval_query` 改为全局剥离 `Conversation info/Sender (untrusted metadata)` 区块。
2. 对齐上游 v1.0.32：`should_capture` 在判定前先清理 OpenClaw metadata block，减少噪声记忆误写入。
3. `auto_recall_min_repeated` 解析漏洞在 AstrBot 版此前已规避，本次无需额外修补。

## v0.4.1

1. 严格对齐上游 v1.0.26：补齐 `reinforcement_factor` / `max_half_life_multiplier`，在 time-decay 中按 `accessCount/lastAccessedAt` 计算有效 half-life。
2. 严格对齐上游 v1.0.26：手动 `memory_recall` 结果写回访问元数据（debounce flush），不再在 auto-recall 中记录访问强化。
3. 严格对齐上游 v1.0.28：新增 `auto_recall_min_repeated`（按会话轮次去重间隔，默认 0）。
4. 严格对齐上游 v1.0.29：`normalize_retrieval_query` 增强，补齐 `Conversation info/Sender (untrusted metadata)` 与 `[cron:...]` 清洗。
5. 同步补齐配置与文档字段，确保实现与说明一致。

## v0.4.0

1. 合并上游 v1.0.24 / v1.0.25 思路：新增 `embedding_api_keys` 多 Key 轮转与 `retry_on_rate_limit` 限流重试切换。
2. 合并上游 v1.0.28 思路：自动召回增加跨轮次去重（`recall_cross_turn_dedup` + `recall_dedup_window_sec`）。
3. 合并上游 v1.0.29 思路：增强 `normalize_retrieval_query`，清洗角色前缀、引用标记和包装标签。
4. 合并上游 v1.0.30 思路：写入去重预检失败时 fail-open，继续存储并告警，降低误拦截。
5. 合并上游 v1.0.26 思路：新增访问强化排序（按近期访问次数/时效进行轻量加权），并开放 `access_boost_weight` 配置。

## v0.3.0

1. 合并上游 v1.0.21 思路：新增超长文本自动分块嵌入（`embedding_chunking`，默认开启）。
2. 合并上游 v1.0.22 思路：新增 `db_path` 预校验（目录创建、可写性、符号链接目标检查）。
3. 改进存储异常提示：LanceDB 打开/建表/写入失败时给出更可操作的报错信息，便于排障。
4. README 增补“自动分块嵌入”和“存储路径预校验”说明。

## v0.2.0

1. 将插件展示名调整为“astrbot持久化记忆插件”。
2. 增加完整中文 README，补充原理、流程、配置、工具与使用说明。
3. 增加中文 Release 说明。
4. 自动捕获新增元指令排除规则，避免将“删除/清理记忆”类操作误写入记忆库。
5. 检索新增 BM25 stale 命中校验，降低幽灵记忆召回概率。
6. 新增 `auto_recall_min_length` 配置项，并在请求前召回阶段生效。
7. 扩展中英触发词、决策分类词与跳过/强制检索关键词（含繁体中文）。

## v0.1.0

1. 基于 AstrBot 实现首版持久化记忆插件。
2. 支持 LanceDB 存储、自动捕获、自动召回、混合检索与重排序。
3. 提供 `memory_recall` / `memory_store` / `memory_forget` / `memory_list` / `memory_stats` 工具。
