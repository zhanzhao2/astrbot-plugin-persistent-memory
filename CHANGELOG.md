# 变更记录

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
