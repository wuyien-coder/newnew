# 验证报告

运行测试和评测后在本目录生成报告。运行数据默认不提交，避免携带本机路径、会话内容或服务配置。

delivery_validation.json 记录本次交付的真实 BGE 全量检索、外部模型问答、ReAct 流式回答及会话保存结果。该项使用独立 SQLite 和内存向量验证本仓库代码；不代表已在此电脑启动 Docker。容器完整栈部署后使用 scripts/verify_full.py 单独验收 PostgreSQL/pgvector、Redis 与 Celery。
