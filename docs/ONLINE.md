# 完整项目的在线部署

`deploy/Dockerfile.online` 直接运行本仓库的 React + FastAPI 应用。它与 Sites 展示版不同：使用原生 Python 检索、真实 BGE 编码和完整知识库，支持网页配置 DeepSeek / 千问、ReAct 工具调用、SSE 回答、引用溯源、持久会话、反馈，以及 TXT / MD / 文本 PDF 的上传、索引、任务重试和删除。

这份配置已准备好部署，**是否上线以托管平台的成功部署和实际访问验收为准**。

本地原生全量验证记录见 [online-validation.json](online-validation.json)。全量 BGE、混合检索、上传与重启后的会话 / 引用已通过检查；Docker 镜像构建和托管平台上的在线验收仍待执行。未提供有效模型 Key，因此真实付费模型生成尚未验收。

## Railway

1. 在 [Railway 官网](https://railway.com/new) 登录，直接选择 Deploy from GitHub repo，创建服务，选择 `wuyien-coder/newnew` 仓库及 `codex/mathknow-online-full` 分支。此流程不依赖 ChatGPT 插件。仓库根目录的 `railway.json` 指定 `deploy/Dockerfile.online`，无需填写额外构建命令。
2. 给应用服务添加一个挂载路径为 `/data` 的持久磁盘。数据库、文档、上传文件和 BGE 向量缓存都在该目录。不要使用临时文件系统代替持久磁盘。
3. 设置服务变量：

   | 变量 | 值 |
   | --- | --- |
   | `MATHKB_SERVER__API_TOKEN` | 自行生成的至少 24 个 ASCII 字符的访问令牌，作为部署平台的秘密变量保存 |
   | `RAILWAY_RUN_UID` | `0`，解决平台挂载磁盘的写权限 |

4. 为容器留出足够的 CPU / 内存；建议从 2 vCPU、4 GB 内存开始，并根据实际资源监控调整。模型下载和全量 BGE 编码在镜像构建时完成，构建日志会显示进度；首次构建可能较久。启动时将预热缓存复制到持久磁盘，后续仅编码新增或更改的资料，不会偷偷降级成 hash 向量。
5. 部署成功后生成 HTTPS 域名，访问 `/web/`。打开「配置 API Key」→ 展开「网站访问令牌」→ 填入上面的令牌。再选择模型服务商、地域、模型名，填写自己的模型 API Key，点击「测试连接」→「应用设置」。

网站访问令牌与模型 API Key 是两个用途不同的凭据。模型 Key 仅用于当前请求，不写入数据库或浏览器存储；页面刷新后重新填写。测试连接会产生一次简短模型调用，按模型账号计费。

这是**受访问令牌保护的共享工作空间**：持有网站令牌的人能查看会话、检索上传资料和管理文档。只向受信任的体验者分享令牌。它不是按用户隔离的多租户服务。

## 运行组件

| 功能 | 这个容器配置的实际组件 |
| --- | --- |
| 语料 | 仓库全量 326 份传记、335 个人物、355 条原始关系；初始 9,921 个文本块 |
| 稠密检索 | `BAAI/bge-small-zh-v1.5`，真实本地编码，512 维 |
| 关键词与图谱 | 原项目 BM25、融合检索、NetworkX、人物资料 / 关系 / 路径 / 领域查询工具 |
| 问答 | 原生 ReAct Agent、最多 4 步工具调用、实际模型正文 SSE |
| 会话 / 反馈 / 任务记录 | `/data/mathkb.db` 内的持久 SQLite 数据库 |
| 向量索引 | 内存索引；BGE 向量持久缓存，重启时恢复索引 |
| 上传与异步摄取 | 原生本地后台任务；中断的任务转为失败，可从网页重试 |
| Redis / Celery / pgvector | 这个单容器配置没有启用；原有生产 Compose 可切换这些组件 |
| BGE reranker | 可选组件，默认关闭；需要另行提供本地精排模型 |

需要 PostgreSQL / pgvector、Redis、Celery 的生产拓扑时，沿用 `docker-compose.yml` + `docker-compose.full.yml`，按 README 挂载本地模型。API 与 worker 的上传目录必须共享。不要在 Railway 上给两个服务分别挂载独立磁盘，再假设 worker 能访问另一个服务上传的文件。

## 普通容器服务器

```sh
docker build -f deploy/Dockerfile.online -t mathknow-online .
docker volume create mathknow-data
# 首次使用时为容器内 UID 10001 设置写权限。
docker run --rm --user root -v mathknow-data:/data --entrypoint chown mathknow-online 10001:10001 /data
docker run -d --name mathknow-online --restart unless-stopped \
  --env-file /secure/path/mathknow-online.env \
  -p 127.0.0.1:8000:8000 -v mathknow-data:/data mathknow-online
```

`mathknow-online.env` 至少包含 `MATHKB_SERVER__API_TOKEN=你生成的访问令牌`。在前面配置 HTTPS 反向代理，并关闭 SSE 缓冲。访问域名的 `/web/`；`/` 会跳转到该界面。容器宿主机重启后，`/data` 中的会话与资料仍然保留。

## 上线验收

- `/health` 返回 `status=ok`、`embedder=BgeEmbedding` 和全量文本块计数。
- 无访问令牌的 `/api/v1/stats` 请求返回 401。
- 用有效网站令牌读取运行配置；确认 BGE、全量语料和实际队列 / 数据库类型。
- 在网页输入有效模型 Key 后测试连接，再提问「华罗庚在数论方面有哪些贡献？」；确认实际模型回答、工具调用与引用。
- 追问、刷新网页并恢复历史；确认引用仍可查看。
- 上传一份包含唯一测试句的 TXT / 文本 PDF，任务完成后检索该句，再删除文档。
- 重启容器后检查历史、上传资料和缓存；不要以本地测试通过替代在线验收。

部署配置参考：[Railway Config as Code](https://docs.railway.com/config-as-code/reference)、[Railway Volumes](https://docs.railway.com/volumes/reference)。BGE 固定模型版本为 `7999e1d3359715c523056ef9478215996d62a620`，来源为 [BAAI 官方模型](https://huggingface.co/BAAI/bge-small-zh-v1.5)。
