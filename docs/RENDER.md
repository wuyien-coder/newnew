# Render 部署完整版 MathKnow

本方案使用原有 React + FastAPI、真实 BGE + BM25 和全量知识库，保留上传、后台摄取、持久会话和引用。它不依赖 Railway。

## 状态与费用

已添加 Render Blueprint 和 Docker 配置，尚未执行 Render 构建或上线验收。此配置创建付费 2 CPU / 4 GB 服务和 10 GB 持久磁盘。请在 Render 确认实际费用后再创建；不要将其理解为免费服务。自动部署已关闭。

## 官网部署，不需要 ChatGPT 插件

1. 登录 https://dashboard.render.com/ ，选择 New → Blueprint，并连接 GitHub。
2. 选择仓库 wuyien-coder/newnew，分支 codex/mathknow-online-full，Blueprint 路径 render.yaml。不要选择尚未包含该配置的 main 分支。
3. 审阅资源、区域和费用后创建。构建时下载固定版本 BGE 并预热 9,921 个文本块，首次构建可能较久。
4. 服务显示 Live 后，使用 Render 实际分配的 HTTPS 域名访问 /web/。
5. 在服务 Environment 中查看自动生成的 MATHKB_SERVER__API_TOKEN，自行复制到网页配置中的「网站访问令牌」，不要发到聊天或提交到 Git。
6. 在网页填入 DeepSeek / 千问的模型 Key，测试连接后应用。模型 Key 仍按请求传递，刷新页面后清除，不需要设为 Render 环境变量。

网站访问令牌保护的是共享工作空间。持有令牌的体验者可访问共享会话和上传资料；请只向受信任的体验者提供令牌。

## 部署细节

- deploy/Dockerfile.render 复用在线镜像构建流程，启动时只调整磁盘根目录权限，再以 UID 10001 运行应用。
- /data 保存 SQLite、上传资料和向量缓存。模型及预热缓存在镜像中，运行时复制缓存到磁盘。
- 容器监听平台 PORT（Blueprint 设置为 10000），健康检查为 /health。
- 单实例与单 API 进程保持本地队列和内存索引一致。
- 更新代码后在 Render 手动部署，避免每次提交都触发付费资源构建。

## 必须完成的线上验收

1. /health 显示正常及 BgeEmbedding、全量语料计数。
2. 未携带网站令牌的 /api/v1/stats 返回 401。
3. 网页使用有效模型 Key 完成真实问答并查看引用。
4. 上传唯一测试文本，等待索引完成后检索，再删除。
5. 重启后确认会话、引用、上传资料仍存在。

当前环境没有 Docker，尚未完成容器构建；配置检查不替代 Render 实际构建与上述验收。

官方参考：https://render.com/docs/blueprint-spec 、https://render.com/docs/disks
