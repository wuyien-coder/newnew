# MathKnow 前端

React + TypeScript + Vite，已包含构建后的 `dist/`，由 FastAPI 在 `/web/` 提供页面。

开发环境：Node.js 22+（含 npm）。

```powershell
npm ci
npm run dev
```

开发地址为 http://127.0.0.1:5173/web/，接口转发到 8000 端口。后端启动方式见根 README。

```powershell
npm run build
```

功能：SSE 问答、当前会话回看、引用摘录、图谱开关、答案反馈、文档上传与任务状态、实际运行配置、访问令牌设置。页面访问令牌仅保存在 sessionStorage。

浏览器验证：问答/引用、连续追问、上传后检索、架构页和手机布局；见 `reports/validation.json`。
