import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import { getJSON, getToken, mutateJSON, saveToken, sendFeedback, streamAnswer, uploadDocument } from './api'
import type { Answer, DocumentItem, Job, SessionDetail, SessionItem, Stats } from './api'

type View = 'chat' | 'documents' | 'architecture'
type Message = { id: string; role: 'user' | 'assistant'; text: string; meta?: Answer; seq?: number; incomplete?: string }
type RuntimeStats = Stats & { config: Stats['config'] & { queue?: string; database?: string; cache_enabled?: boolean } }
const prompts = ['华罗庚在数论方面有哪些贡献？', '苏步青师从谁？', '研究微分几何的有哪些人？']
const iconPaths: Record<string, string> = {
  chat: 'M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.4 8.4 0 0 1 3.8-.9h.5a8.5 8.5 0 0 1 8 8v.5Z',
  documents: 'M4 19.5A2.5 2.5 0 0 1 6.5 17H20M6.5 3H20v19H6.5A2.5 2.5 0 0 1 4 19.5v-14A2.5 2.5 0 0 1 6.5 3Z',
  architecture: 'M12 3v5M5 21v-5m14 5v-5M5 16H2v-5h6v5H5Zm14 0h-3v-5h6v5h-3ZM12 8H9V3h6v5h-3Zm-7 3V9h14v2',
  arrow: 'm5 12 7-7 7 7M12 5v15', plus: 'M12 5v14M5 12h14',
  settings: 'M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8ZM12 2v3m0 14v3M2 12h3m14 0h3M5 5l2 2m10 10 2 2M5 19l2-2M17 7l2-2',
  check: 'm5 12 4 4L19 6', upload: 'M12 16V3m-5 5 5-5 5 5M4 16v5h16v-5',
}
function Icon({ name, size = 20 }: { name: string; size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={iconPaths[name] ?? iconPaths.chat} /></svg>
}
const confidenceLabel: Record<string, string> = { low: '较低', medium: '中等', high: '较高' }
const toolLabels: Record<string, string> = { text_search: '原文检索', person_profile: '人物资料', graph_neighbors: '师承与关系查询', graph_path: '关系路径', find_people: '领域与籍贯查询' }
const argumentLabels: Record<string, string> = { name: '人物', person: '限定人物', query: '检索词', relation: '关系类型', direction: '查询方向', source: '起点', target: '终点', keyword: '关键词', attribute: '属性', offset: '起始位置', limit: '返回条数', max_hops: '最大跳数' }
function argumentText(key: string, value: unknown): string {
  if (key === 'direction') return ({ in: '入向（指向该人物）', out: '出向（从该人物出发）', both: '入向与出向' } as Record<string, string>)[String(value)] ?? String(value)
  if (key === 'relation') return ({ teacher: '师从（学生 → 老师）', student: '培养学生（老师 → 学生）', advisor: '导师' } as Record<string, string>)[String(value)] ?? String(value)
  return typeof value === 'string' ? value : JSON.stringify(value)
}

function normalizeMath(text: string) {
  return text.replaceAll(String.raw`\[`, () => '$$').replaceAll(String.raw`\]`, () => '$$')
    .replaceAll(String.raw`\(`, '$').replaceAll(String.raw`\)`, '$')
}

export default function App() {
  const [view, setView] = useState<View>('chat')
  const [stats, setStats] = useState<RuntimeStats | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [session, setSession] = useState<string>()
  const [sessions, setSessions] = useState<SessionItem[]>([])
  const [loadingSession, setLoadingSession] = useState(false)
  const [historyOpen, setHistoryOpen] = useState(() => window.innerWidth > 1050)
  const [retrying, setRetrying] = useState<string>()
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const [graph, setGraph] = useState(true)
  const [selected, setSelected] = useState<string>()
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [token, setToken] = useState(getToken)
  const [documents, setDocuments] = useState<DocumentItem[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [uploading, setUploading] = useState(false)
  const abort = useRef<AbortController | null>(null)
  const scrollEnd = useRef<HTMLDivElement>(null)
  const fileInput = useRef<HTMLInputElement>(null)
  const sessionLoadVersion = useRef(0)

  async function loadStats() { try { setStats(await getJSON<RuntimeStats>('/stats')) } catch (e) { setError((e as Error).message) } }
  async function loadSessions() { try { const data = await getJSON<{ sessions: SessionItem[] }>('/sessions?limit=30'); setSessions(data.sessions) } catch (e) { setError((e as Error).message) } }
  async function openSession(id: string) {
    if (busy) return
    const version = ++sessionLoadVersion.current
    setLoadingSession(true); setError(''); setNotice('')
    try {
      const data = await getJSON<SessionDetail>(`/sessions/${encodeURIComponent(id)}`)
      if (version !== sessionLoadVersion.current) return
      setSession(id); setSelected(undefined); setView('chat')
      setMessages(data.messages.map(m => ({ id: `history-${m.seq}`, role: m.role, text: m.content, seq: m.seq, meta: { citations: m.citations, ...m.agent_metadata } })))
      sessionStorage.setItem('mathknow.session', id)
    } catch (e) { if (version === sessionLoadVersion.current) setError((e as Error).message) }
    finally { if (version === sessionLoadVersion.current) setLoadingSession(false) }
  }
  async function deleteSession(id: string) {
    if (busy || loadingSession) return
    try {
      await mutateJSON(`/sessions/${encodeURIComponent(id)}`, 'DELETE')
      if (session === id) freshConversation()
      await loadSessions(); setNotice('会话已删除。')
    } catch (e) { setError((e as Error).message) }
  }
  async function loadDocuments() {
    try {
      const [docs, tasks] = await Promise.all([getJSON<DocumentItem[]>('/documents/list'), getJSON<Job[]>('/documents/jobs')])
      setDocuments(docs); setJobs(tasks)
      await loadStats()
    } catch (e) { setError((e as Error).message) }
  }
  useEffect(() => {
    void loadStats(); void loadDocuments(); void loadSessions()
    const saved = sessionStorage.getItem('mathknow.session')
    if (saved) {
      void openSession(saved)
    }
    return () => abort.current?.abort()
  }, [])
  useEffect(() => { scrollEnd.current?.scrollIntoView({ block: 'nearest' }) }, [messages])
  useEffect(() => {
    if (view !== 'documents') return
    void loadDocuments()
    const timer = window.setInterval(() => { void loadDocuments() }, 2000)
    return () => window.clearInterval(timer)
  }, [view])

  async function ask(input = question) {
    if (busy || loadingSession || !input.trim()) return
    setError(''); setNotice(''); setQuestion(''); setBusy(true); setView('chat')
    const id = crypto.randomUUID()
    setSelected(id)
    setMessages(previous => [...previous, { id: `${id}-user`, role: 'user', text: input.trim() }, { id, role: 'assistant', text: '' }])
    const controller = new AbortController(); abort.current = controller
    try {
      await streamAnswer(input.trim(), session, graph, (event, data) => {
        setMessages(previous => previous.map(m => m.id !== id ? m : {
          ...m, text: event === 'token' ? m.text + (data.text ?? '') : event === 'done' ? data.answer ?? m.text : m.text,
          meta: event === 'retrieval' || event === 'done' ? { ...m.meta, ...data } : m.meta,
        }))
        if (event === 'done' && data.session_id) { setSession(data.session_id); sessionStorage.setItem('mathknow.session', data.session_id) }
      }, controller.signal)
      void loadStats(); void loadSessions()
    } catch (e) {
      const message = controller.signal.aborted ? '已停止生成。未完成的回答不会保存到会话。' : (e as Error).message
      setError(message)
      setMessages(previous => previous.map(m => m.id === id ? { ...m, incomplete: message } : m))
    } finally { setBusy(false); abort.current = null }
  }
  async function upload(file?: File) {
    if (!file) return
    if (!/\.(txt|md|pdf)$/i.test(file.name)) { setError('请选择 .txt、.md 或可提取文本的 .pdf 文档'); return }
    if (file.size > 20 * 1024 * 1024) { setError('文件不能超过 20MB'); return }
    setUploading(true); setError(''); setNotice('')
    try { const result = await uploadDocument(file); setNotice(`任务 #${result.job_id} 已提交，完成后可在问答中检索。`); await loadDocuments(); void loadStats() }
    catch (e) { setError((e as Error).message) }
    finally { setUploading(false); if (fileInput.current) fileInput.current.value = '' }
  }
  async function retryJob(id: string) {
    setRetrying(id); setError('')
    try { await mutateJSON(`/documents/jobs/${encodeURIComponent(id)}/retry`, 'POST'); setNotice(`任务 #${id} 已重新提交。`); await loadDocuments() }
    catch (e) { setError((e as Error).message) }
    finally { setRetrying(undefined) }
  }
  async function removeDocument(document: DocumentItem) {
    if (!window.confirm(`删除“${document.person_name}”的索引？原始文件会保留。`)) return
    setError('')
    try {
      await mutateJSON(`/documents/${document.id}`, 'DELETE')
      setNotice('文档已从知识库移除。'); await loadDocuments()
    } catch (e) { setError((e as Error).message) }
  }
  const active = messages.find(m => m.id === selected) ?? [...messages].reverse().find(m => m.role === 'assistant')
  const citations = active?.meta?.citations ?? []
  const offline = !stats || stats.llm_provider === 'stub'
  const freshConversation = () => { if (busy) return; ++sessionLoadVersion.current; setLoadingSession(false); setMessages([]); setSession(undefined); setSelected(undefined); sessionStorage.removeItem('mathknow.session'); setView('chat'); setError('') }

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="/web/" aria-label="MathKnow 首页"><span className="brand-symbol">M<span>·</span></span><div>MathKnow<small>领域知识工作台</small></div></a>
      <button className="new-chat" onClick={freshConversation} disabled={busy}><Icon name="plus" />新建对话</button>
      <p className="nav-label">工作空间</p>
      <nav>{([['chat', '知识问答'], ['documents', '知识库'], ['architecture', '技术架构']] as [View, string][]).map(([key, label]) => <button key={key} className={view === key ? 'active' : ''} onClick={() => setView(key)}><Icon name={key} />{label}{key === 'documents' && <span className="nav-count">{documents.length || '—'}</span>}</button>)}</nav>
      <details className="session-history" open={historyOpen}><summary onClick={e => { e.preventDefault(); setHistoryOpen(!historyOpen) }}>历史会话 <span>{sessions.length}</span><b>{historyOpen ? '−' : '+'}</b></summary><div className="session-history-actions"><button disabled={busy} onClick={freshConversation}>＋ 新建对话</button><button onClick={() => void loadSessions()}>刷新</button></div><div className="session-list">{!sessions.length ? <p>开始提问后，会话会自动保存。</p> : sessions.map(s => <div key={s.id} className={`session-row ${s.id === session ? 'selected' : ''}`}><button className="session-title" disabled={busy || loadingSession} onClick={() => void openSession(s.id)} title={s.title}><strong>{s.title || '未命名对话'}</strong><small>{s.turn_count} 轮 · {s.updated_at.slice(0, 10)}</small></button><button className="session-delete" disabled={busy || loadingSession} aria-label={`删除会话：${s.title}`} onClick={() => void deleteSession(s.id)}>×</button></div>)}</div></details>
      <div className="sidebar-note"><span className="tiny-dot" />数学人物 · 知识分析<p>从数学人物传记中检索知识，<br />让每个回答都有据可循。</p><span className="mini-tag">RAG</span><span className="mini-tag">知识图谱</span></div>
      <button className="settings-button" onClick={() => setSettingsOpen(true)}><Icon name="settings" />连接设置<span>↗</span></button>
      <div className="profile"><span className="avatar">研</span><div>个人研究空间<small>数学人物 · 知识分析</small></div><span className="profile-dot" /></div>
    </aside>
    <main className="workspace">
      <header className="topbar"><div className="breadcrumb">工作空间 <span>/</span> <strong>{{ chat: '知识问答', documents: '知识库', architecture: '技术架构' }[view]}</strong></div><div className="top-status"><span className="tiny-dot" />{stats ? '服务已连接' : '连接中'}<span className={`mode-badge ${offline ? '' : 'real'}`} title={stats?.llm_model}>{offline ? '离线演示' : stats?.llm_model || '模型服务'}</span><button className="mobile-settings" aria-label="连接设置" onClick={() => setSettingsOpen(true)}><Icon name="settings" size={16} /></button></div></header>
      {(error || notice) && <div className={`notification ${error ? 'error' : ''}`} role="status">{error || notice}<button aria-label="关闭提示" onClick={() => { setError(''); setNotice('') }}>×</button></div>}

      {view === 'chat' && <div className="chat-layout">
        <section className="chat-pane">
          <div className="section-heading"><div><span className="eyebrow">KNOWLEDGE ASSISTANT</span><h1>知识问答</h1><p>连接人物、文献与知识背后的关系。</p></div><span className="domain-label">数学人物知识库</span></div>
          <div className="conversation">
            {loadingSession ? <div className="loading-history" role="status">正在读取会话与参考依据…</div> : !messages.length ? <div className="welcome"><div className="welcome-icon"><Icon name="documents" size={32} /><span>✦</span></div><h2>从一个问题，发现更多知识</h2><p>探索数学家的生平、学术贡献与师承关系。<br />回答将附上原文摘录和图谱证据，供你进一步查阅。</p><div className="prompt-grid">{prompts.map((prompt, i) => <button key={prompt} onClick={() => void ask(prompt)}><span>0{i + 1}<Icon name={i === 1 ? 'architecture' : 'documents'} size={18} /></span>{prompt}<b>↗</b></button>)}</div><div className="welcome-foot"><Icon name="check" size={14} />混合检索<span>·</span>图谱增强<span>·</span>引用溯源</div></div>
              : <div className="messages">{messages.map((m, index) => <article key={m.id} className={`message ${m.role}`} onClick={() => m.role === 'assistant' && setSelected(m.id)}><div className="message-avatar">{m.role === 'user' ? '我' : <Icon name="documents" size={18} />}</div><div className="message-content"><div className="message-label">{m.role === 'user' ? '你' : 'MathKnow'}{m.role === 'assistant' && <span>{offline ? '离线资料摘录' : '基于知识库回答'}</span>}</div><div className="message-text">{(m.text ? (m.role === 'assistant' ? <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{normalizeMath(m.text)}</ReactMarkdown> : m.text) : null) || (m.incomplete ? null : <span className="thinking"><i /><i /><i />正在检索知识与生成回答</span>)}</div>{m.incomplete && <div className="incomplete-answer" role="status">{m.incomplete}<button disabled={busy || loadingSession} onClick={e => { e.stopPropagation(); void ask(messages[index - 1]?.text) }}>重新提问</button></div>}{m.role === 'assistant' && m.text && <div className="message-actions"><button onClick={e => { e.stopPropagation(); setSelected(m.id) }}>{m.meta?.citations?.length ?? 0} 条参考依据 ↗</button>{m.meta?.engine && <span>{m.meta.engine}</span>}{session && <button onClick={e => { e.stopPropagation(); void sendFeedback(session, m.seq ?? index, 1).then(() => setNotice('反馈已保存，谢谢。')).catch(e => setError(e.message)) }}>有帮助</button>}</div>}</div></article>)}</div>}
            <div ref={scrollEnd} />
          </div>
          <div className="composer-area"><div className="composer"><textarea aria-label="输入问题" disabled={loadingSession} maxLength={500} value={question} onChange={e => setQuestion(e.target.value)} placeholder="向知识库提问，例如：华罗庚与熊庆来有什么关系？" onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); void ask() } }} /><div className="composer-bottom"><button className={`graph-switch ${graph ? 'enabled' : ''}`} onClick={() => setGraph(!graph)} aria-pressed={graph}><Icon name="architecture" size={15} />图谱增强{graph && <Icon name="check" size={13} />}</button><span className="composer-hint">Enter 发送 · Shift + Enter 换行</span>{busy ? <button className="stop-button" onClick={() => abort.current?.abort()}>停止生成</button> : <button className="send-button" aria-label="发送问题" disabled={!question.trim() || loadingSession} onClick={() => void ask()}><Icon name="arrow" size={18} /></button>}</div></div><p className="composer-disclaimer">{offline ? '当前为离线抽取式演示，不调用付费模型。' : '模型回答请结合原文引用核实。'} 检索强度仅供参考。</p></div>
        </section>
        <aside className="evidence-pane"><div className="evidence-title"><Icon name="documents" size={18} /><h2>参考依据</h2><span>{citations.length.toString().padStart(2, '0')}</span></div><p className="evidence-subtitle">回到原文，验证每一处知识。</p>
          {!!active?.meta?.agent_trace?.length && <details className="agent-trace" open><summary>ReAct 工具调用 · {active.meta.agent_trace.length} 步</summary><ol>{active.meta.agent_trace.map(step => <li key={step.step}><div className="trace-step-heading"><strong>{toolLabels[step.tool] ?? step.tool}</strong><span className="trace-duration">{Math.round(step.duration_ms)} ms</span></div><p className="trace-result">{step.status === 'ok' ? (step.result_summary ?? '调用完成（历史记录未保存命中数）') : '参数或调用需修正'}</p><dl className="trace-arguments">{Object.entries(step.arguments).map(([key, value]) => <div key={key} style={{ display: 'contents' }}><dt>{argumentLabels[key] ?? key}</dt><dd>{argumentText(key, value)}</dd></div>)}</dl></li>)}</ol><p>{active.meta.agent_stop === 'step_limit' ? '已达调用上限，依据已有证据回答。' : '检索结束，依据工具返回结果回答。'}</p></details>}
          {citations.length ? <><div className="evidence-meta"><span>检索参考强度</span><b>{confidenceLabel[active?.meta?.confidence ?? ''] ?? '历史记录'}</b></div><div className="citation-list">{citations.map(c => <div className="citation-card" key={`${c.index}-${c.chunk_id}`}><div className="citation-top"><span className="reference-number">{c.index}</span><strong>{c.person || '知识库资料'}</strong><span className="source-kind">{(c.chunk_id.startsWith('graph:') || c.chunk_id.startsWith('react-graph:')) ? '图谱' : '原文'}</span></div><p>{c.snippet}</p><footer>{c.section || '正文'}{c.page ? ` · 第 ${c.page} 页` : ''}</footer><small>{c.source || ((c.chunk_id.startsWith('graph:') || c.chunk_id.startsWith('react-graph:')) ? '人物关系表' : '本地文档')}</small></div>)}</div></> : <div className="empty-evidence"><span>“ ”</span><h3>让回答有据可循</h3><p>提问后，这里会展示参考文献、<br />原文摘录与人物关系证据。</p><div className="empty-lines"><i /><i /><i /></div></div>}
          <div className="evidence-footer"><div><span className="tiny-dot" />{busy ? '正在检索与生成' : '知识库已就绪'}<b>{active?.meta?.timings_ms?.total ? `${Math.round(active.meta.timings_ms.total)} ms` : 'RAG + Graph'}</b></div>{active?.meta?.timings_ms && <dl className="timings">{Object.entries(active.meta.timings_ms).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{Math.round(value)} ms</dd></div>)}</dl>}<p>{stats?.index.documents ?? documents.length} 文档 <span>·</span> {stats?.index.chunks ?? '—'} 文本块 <span>·</span> {stats?.index.persons ?? '—'} 人物实体 <span>·</span> {stats?.index.relations ?? '—'} 关系</p></div>
        </aside>
      </div>}

      {view === 'documents' && <div className="page-content"><div className="page-title"><div><span className="eyebrow">KNOWLEDGE BASE</span><h1>管理你的知识来源</h1><p>文档摄取、索引状态与可追溯内容，一目了然。</p></div><button className="primary-button" disabled={uploading} onClick={() => fileInput.current?.click()}><Icon name="plus" size={17} />{uploading ? '正在上传…' : '上传文档'}</button></div><input className="file-input" ref={fileInput} type="file" accept=".txt,.md,.pdf" onChange={e => void upload(e.target.files?.[0])} />
        <div className="stat-grid">{[['已入库文档', stats?.index.documents ?? documents.length], ['文本分块', stats?.index.chunks ?? '—'], ['人物实体', stats?.index.persons ?? '—'], ['图谱关系', stats?.index.relations ?? '—']].map(([label, value]) => <div className="stat-card" key={label}><span>{label}</span><strong>{value}</strong><small>来自当前运行实例</small></div>)}</div>
        <div className="upload-banner"><div className="upload-icon"><Icon name="upload" size={23} /></div><div><h3>把新资料加入知识库</h3><p>支持 .txt / .md / PDF，单文件最多 20MB。PDF 需包含可提取文本；扫描件请先转为文本。</p></div><button onClick={() => fileInput.current?.click()} disabled={uploading}>选择文件 ↗</button></div>
        <section className="surface"><div className="surface-heading"><h2>文档列表 <span>{documents.length}</span></h2><button onClick={() => { void loadDocuments(); void loadStats() }}>刷新</button></div><div className="table-wrap"><table><thead><tr><th>文档 / 人物</th><th>来源</th><th>文本块</th><th>字符</th><th>状态</th><th>操作</th></tr></thead><tbody>{documents.map(d => <tr key={d.id}><td><span className="document-icon"><Icon name="documents" size={16} /></span><strong>{d.person_name}</strong></td><td>{d.source || '本地资料'}</td><td>{d.chunks}</td><td>{d.char_count.toLocaleString()}</td><td><span className="status-pill">已入库</span></td><td><button className="document-delete" onClick={() => void removeDocument(d)}>删除</button></td></tr>)}</tbody></table></div></section>
        <section className="surface"><div className="surface-heading"><h2>摄取任务</h2><span>自动刷新 · {stats?.config.queue === 'celery' ? 'Celery' : '本地任务'}</span></div>{!jobs.length ? <div className="empty-jobs">上传一份新文档，在这里查看摄取进度。</div> : <div className="job-list">{jobs.map(j => <div className="job-row" key={j.id}><span className="job-id">#{j.id}</span><div><strong>{j.path.split(/[\\/]/).pop()}</strong><small>{j.error || `${j.chunks ?? 0} 个文本块 · ${j.started_at?.slice(0, 19).replace('T', ' ') ?? ''}`}</small></div><span className={`job-status ${j.status}`}>{{ pending: '等待处理', running: '处理中', done: '已完成', failed: '失败' }[j.status] ?? j.status}</span>{j.status === 'failed' && <button className="retry-job" disabled={retrying === j.id} onClick={() => void retryJob(j.id)}>{retrying === j.id ? '提交中…' : '重试'}</button>}</div>)}</div>}</section>
      </div>}

      {view === 'architecture' && <div className="page-content architecture-page"><div className="page-title"><div><span className="eyebrow">UNDER THE HOOD</span><h1>从论文方法，到工程系统</h1><p>保留领域研究特色，用可运行的服务串起完整问答链路。</p></div><span className="domain-label">ReAct Agent · v2.2</span></div><div className="architecture-flow">{[['React + TypeScript', '交互与引用展示'], ['FastAPI + ReAct', '模型决策与工具循环'], ['混合检索 + 图谱', '召回、融合与证据'], ['兼容模型接口', '生成与流式返回']].map(([name, detail], i) => <div key={name}><span>0{i + 1}</span><h3>{name}</h3><p>{detail}</p>{i < 3 && <b>→</b>}</div>)}</div><section className="surface"><div className="surface-heading"><h2>当前运行配置</h2><span className="mode-badge">{offline ? '离线演示' : '模型已连接'}</span></div><div className="runtime-grid">{[['业务数据库', stats?.config.database ?? '—'], ['向量存储', stats?.index.vector_store ?? '—'], ['向量编码', stats?.index.embedder ?? '—'], ['摄取任务', stats?.config.queue ?? '—'], ['检索缓存', stats?.config.cache_enabled ? 'Redis' : '关闭'], ['模型提供方', stats?.llm_provider ?? '—'], ['生成模型', stats?.llm_model ?? '—'], ['知识库文档', stats?.index.documents ?? documents.length], ['文本块', stats?.index.chunks ?? '—']].map(([name, value]) => <div key={name}><span>{name}</span><strong>{value}</strong></div>)}</div></section><div className="architecture-cards"><section className="surface architecture-card"><span className="card-number">01 / RETRIEVAL</span><h2>有领域特色的检索</h2><p>章节感知分块、稠密向量与 BM25 双路召回、加权融合（支持 RRF），以及可选的 BGE 精排。人物关系表通过 NetworkX 支持师承、路径与领域聚合。</p><div className="tag-row"><span>BM25</span><span>Weighted / RRF</span><span>GraphRAG</span></div></section><section className="surface architecture-card"><span className="card-number">02 / ENGINEERING</span><h2>可切换的服务架构</h2><p>已实现 PostgreSQL / pgvector HNSW、Redis 检索缓存和 Celery 摄取队列，配套 Alembic 迁移、Docker Compose 与 Nginx。当前实际组件以运行配置为准。</p><div className="tag-row"><span>pgvector</span><span>Redis</span><span>Celery</span></div></section><section className="surface architecture-card"><span className="card-number">03 / VALIDATION</span><h2>可以检验的工程结果</h2><p>单轮用量、流式文本、引用编号检查与索引版本更新均有可验证实现。完整模式使用全量语料与真实 BGE 编码，支持异步摄取、历史会话与可追溯引用。</p><div className="tag-row"><span>pytest</span><span>Alembic</span><span>SSE</span></div></section></div><div className="architecture-note"><Icon name="check" size={18} /><p>当前运行：{stats?.config.database ?? '读取中'} · {stats?.index.vector_store ?? '读取中'} · {stats?.llm_model ?? stats?.llm_provider ?? '读取中'} · {stats?.index.documents ?? documents.length} 份文档。可在连接设置中刷新服务状态，部署说明见项目 README。</p></div></div>}
    </main>
    {settingsOpen && <div className="modal-backdrop" onClick={() => setSettingsOpen(false)}><div className="settings-modal" role="dialog" aria-modal="true" aria-labelledby="settings-title" onClick={e => e.stopPropagation()}><span className="eyebrow">CONNECTION</span><h2 id="settings-title">连接设置</h2><p>如果服务启用了访问令牌，请填写对应的 Bearer token。令牌仅保存在当前浏览器会话中。</p><label>访问令牌<input type="password" autoComplete="off" value={token} onChange={e => setToken(e.target.value)} placeholder="本地离线演示可留空" /></label><div className="modal-actions"><button onClick={() => setSettingsOpen(false)}>取消</button><button className="primary-button" onClick={() => { saveToken(token); setSettingsOpen(false); setError(''); void loadStats(); void loadDocuments(); void loadSessions() }}>保存并连接</button></div></div></div>}
  </div>
}
