export interface Citation {
  index: number
  chunk_id: string
  person: string
  section: string
  source: string
  page: string
  score: number
  snippet: string
}
export interface Stats {
  index: { documents?: number; chunks?: number; persons?: number; relations?: number; embedder?: string; vector_store?: string }
  llm_provider: string
  llm_model?: string
  config: { embedding?: string; vector_store?: string; rerank_enabled?: boolean; kg_provider?: string }
  usage?: Record<string, number>
}
export interface Answer {
  agent_trace?: { step: number; tool: string; arguments: Record<string, unknown>; status: string; duration_ms: number; evidence_count: number; result_count?: number; result_summary?: string }[]
  agent_stop?: string
  session_id?: string
  answer?: string
  citations?: Citation[]
  confidence?: string
  entities?: string[]
  timings_ms?: Record<string, number>
  retrieval?: { pool_size?: number; hits?: number }
  usage?: Record<string, number>
  cost_yuan?: number
  notes?: string[]
  engine?: string
}
export interface DocumentItem { id: string; person_name: string; source: string; char_count: number; chunks: number; status: string }
export interface Job { id: string; path: string; status: string; chunks?: number; error?: string; started_at?: string }
export interface SessionItem { id: string; title: string; turn_count: number; updated_at: string }
export interface SessionDetail { messages: { role: 'user' | 'assistant'; content: string; citations?: Citation[]; seq: number; agent_metadata?: Pick<Answer, 'engine' | 'agent_trace' | 'agent_stop'> }[] }

const tokenKey = 'mathknow.token'
export const getToken = () => sessionStorage.getItem(tokenKey) ?? ''
export function saveToken(token: string) { token ? sessionStorage.setItem(tokenKey, token.trim()) : sessionStorage.removeItem(tokenKey) }
function headers() { const token = getToken(); return token ? { Authorization: `Bearer ${token}` } : undefined }

async function ensureOk(response: Response) {
  if (response.ok) return response
  let message = `请求失败（${response.status}）`
  try {
    const data = await response.json()
    const detail = data.detail ?? data.error?.message ?? data.error ?? data.message
    if (typeof detail === 'string') message = detail
    else if (detail) message = JSON.stringify(detail)
  } catch { /* 保留 HTTP 状态。 */ }
  if (response.status === 401 || response.status === 403) message = '身份验证失败，请在设置中填写有效的访问令牌。'
  throw new Error(message)
}

export async function getJSON<T>(path: string, signal?: AbortSignal): Promise<T> {
  return (await ensureOk(await fetch(`/api/v1${path}`, { headers: headers(), signal }))).json()
}
export async function mutateJSON<T>(path: string, method: 'POST' | 'DELETE'): Promise<T> {
  return (await ensureOk(await fetch(`/api/v1${path}`, { method, headers: headers() }))).json()
}
export async function sendFeedback(sessionId: string, seq: number, rating: number) {
  await ensureOk(await fetch(`/api/v1/sessions/${encodeURIComponent(sessionId)}/feedback`, {
    method: 'POST', headers: { ...headers(), 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message_seq: seq, rating, comment: '' }),
  }))
}
export async function uploadDocument(file: File): Promise<{ job_id: string; status: string; path: string }> {
  const data = new FormData(); data.append('file', file)
  return (await ensureOk(await fetch('/api/v1/documents/upload', { method: 'POST', headers: headers(), body: data }))).json()
}

/** An incremental SSE parser: keep incomplete frames, including split CRLF delimiters. */
export class SSEParser {
  private buffer = ''
  constructor(private readonly emit: (event: string, payload: unknown) => void) {}
  push(text: string, final = false) {
    this.buffer += text
    let match: RegExpExecArray | null
    while ((match = /\r?\n\r?\n/.exec(this.buffer))) {
      const block = this.buffer.slice(0, match.index)
      this.buffer = this.buffer.slice(match.index + match[0].length)
      this.dispatch(block)
    }
    if (final && this.buffer.trim()) { this.dispatch(this.buffer); this.buffer = '' }
  }
  private dispatch(frame: string) {
    let event = 'message'
    const lines: string[] = []
    for (const line of frame.split(/\r?\n/)) {
      if (line.startsWith('event:')) event = line.slice(6).trim()
      if (line.startsWith('data:')) lines.push(line.slice(5).replace(/^ /, ''))
    }
    if (!lines.length) return
    const data = lines.join('\n')
    let payload: unknown
    try { payload = JSON.parse(data) } catch { throw new Error('服务返回了无法解析的事件，请检查后端日志。') }
    this.emit(event, payload)
  }
}

export async function streamAnswer(question: string, sessionId: string | undefined, useGraph: boolean,
  onEvent: (event: string, data: Answer & { text?: string }) => void, signal: AbortSignal) {
  const response = await ensureOk(await fetch('/api/v1/chat/stream', {
    method: 'POST', headers: { ...headers(), 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ question, session_id: sessionId, mode: 'agent', engine: 'auto', overrides: { use_graph: useGraph } }), signal,
  }))
  if (!response.body) throw new Error('当前环境不支持流式响应。')
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let doneReceived = false
  const parser = new SSEParser((event, raw) => {
    const data = raw as Answer & { text?: string; message?: string; detail?: string; error?: string }
    if (event === 'error') throw new Error(data.message ?? data.detail ?? data.error ?? '问答服务发生错误。')
    if (event === 'done') doneReceived = true
    onEvent(event, data)
  })
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) { parser.push(decoder.decode(), true); break }
      parser.push(decoder.decode(value, { stream: true }))
    }
    if (!doneReceived) throw new Error('响应连接提前结束，当前回答可能不完整，请重试。')
  } catch (error) { await reader.cancel().catch(() => undefined); throw error }
  finally { reader.releaseLock() }
}
