import { useEffect, useRef, useState } from 'react'
import { getModelConnection, getToken, saveModelConnection, saveToken, testModelConnection, validateModelConnection } from './api'
import type { ModelConnection } from './api'

const presets = {
  deepseek: { label: 'DeepSeek', provider: 'deepseek', base_url: 'https://api.deepseek.com/v1', model: 'deepseek-chat' },
  beijing: { label: '通义千问 · 北京', provider: 'qwen', base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1', model: 'qwen-plus' },
  singapore: { label: '通义千问 · 新加坡', provider: 'qwen', base_url: 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1', model: 'qwen-plus' },
  us: { label: '通义千问 · 美国', provider: 'qwen', base_url: 'https://dashscope-us.aliyuncs.com/compatible-mode/v1', model: 'qwen-plus' },
} as const
type Preset = keyof typeof presets

export function ConnectionSettings({ close, applied }: { close: () => void; applied: () => void }) {
  const [token, setToken] = useState(getToken)
  const [draft, setDraft] = useState<ModelConnection>(() => getModelConnection() ?? { ...presets.deepseek, api_key: '' })
  const [result, setResult] = useState('')
  const [testing, setTesting] = useState(false)
  const controller = useRef<AbortController | null>(null)
  useEffect(() => () => controller.current?.abort(), [])
  function update(next: ModelConnection) {
    controller.current?.abort(); controller.current = null; setTesting(false); setResult(''); setDraft(next)
  }
  async function test() {
    controller.current?.abort()
    const active = new AbortController(); controller.current = active
    setTesting(true); setResult('')
    try {
      validateModelConnection(draft)
      const answer = await testModelConnection(draft, token, active.signal)
      if (controller.current === active) setResult(`连接成功 · ${answer.model} · ${Math.round(answer.latency_ms)} ms`)
    } catch (error) {
      if (controller.current === active && !active.signal.aborted) setResult((error as Error).message)
    } finally { if (controller.current === active) setTesting(false) }
  }
  function apply() {
    try { validateModelConnection(draft) } catch (error) { setResult((error as Error).message); return }
    saveToken(token); saveModelConnection(draft.api_key.trim() ? draft : undefined); applied(); close()
  }
  return <div className="modal-backdrop" onClick={close}><div className="settings-modal connection-settings" role="dialog" aria-modal="true" aria-labelledby="settings-title" onClick={e => e.stopPropagation()}>
    <span className="eyebrow">MODEL CONNECTION</span><h2 id="settings-title">配置 API Key</h2>
    <p>填写自己的模型密钥即可体验 ReAct 问答。API Key 仅留在当前页面内存中，刷新后需重新填写。</p>
    <label>服务商与地域<select onChange={e => { const preset = presets[e.target.value as Preset]; update({ ...preset, api_key: '' }) }} defaultValue={Object.entries(presets).find(([, p]) => p.base_url === draft.base_url)?.[0] ?? 'deepseek'}>{Object.entries(presets).map(([key, p]) => <option key={key} value={key}>{p.label}</option>)}</select></label>
    <label>模型名称<input autoComplete="off" value={draft.model} onChange={e => update({ ...draft, model: e.target.value })} /></label>
    <label>Base URL<input type="url" autoComplete="off" value={draft.base_url} onChange={e => update({ ...draft, base_url: e.target.value })} /></label>
    <label>API Key<input type="password" autoComplete="off" spellCheck={false} value={draft.api_key} placeholder="填写服务商提供的 API Key" onChange={e => update({ ...draft, api_key: e.target.value })} /></label>
    <small>千问 Key、模型和 Base URL 需属于同一地域。测试连接会发送一次简短请求，由你的账号计费。</small>
    <details className="access-settings"><summary>网站访问令牌</summary><p>部署者设置了访问令牌时填写。该令牌与模型 API Key 不同，仅保存在当前浏览器会话中。</p><label>网站访问令牌<input type="password" autoComplete="off" value={token} onChange={e => { controller.current?.abort(); controller.current = null; setTesting(false); setResult(''); setToken(e.target.value) }} /></label></details>
    {result && <div className="connection-result" role="status">{result}</div>}
    <div className="modal-actions"><button onClick={close}>取消</button><button disabled={testing || !draft.api_key.trim()} onClick={() => void test()}>{testing ? '测试中…' : '测试连接'}</button><button className="primary-button" disabled={testing || !draft.api_key.trim()} onClick={apply}>应用设置</button></div>
    <button className="offline-connection" onClick={() => { saveToken(token); saveModelConnection(undefined); applied(); close() }}>使用服务器配置 / 离线资料摘录</button>
  </div></div>
}
