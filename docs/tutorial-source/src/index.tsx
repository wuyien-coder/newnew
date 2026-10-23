import React from 'react';
import {registerRoot,Composition,AbsoluteFill,useCurrentFrame,useVideoConfig,interpolate,staticFile} from 'remotion';
const slides=[
 {tag:'开始之前',title:'MathKnow 使用教程',sub:'数学人物知识问答 · 从提问到证据核验',lines:['01  启动与连接','02  提问、追问与引用','03  关系查询与资料管理'],note:'基于 GitHub 当前版本制作 · 中文无声动画教程 · 界面为操作示意'},
 {tag:'01 / 本地启动',title:'先启动服务，再打开网页',sub:'Windows · Python 3.11+ · Node.js 22+',code:['python -m venv .venv','.\\.venv\\Scripts\\python.exe -m pip install -e ".[dev,prod]"','Copy-Item .env.example .env','cd frontend','npm ci','npm run build','cd ..','.\\.venv\\Scripts\\python.exe scripts/run_api.py'],note:'首次运行会导入语料并建立索引，请等待“应用就绪”。'},
 {tag:'02 / 访问入口',title:'打开知识问答工作台',sub:'本地轻量模式入口',big:'http://127.0.0.1:8010/web/',lines:['确认页面右上角显示“服务已连接”','默认“离线演示”：资料摘录，不调用付费模型','Docker 完整版入口使用 8080 端口'],note:'GitHub 仓库用于获取代码；启动后的服务地址用于问答。'},
 {tag:'03 / 连接设置',title:'访问令牌填在哪里？',sub:'左下角「连接设置」→「保存并连接」',panel:'settings',lines:['本地默认未启用令牌，可以留空','启用认证时，填写应用访问令牌','此处不填写模型 API 密钥'],note:'模型 API 密钥只配置在服务端 .env / .env.full 中。'},
 {tag:'04 / 第一次提问',title:'用完整问题描述你的需求',sub:'点击示例问题，或在底部输入后按 Enter',panel:'ask',lines:['华罗庚在数论方面有哪些贡献？','苏步青师从谁？','研究微分几何的有哪些人？'],note:'Enter 发送 · Shift + Enter 换行 · 生成中可点击“停止生成”。'},
 {tag:'05 / 核对引用',title:'回答之后，回到参考依据',sub:'点击回答下方「条参考依据 ↗」，查看右侧证据',panel:'evidence',lines:['核对人物、原文摘录和来源','区分“原文”与“图谱”证据','参考强度不等于事实正确率'],note:'资料中没有记载的内容，请继续查证；模型回答需结合原文核实。'},
 {tag:'06 / 人物关系',title:'把问题问到关系上',sub:'确认输入区的「图谱增强」已开启',panel:'graph',lines:['华罗庚与熊庆来有什么关系？','苏步青师从谁？','请列出支持这一关系的依据。'],note:'通过问答查询关系；真实 ReAct 模式可在右侧查看实际工具调用。'},
 {tag:'07 / 多轮追问',title:'在同一会话中继续探索',sub:'左侧「历史会话」可以回看已有对话',panel:'history',lines:['追问尽量补充人物和具体细节','打开历史会话回看问答与依据','切换研究主题时点击“新建对话”'],note:'离线摘录能力有限；接入真实模型后体验模型驱动的多轮问答。'},
 {tag:'08 / 补充资料',title:'让新文档进入知识库',sub:'左侧「知识库」→「上传文档」',panel:'upload',lines:['支持 TXT、Markdown、文字 PDF','单文件最多 20MB；扫描件先转文本','等待摄取“已完成”再提问'],note:'摄取失败可重试；删除文档会将其移出知识库，请核对后再操作。'},
 {tag:'09 / 真实模型模式',title:'接入模型，体验 ReAct 工具循环',sub:'在本机 .env 配置模型，然后重启服务',code:['MATHKB_LLM__PROVIDER=openai','MATHKB_LLM__BASE_URL=<兼容 API 地址>','MATHKB_LLM__MODEL=<模型名>','MATHKB_LLM__API_KEY=<仅在本机填写>'],lines:['右上角核对模型；右侧查看实际工具调用和引用','完整 BGE / pgvector / Celery 部署请按 README 执行'],note:'真实模型可能产生服务商费用；不要把 .env 或密钥上传到 GitHub。'},
 {tag:'10 / 常见问题',title:'遇到问题，先检查这三项',sub:'服务连接 → 访问认证 → 运行模式',lines:['网页打不开：确认服务运行、端口与 /web/ 路径','身份验证失败：核对连接设置中的应用访问令牌','只有资料摘录：检查是否仍为 stub 离线模式'],note:'详细命令、章节导航和常见问题见 docs/使用教程.md。'},
 {tag:'现在开始',title:'问一个问题，核对一份证据',sub:'MathKnow · 数学人物知识分析',big:'提问 → 阅读 → 核对 → 追问',lines:['代码与完整部署说明','github.com/wuyien-coder/newnew'],note:'本片展示操作流程，不展示虚构的模型回答或实测性能。'},
];
function Panel({kind,frame}:{kind:string;frame:number}){
const p=interpolate(frame,[0,80],[0,1],{extrapolateRight:'clamp'});
return <div style={{background:'#fff',border:'1px solid #dce2dc',borderRadius:20,height:370,display:'flex',overflow:'hidden',boxShadow:'0 15px 40px #152e2210'}}>
<div style={{width:235,background:'#edf1ed',padding:22,fontSize:17,lineHeight:2.2}}><b style={{fontSize:23}}>M· MathKnow</b><div>＋ 新建对话</div><div style={{color:kind==='upload'?'#aaa':'#197c67'}}>知识问答</div><div style={{color:kind==='upload'?'#197c67':'#777'}}>知识库</div><div>技术架构</div><div style={{marginTop:20}}>历史会话</div><div style={{fontSize:16,color:'#777'}}>连接设置 ↗</div></div>
<div style={{flex:1,padding:28}}><div style={{display:'flex',justifyContent:'space-between',fontSize:16,color:'#7a8880'}}><span>工作空间 / 知识问答</span><span>操作示意</span></div>
{kind==='settings'?<div style={{margin:'24px auto',width:600,padding:22,background:'#f4f7f4',borderRadius:14}}><b style={{fontSize:26}}>连接设置</b><p style={{fontSize:19}}>访问令牌</p><div style={{background:'white',padding:14,color:'#89948d',borderRadius:8}}>本地离线演示可留空</div><div style={{marginTop:20,color:'#197c67',textAlign:'right'}}>保存并连接 ↗</div></div>:
kind==='upload'?<><h2 style={{fontSize:30}}>管理你的知识来源</h2><div style={{border:'2px dashed #84ac9c',padding:22,borderRadius:14}}>上传文档 ＋<p style={{fontSize:20,color:'#73867b'}}>TXT / Markdown / PDF · 最大 20MB</p></div><p style={{fontSize:20}}>摄取任务：等待处理 → 处理中 → 已完成</p></>:
kind==='evidence'?<><h2 style={{fontSize:30}}>参考依据</h2><div style={{display:'flex',gap:18}}>{['原文摘录','人物关系'].map((x,i)=><div key={x} style={{flex:1,background:'#f3f7f3',padding:22,borderRadius:12}}><b style={{color:'#197c67'}}>0{i+1} / {x}</b><p style={{fontSize:20,color:'#68796e'}}>核对人物、文本与来源</p><div style={{height:7,background:'#dce5dd',marginTop:18}}/><div style={{height:7,width:'70%',background:'#dce5dd',marginTop:12}}/></div>)}</div><p style={{fontSize:19}}>点击某条回答，即可切换其参考依据。</p></>:
kind==='history'?<><h2 style={{fontSize:30}}>继续当前对话</h2><div style={{background:'#edf4ef',padding:22,borderRadius:12,fontSize:24}}>请补充原文依据，并说明资料来源。</div><p style={{fontSize:20,color:'#73867b'}}>历史会话保留已保存的对话与引用。</p><div style={{color:'#197c67',marginTop:30}}>＋ 新建对话：开始新的研究主题</div></>:
<><h2 style={{fontSize:30}}>从一个问题，发现更多知识</h2><div style={{padding:22,background:'#f4f7f4',borderRadius:14,fontSize:26,minHeight:72}}>{(kind==='graph'?'华罗庚与熊庆来有什么关系？':'华罗庚在数论方面有哪些贡献？').slice(0,Math.floor(p*25))}</div><div style={{display:'flex',justifyContent:'space-between',marginTop:26,color:'#197c67',fontSize:20}}><span>✓ 图谱增强</span><span>Enter 发送 ↑</span></div><p style={{fontSize:18,color:'#87958b',marginTop:25}}>回答结果以当前运行实例实际返回内容为准。</p></>}
</div></div>;
}
function Tutorial(){const f=useCurrentFrame();const {fps,durationInFrames}=useVideoConfig();const idx=Math.min(slides.length-1,Math.floor(f/(10*fps)));const s=slides[idx];const local=f%(10*fps);
return <AbsoluteFill style={{background:'#f5f4ee',fontFamily:'NotoSC, sans-serif',color:'#173b31'}}>
<style>{`@font-face{font-family:NotoSC;src:url('${staticFile('NotoSansSC.ttf')}')}*{box-sizing:border-box}`}</style>
<div style={{position:'absolute',left:65,top:40,fontSize:22,fontWeight:600}}>M· MathKnow <span style={{fontWeight:400,color:'#859387',fontSize:18,marginLeft:18}}>使用教程</span></div>
<div style={{position:'absolute',right:65,top:43,fontSize:17,color:'#75887b'}}>中文步骤说明 / 无声动画</div>
<div style={{position:'absolute',left:65,top:106,color:'#24836a',fontSize:18,letterSpacing:3}}>{s.tag}</div>
<div style={{position:'absolute',left:65,right:65,top:148,opacity:interpolate(local,[0,12],[0,1],{extrapolateRight:'clamp'}),translate:`0 ${interpolate(local,[0,20],[15,0],{extrapolateRight:'clamp'})}px`}}>
<h1 style={{fontSize:53,lineHeight:1.3,margin:'0 0 12px',fontWeight:650}}>{s.title}</h1><p style={{fontSize:24,color:'#75847a',margin:'0 0 26px'}}>{s.sub}</p>
{'panel' in s?<><Panel kind={s.panel!} frame={local}/><div style={{display:'flex',gap:25,marginTop:20}}>{s.lines?.map((x,i)=><div key={x} style={{flex:1,fontSize:21,lineHeight:1.6,color:'#456456'}}><span style={{color:'#21846b',marginRight:10}}>0{i+1}</span>{x}</div>)}</div></>:
<>{'code' in s&&<div style={{background:'#183c33',borderRadius:18,padding:'20px 30px',color:'#e4f1e8',fontFamily:'NotoSC, monospace',fontSize:idx===1?23:26,lineHeight:1.6}}>{s.code?.map(x=><div key={x}>{x}</div>)}</div>}{'big' in s&&<div style={{padding:'38px 32px',margin:'25px 0',background:'#e5eee5',borderRadius:18,fontSize:38,fontWeight:600,color:'#19745c'}}>{s.big}</div>}<div style={{marginTop:24}}>{s.lines?.map((x,i)=><div key={x} style={{fontSize:idx===9?24:30,lineHeight:1.85,opacity:interpolate(local,[12+i*12,24+i*12],[0,1],{extrapolateLeft:'clamp',extrapolateRight:'clamp'})}}><span style={{color:'#1f8a6d',marginRight:20}}>↗</span>{x}</div>)}</div></>}
</div><div style={{position:'absolute',left:65,right:65,bottom:48,display:'flex',justifyContent:'space-between',gap:25,fontSize:18,color:'#657a6b',borderTop:'1px solid #d5dfd4',paddingTop:18}}><span>{s.note}</span><span style={{whiteSpace:'nowrap'}}>{String(idx+1).padStart(2,'0')} / 12</span></div><div style={{position:'absolute',bottom:0,height:5,left:0,width:`${f/durationInFrames*100}%`,background:'#1c8c6e'}}/></AbsoluteFill>}
registerRoot(()=> <Composition id="MathKnowTutorial" component={Tutorial} durationInFrames={2880} fps={24} width={1600} height={900}/>);
