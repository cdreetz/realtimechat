// Browser execution and autonomous worker controls shared by humans and tools.
function configureBrowserExecution(browser) {
  const bar = document.createElement('div'); bar.className = 'workspace-toolbar';
  const status = document.createElement('span'); status.textContent = 'Browser Use ready';
  const panel = document.createElement('div'); panel.hidden = true;
  const code = document.createElement('textarea'); code.setAttribute('aria-label', 'Browser Python code');
  code.placeholder = 'print(page_info())'; code.style.cssText = 'width:100%;height:85px;font:12px monospace;resize:vertical';
  const output = document.createElement('pre'); output.style.cssText = 'margin:3px;max-height:100px;overflow:auto;white-space:pre-wrap;user-select:text';
  const run = actionButton('Run Python', async () => {
    const task = await submitBrowserCode(browser, code.value, 'human');
    output.textContent = `Started ${task.task_id}`;
  });
  panel.append(code, run, output);
  let inputQueue = Promise.resolve();
  const control = actionButton('Take control', async () => {
    const mode = browser.controlMode === 'human' ? 'agent' : 'human';
    if (mode === 'agent') await inputQueue;
    await api('POST', `/api/browser/${browser.browserId}/control`, {mode});
    browser.controlMode = mode;
    control.textContent = mode === 'human' ? 'Resume agent' : 'Take control';
    browser.shot.focus();
    training.event('browser_control', {browser_id: browser.browserId, mode});
  });
  bar.append(status, control, actionButton('Python', () => { panel.hidden = !panel.hidden; }),
    actionButton('Stop script', () => api('POST', `/api/browser/${browser.browserId}/execution/stop`)));
  browser.win.body.insertBefore(bar, browser.viewport);
  browser.win.body.insertBefore(panel, browser.viewport);
  browser.shot.tabIndex = 0;
  browser.shot.setAttribute('aria-label','Browser page. Choose Take control to click, type, and scroll.');
  let closed = false, timer, busy = false;
  const input = data => {
    if (browser.controlMode !== 'human') { status.textContent = 'Choose Take control to interact'; return; }
    inputQueue = inputQueue.catch(() => {}).then(async () => {
      try { await api('POST', `/api/browser/${browser.browserId}/input`, data); refreshBrowserScreenshot(browser); }
      catch (e) { status.textContent = e.message; }
    });
    return inputQueue;
  };
  browser.shot.addEventListener('click', e => {
    const r = browser.shot.getBoundingClientRect();
    input({kind:'click', x:(e.clientX-r.left)*browser.shot.naturalWidth/r.width, y:(e.clientY-r.top)*browser.shot.naturalHeight/r.height});
  });
  browser.shot.addEventListener('keydown', e => {
    if (browser.controlMode !== 'human') return;
    if (e.key.length === 1 && !e.metaKey && !e.ctrlKey) { e.preventDefault(); input({kind:'text',text:e.key}); }
    else if (['Enter','Tab','Backspace','Delete','Escape','ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)) { e.preventDefault(); input({kind:'key',key:e.key}); }
  });
  browser.shot.addEventListener('paste', e => { if (browser.controlMode==='human') { e.preventDefault(); input({kind:'text',text:e.clipboardData.getData('text')}); } });
  browser.shot.addEventListener('wheel', e => {
    if (browser.controlMode !== 'human') return;
    e.preventDefault(); const r = browser.shot.getBoundingClientRect();
    input({kind:'scroll', x:(e.clientX-r.left)*browser.shot.naturalWidth/r.width,y:(e.clientY-r.top)*browser.shot.naturalHeight/r.height,delta_y:e.deltaY});
  },{passive:false});
  const poll = async () => {
    if (closed) return;
    try {
      const s = await api('GET', `/api/browser/${browser.browserId}/execution`);
      if (closed) return;
      browser.controlMode = s.mode; browser.generation = s.generation;
      control.textContent = s.mode==='human' ? 'Resume agent' : 'Take control';
      const task = s.jobs.at(-1); busy = task?.status === 'running';
      status.textContent = `${s.mode==='human' ? 'You have control' : 'Agent control'} · ${task ? task.status : 'ready'}`;
      if (task) output.textContent = `${task.task_id} · ${task.status}\n${task.output || ''}${task.error || ''}`;
      if (!browser.win.minimized) await refreshBrowserView(browser);
    } catch (e) { status.textContent = e.message; }
    if (!closed) timer = setTimeout(poll, busy ? 800 : 2500);
  };
  browser.executionCleanup = () => { closed = true; clearTimeout(timer); };
  poll();
}

async function submitBrowserCode(browser, code, owner='agent', timeout=30) {
  const result = await api('POST', `/api/browser/${browser.browserId}/exec`, {code,owner,timeout,job_id:crypto.randomUUID()});
  training.event('browser_exec_submitted', {browser_id:browser.browserId,code,...result});
  return result;
}

function openAgents() {
  const existing = Object.values(windows).find(w => w.kind==='agents');
  if (existing) { existing.win.restore(); return [existing.id,'Background agents opened']; }
  const id=nextWindowId++; let closed=false,timer,selected=null,last=[];
  const win=makeWindow({title:'Background Agents',icon:'🤖',x:65,y:45,w:650,h:520,
    onClose:()=>{closed=true;clearTimeout(timer);delete windows[id];}});
  windows[id]={id,kind:'agents',title:'Background Agents',win};
  const objective=document.createElement('textarea'); objective.setAttribute('aria-label','Background task objective');
  objective.placeholder='Describe work to do in the background';objective.style.cssText='width:100%;height:65px;resize:vertical';
  const workspace=document.createElement('select');workspace.setAttribute('aria-label','Agent workspace');
  const defaultOption=document.createElement('option');defaultOption.value='';defaultOption.textContent='New workspace for this task (recommended)';workspace.append(defaultOption);
  api('GET','/api/workspaces').then(s=>{for(const w of s.workspaces){const o=document.createElement('option');o.value=w.workspace_id;o.textContent=w.name;workspace.append(o);}});
  const bar=document.createElement('div');bar.className='workspace-toolbar';
  bar.append(workspace,actionButton('Start agent',async()=>{
    const task=await api('POST','/api/agents',{objective:objective.value,name:objective.value.slice(0,60)||'Background task',workspace_id:workspace.value||null});
    watchedWorkers.add(task.task_id);selected=task.task_id;objective.value='';await refresh();
  }));
  const list=document.createElement('div');list.style.cssText='max-height:110px;overflow:auto';
  const detail=document.createElement('pre');detail.style.cssText='flex:1;min-height:0;margin:3px;overflow:auto;white-space:pre-wrap;user-select:text';
  const steering=document.createElement('input');steering.setAttribute('aria-label','Steer selected agent');steering.placeholder='Reply or change direction';steering.style.flex='1';
  const controls=document.createElement('div');controls.className='workspace-toolbar';
  const act=async(action,data)=>{if(!selected)throw new Error('Select a task');await api('POST',`/api/agents/${selected}/${action}`,data);await refresh();};
  controls.append(steering,actionButton('Send',async()=>{await act('steer',{message:steering.value});steering.value='';}),
    actionButton('Pause',()=>act('pause')),actionButton('Resume',()=>act('resume')),actionButton('Stop',()=>act('stop')),
    actionButton('Workspace',async()=>{const j=last.find(j=>j.task_id===selected);if(j?.workspace_id)await workspaceEditor(j.workspace_id);}),
    actionButton('Terminal',async()=>{const j=last.find(j=>j.task_id===selected);if(j?.terminal_id)await workspaceTerminal(j.workspace_id,j.terminal_id);}),
    actionButton('Browser',async()=>{const j=last.find(j=>j.task_id===selected);if(j?.browser_id)ensureWorkerBrowser(j.browser_id);}));
  win.body.append(objective,bar,list,detail,controls);
  let refreshing=false;
  async function refresh(){
    if(closed||refreshing)return;refreshing=true;
    try{
      const s=await api('GET','/api/agents');last=s.tasks;
      if(!selected&&last.length)selected=last.at(-1).task_id;
      list.replaceChildren(...[...last].reverse().map(j=>actionButton(`${j.status} · ${j.name}`,async()=>{selected=j.task_id;await refresh();})));
      if(selected){const j=await api('GET',`/api/agents/${selected}`);detail.textContent=[j.objective,`Status: ${j.status} · round ${j.round||0}`,j.question?`Needs input: ${j.question}`:'',j.result||j.error||j.progress||'',...(j.events||[]).slice(-12).map(e=>`${e.kind}: ${e.text||e.name||e.error||e.question||''}`)].filter(Boolean).join('\n\n');}
    }catch(e){detail.textContent=e.message;}finally{refreshing=false;}
  }
  async function tick(){await refresh();if(!closed)timer=setTimeout(tick,1500);}tick();
  return [id,'Background agents opened. Work continues when this window closes.'];
}

function ensureWorkerBrowser(browserId) {
  const existing=Object.values(windows).find(w=>w.browserId===browserId);
  if(existing){existing.win.restore();return existing;}
  return openBrowserWindow(nextWindowId++,browserId);
}

const advancedTools = {
  async browser_exec({window_id,code,timeout=30}) {
    const [b,err]=requireWindow(window_id,'browser');if(err)return err;
    return JSON.stringify(await submitBrowserCode(b,code,'agent',timeout));
  },
  async read_browser_exec({window_id,task_id}) {
    const [b,err]=requireWindow(window_id,'browser');if(err)return err;
    const s=await api('GET',`/api/browser/${b.browserId}/execution`);
    return JSON.stringify(task_id?{generation:s.generation,mode:s.mode,task:s.jobs.find(j=>j.task_id===task_id)||{status:'unknown',error:'Execution no longer retained; do not replay blindly'}}:s);
  },
  async stop_browser_exec({window_id}) {const [b,err]=requireWindow(window_id,'browser');return err||JSON.stringify(await api('POST',`/api/browser/${b.browserId}/execution/stop`));},
  async spawn_agent({objective,name,workspace_id}) {
    const job=await api('POST','/api/agents',{objective,name:name||objective.slice(0,60),workspace_id:workspace_id||null});
    watchedWorkers.add(job.task_id);openAgents();return JSON.stringify(job);
  },
  async list_agents(){return JSON.stringify(await api('GET','/api/agents'));},
  async read_agent({task_id}){return JSON.stringify(await api('GET',`/api/agents/${task_id}`));},
  async steer_agent({task_id,message}){return JSON.stringify(await api('POST',`/api/agents/${task_id}/steer`,{message}));},
  async stop_agent({task_id}){return JSON.stringify(await api('POST',`/api/agents/${task_id}/stop`));},
  async resume_agent({task_id}){return JSON.stringify(await api('POST',`/api/agents/${task_id}/resume`));},
};
function advancedToolSchemas(){
  const string={type:'string'},integer={type:'integer'};
  const spec=(name,description,properties,required=Object.keys(properties))=>({type:'function',function:{name,description,parameters:{type:'object',properties,required}}});
  return [
    spec('browser_exec','Primary browser tool: execute Python through Browser Use with persistent variables. Preloaded synchronous helpers: new_tab(url), goto_url(url), wait_for_load(), page_info(), js(expression) returning JSON values (extract DOM text inside JS, not Python), fill_input(selector,text), press_key(key), click_at_xy(x,y), cdp("Domain.method", **params), list_tabs(), switch_tab(target). Use print for output. Returns task_id immediately; call read_browser_exec for results. Use a background agent for a multi-step investigation. One script at a time; human takeover blocks automation.',{window_id:integer,code:string,timeout:integer},['window_id','code']),
    spec('read_browser_exec','Read browser Python execution results and generation. Errors/cancellation may leave already-completed page actions; inspect before retrying. Variables reset on controller restart or cancellation.',{window_id:integer,task_id:string},['window_id']),
    spec('stop_browser_exec','Stop the browser script and reset its Python variables. Does not undo completed page actions.',{window_id:integer}),
    spec('spawn_agent','Start an autonomous background model worker; returns immediately. It plans, runs tools, checks results and continues independently while you keep talking. Defaults to a new isolated workspace and its own browser. Optional workspace_id explicitly grants use of an existing workspace; only one active worker per workspace. Max two simultaneous workers, others queue.',{objective:string,name:string,workspace_id:string},['objective']),
    spec('list_agents','List autonomous background workers, status, progress, workspace and browser IDs.',{}),
    spec('read_agent','Read one worker result, events, question, and current progress.',{task_id:string}),
    spec('steer_agent','Send a correction or answer to a worker. Applied at the next tool-round boundary; resumes a paused worker.',{task_id:string,message:string}),
    spec('stop_agent','Cancel a background worker and stop its owned foreground command/browser script. Completed changes remain.',{task_id:string}),
    spec('resume_agent','Resume a paused or interrupted worker from inspected existing state.',{task_id:string}),
  ];
}

const watchedWorkers=new Set();
const attachedWorkerViews=new Set();
const workerNotifications=new Set(JSON.parse(sessionStorage.getItem('retrovoice-worker-notifications')||'[]'));
let workerPollBusy=false;
setInterval(async()=>{
  if(!desktopStateReady||workerPollBusy)return;
  workerPollBusy=true;
  try{
    const {tasks}=await api('GET','/api/agents');
    for(const j of tasks){
      if(watchedWorkers.has(j.task_id)){
        if(j.workspace_id && j.terminal_id && !attachedWorkerViews.has(j.task_id+':workspace')){
          attachedWorkerViews.add(j.task_id+':workspace');
          await workspaceEditor(j.workspace_id);await workspaceTerminal(j.workspace_id,j.terminal_id);
        }
        if(j.browser_id && !attachedWorkerViews.has(j.task_id+':browser')){
          attachedWorkerViews.add(j.task_id+':browser');ensureWorkerBrowser(j.browser_id);
        }
      }
      // Completion is displayed once in this desktop, without interrupting speech.
      const key=`${j.task_id}:${j.status}:${j.finished_at||j.question||''}`;
      if(['completed','failed','cancelled','interrupted','paused'].includes(j.status)&&!workerNotifications.has(key)){
        workerNotifications.add(key);
        sessionStorage.setItem("retrovoice-worker-notifications",JSON.stringify([...workerNotifications].slice(-1000)));
        feed('sys',`* Agent ${j.name}: ${j.status}. ${(j.question||j.result||j.error||'').slice(0,500)}`);
        training.event('background_agent_status',{task_id:j.task_id,status:j.status,result:j.result,error:j.error});
      }
    }
  }catch{}finally{workerPollBusy=false;}
},2000);
