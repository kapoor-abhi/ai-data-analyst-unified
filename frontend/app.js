// frontend/app.js
const API_BASE_URL = 'http://localhost:8000';

// 1. SESSION STORAGE: Remember thread ID and logs across page reloads
let currentThreadId = sessionStorage.getItem('ai_thread_id');
if (!currentThreadId) {
    currentThreadId = crypto.randomUUID();
    sessionStorage.setItem('ai_thread_id', currentThreadId);
}

const AppState = {
    threadId: currentThreadId,
    mode: 'upload',
    auditLog: JSON.parse(sessionStorage.getItem('ai_audit_log') || '[]'),
    hasShownCleaningPlan: false // Tracks if we've shown the JSON plan for this cycle
};

// DOM Elements
const chatHistory = document.getElementById('chat-history');
const uploadForm = document.getElementById('upload-form');
const chatForm = document.getElementById('chat-form');
const fileInput = document.getElementById('file-input');
const uploadInstruction = document.getElementById('upload-instruction');
const chatInput = document.getElementById('chat-input');
const loadingIndicator = document.getElementById('loading-indicator');
const pipelineStatus = document.getElementById('pipeline-status');
const tabStats = document.getElementById('tab-stats');
const tabCharts = document.getElementById('tab-charts');
const statsContainer = document.getElementById('stats-container');
const chartContainer = document.getElementById('chart-container');
const workspaceEmpty = document.getElementById('workspace-empty');
const downloadBtn = document.getElementById('download-btn');
const statsOverview = document.getElementById('stats-overview');
const statsColumns = document.getElementById('stats-columns');
const previewThead = document.getElementById('preview-thead');
const previewTbody = document.getElementById('preview-tbody');
const previewContainer = document.getElementById('data-preview-container');

// NEW DOM Elements for EDA Feature
const tabEDA = document.getElementById('tab-eda');
const edaContainer = document.getElementById('eda-container');
const edaFrame = document.getElementById('eda-frame');

marked.setOptions({ breaks: true, gfm: true });

function addMessage(sender, text, isMarkdown = false) {
    const msgDiv = document.createElement('div');
    if (sender === 'user') {
        msgDiv.className = 'message user-msg bg-slate-900 text-white p-3 rounded-lg text-sm self-end ml-auto max-w-[85%] shadow-sm';
        msgDiv.textContent = text;
    } else if (sender === 'error') {
        msgDiv.className = 'message error-msg bg-red-50 text-red-700 border border-red-200 p-3 rounded-lg text-sm max-w-[90%]';
        msgDiv.textContent = `Error: ${text}`;
    } else {
        msgDiv.className = 'message ai-msg bg-white border border-gray-200 p-3 rounded-lg text-sm text-gray-800 shadow-sm max-w-[90%]';
        msgDiv.innerHTML = isMarkdown ? marked.parse(text) : text;
    }
    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function setLoading(isLoading, statusText = 'Processing...') {
    if (isLoading) {
        loadingIndicator.classList.remove('hidden');
        uploadForm.querySelector('button').disabled = true;
        if(chatForm) chatForm.querySelector('button').disabled = true;
    } else {
        loadingIndicator.classList.add('hidden');
        uploadForm.querySelector('button').disabled = false;
        if(chatForm) chatForm.querySelector('button').disabled = false;
    }
    pipelineStatus.textContent = `Status: ${isLoading ? statusText : 'Idle'}`;
}

// 2. LIVE MULTI-FILE PREVIEW LOGIC
// Instantly display all selected files in the workspace before ingestion starts
fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        let fileListHtml = `
            <div class="bg-white p-6 rounded-xl shadow-sm border border-gray-200 w-full max-w-2xl mx-auto mt-10">
                <h3 class="font-bold text-slate-700 mb-4 uppercase text-xs tracking-wider">Staged Files for Ingestion (${fileInput.files.length})</h3>
                <ul class="space-y-2">`;
                
        for (let i = 0; i < fileInput.files.length; i++) {
            const sizeKB = (fileInput.files[i].size / 1024).toFixed(1);
            fileListHtml += `
                <li class="flex justify-between items-center text-sm font-mono bg-slate-50 p-3 rounded border border-slate-100">
                    <span class="text-blue-700 font-bold">📄 ${fileInput.files[i].name}</span>
                    <span class="text-slate-400 text-xs">${sizeKB} KB</span>
                </li>`;
        }
        
        fileListHtml += `</ul>
            <p class="text-[10px] text-slate-400 mt-4 text-center italic">Ready for ingestion pipeline...</p>
            </div>`;
        
        workspaceEmpty.innerHTML = fileListHtml;
    }
});

async function updateAuditAndStats(stepName) {
    try {
        const response = await fetch(`${API_BASE_URL}/statistics`);
        const stats = await response.json();
        if (response.ok) {
            AppState.auditLog.push({
                step: stepName,
                rows: stats.total_rows,
                cols: stats.total_columns,
                timestamp: new Date().toLocaleTimeString()
            });
            
            // Save audit log to session storage so it survives refresh
            sessionStorage.setItem('ai_audit_log', JSON.stringify(AppState.auditLog));
            
            renderAuditTrail();
            renderDataPreview(stats.sample_data);
            renderStatsCards(stats);
        }
    } catch (e) { console.error("Stats Update Failed:", e); }
}

function renderAuditTrail() {
    const container = document.getElementById('audit-trail-container');
    container.innerHTML = `
        <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-4">
            <h3 class="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-3">Pipeline Audit History</h3>
            <table class="w-full text-[11px] text-left">
                <thead>
                    <tr class="text-slate-400 border-b">
                        <th class="pb-2">Execution Phase</th>
                        <th class="pb-2 text-center">Rows</th>
                        <th class="pb-2 text-center">Cols</th>
                        <th class="pb-2 text-right">Timestamp</th>
                    </tr>
                </thead>
                <tbody class="divide-y">
                    ${AppState.auditLog.map(log => `
                        <tr class="hover:bg-blue-50/50 transition">
                            <td class="py-2 font-bold text-blue-600">${log.step}</td>
                            <td class="py-2 text-center font-mono">${log.rows.toLocaleString()}</td>
                            <td class="py-2 text-center font-mono">${log.cols}</td>
                            <td class="py-2 text-right text-slate-400 font-mono">${log.timestamp}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}

function renderDataPreview(sampleData) {
    if (!sampleData || sampleData.length === 0) return;
    previewContainer.classList.remove('hidden');
    const cols = Object.keys(sampleData[0]);
    previewThead.innerHTML = `<tr>${cols.map(c => `<th class="p-2 border-r whitespace-nowrap">${c}</th>`).join('')}</tr>`;
    previewTbody.innerHTML = sampleData.map(row => `
        <tr class="hover:bg-gray-50 transition">
            ${cols.map(c => `
                <td class="p-2 border-r truncate max-w-[180px] ${row[c] === null ? 'text-red-400 italic' : ''}">
                    ${row[c] !== null ? row[c] : 'null'}
                </td>
            `).join('')}
        </tr>
    `).join('');
}

function renderStatsCards(stats) {
    statsOverview.innerHTML = `
        <div class="bg-white p-4 rounded-xl border border-gray-200 shadow-sm">
            <span class="text-[10px] text-slate-400 uppercase font-black tracking-widest">Total Observations</span>
            <div class="text-2xl font-bold text-slate-800">${stats.total_rows.toLocaleString()}</div>
        </div>
        <div class="bg-white p-4 rounded-xl border border-gray-200 shadow-sm">
            <span class="text-[10px] text-slate-400 uppercase font-black tracking-widest">Feature Dimensions</span>
            <div class="text-2xl font-bold text-slate-800">${stats.total_columns}</div>
        </div>
    `;

    statsColumns.innerHTML = '';
    for (const [name, data] of Object.entries(stats.columns)) {
        const card = document.createElement('div');
        card.className = 'bg-slate-50 border border-slate-200 rounded-lg p-3 hover:bg-white transition shadow-sm';
        let detailHtml = `<div class="flex justify-between items-start mb-2"><h4 class="text-xs font-black text-slate-700 truncate w-2/3">${name}</h4><span class="text-[9px] bg-slate-200 px-1.5 py-0.5 rounded font-mono">${data.dtype}</span></div>`;
        detailHtml += `<div class="flex gap-3 text-[10px] text-slate-500 mb-2 font-mono"><div>NULL: <span class="${data.missing_values > 0 ? 'text-red-500 font-bold' : ''}">${data.missing_values}</span></div><div>UNIQ: ${data.unique_values}</div></div>`;
        
        if (data.mean !== "") {
            detailHtml += `<div class="grid grid-cols-3 gap-1 text-[9px] border-t pt-2 mt-1 font-mono">
                <div>AVG: ${Number(data.mean).toFixed(1)}</div><div>MIN: ${Number(data.min).toFixed(1)}</div><div>MAX: ${Number(data.max).toFixed(1)}</div>
            </div>`;
        } else if (data.top) {
            detailHtml += `<div class="text-[9px] border-t pt-2 mt-1 font-mono text-slate-500 italic truncate">TOP: ${data.top} (${data.freq}x)</div>`;
        }
        card.innerHTML = detailHtml;
        statsColumns.appendChild(card);
    }
}

function setMode(newMode) {
    AppState.mode = newMode;
    if (newMode === 'upload') {
        uploadForm.classList.remove('hidden');
        chatForm.classList.add('hidden');
        if(tabEDA) tabEDA.classList.add('hidden'); // Keep EDA locked during upload/processing
    } else {
        uploadForm.classList.add('hidden');
        chatForm.classList.remove('hidden');
        workspaceEmpty.classList.add('hidden');
        statsContainer.classList.remove('hidden');
        downloadBtn.classList.remove('hidden');
        if(tabEDA) tabEDA.classList.remove('hidden'); // UNLOCK EDA TAB
    }
}

async function handleGraphResponse(data, stepName) {
    if (stepName) await updateAuditAndStats(stepName);

    if (data.status === 'paused') {
        const state = data.pending_state || {};
        const msg = data.interrupt_msg || "";
        let aiResponse = "";

        if (state.error && state.error !== null && state.error !== "None") {
            aiResponse = `### ⚠️ Process Interrupted\n**Execution Error:**\n\`\`\`text\n${state.error}\n\`\`\`\n**${msg || "Please review and type a correction or 'skip'."}**`;
        } else if (msg.toLowerCase().includes("iteration executed successfully") || (state.cleaning_plan && AppState.hasShownCleaningPlan)) {
            aiResponse = `### ✨ Agent Paused\n${msg || "Iteration executed successfully. Review the updated statistics. Type 'approve' to finalize, or provide further cleaning instructions."}`;
            AppState.hasShownCleaningPlan = true;
        } else if (state.cleaning_plan && !AppState.hasShownCleaningPlan) {
            try {
                const plan = JSON.parse(state.cleaning_plan);
                aiResponse = `### 🔧 Logic Discovery: Cleaning Plan\nI have profiled the dataset and propose the following cleaning strategy:\n\n`;
                plan.actions.forEach(a => {
                    const target = a.target_column ? `Column \`${a.target_column}\`` : `Dataset Level`;
                    aiResponse += `* **${target}**: ${a.code_instruction} _[${a.action_type}]_\n`;
                });
                aiResponse += `\n**${msg || "Confirm this strategy by typing 'approve', or provide your own custom modifications."}**`;
                AppState.hasShownCleaningPlan = true;
            } catch (e) {
                aiResponse = `### 🔧 Logic Discovery: Cleaning Plan\nI have a proposed cleaning plan, but it could not be parsed. **${msg || "Please type your manual cleaning instructions to proceed."}**`;
                AppState.hasShownCleaningPlan = true;
            }
        } else if (state.suggestion || msg.toLowerCase().includes("merge")) {
            const strategyMsg = state.suggestion || "Suggested merge based on schema analysis.";
            aiResponse = `### 🧬 Logic Discovery: Merging\n**Strategy:** ${strategyMsg}\n\n**${msg || "Approve this merge or suggest an alternative."}**`;
        } else {
            aiResponse = `### 🔍 Data Process Paused\n**${msg || "The raw data has been loaded and initial checks are done. Does the preview look correct? Type 'approve' to continue."}**`;
        }
        
        addMessage('ai', aiResponse, true);
        setMode('resume');
        
    } else if (data.status === 'success') {
        addMessage('ai', "### ✅ Analysis Engine Ready\nETL Pipeline completed. Audit logs, deep EDA report, and data samples updated in the workspace. Ask me anything about the data.", true);
        setMode('chat');
    }
}

uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (fileInput.files.length === 0) return;
    
    // FILE COMPARISON LOGIC: Check if uploading new files or reusing old files
    const fileNames = Array.from(fileInput.files).map(f => f.name).join(',');
    const lastUploaded = sessionStorage.getItem('last_uploaded_files');

    if (fileNames !== lastUploaded) {
        AppState.threadId = crypto.randomUUID();
        sessionStorage.setItem('ai_thread_id', AppState.threadId);
        sessionStorage.setItem('last_uploaded_files', fileNames);
        AppState.auditLog = [];
        sessionStorage.removeItem('ai_audit_log');
    } else {
        AppState.threadId = sessionStorage.getItem('ai_thread_id');
    }

    AppState.hasShownCleaningPlan = false;
    
    const formData = new FormData();
    formData.append('thread_id', AppState.threadId);
    formData.append('user_input', uploadInstruction.value || "Load and process.");
    for (let i = 0; i < fileInput.files.length; i++) formData.append('files', fileInput.files[i]);

    addMessage('user', `Initializing ingestion for ${fileInput.files.length} file(s)...`);
    setLoading(true, 'Running Ingestion...');
    try {
        const res = await fetch(`${API_BASE_URL}/upload`, { method: 'POST', body: formData });
        const data = await res.json();
        if (res.ok) await handleGraphResponse(data, data.status === 'success' ? 'Session Restored' : 'Ingestion');
        else addMessage('error', data.error);
    } catch (e) { addMessage('error', 'Service Offline'); }
    finally { setLoading(false); }
});

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const input = chatInput.value.trim();
    if (!input) return;
    addMessage('user', input);
    chatInput.value = '';

    if (AppState.mode === 'resume') {
        setLoading(true, 'Updating Logic...');
        const formData = new FormData();
        formData.append('thread_id', AppState.threadId);
        formData.append('user_feedback', input);
        try {
            const res = await fetch(`${API_BASE_URL}/resume`, { method: 'POST', body: formData });
            const data = await res.json();
            if (res.ok) {
                const nextStep = data.status === 'success' ? 'Finalization' : 'Update';
                await handleGraphResponse(data, nextStep);
            } else addMessage('error', data.error);
        } catch (e) { addMessage('error', 'Service Offline'); }
        finally { setLoading(false); }
    } else {
        setLoading(true, 'Querying Engine...');
        const formData = new FormData();
        formData.append('thread_id', AppState.threadId);
        formData.append('message', input);
        try {
            const res = await fetch(`${API_BASE_URL}/chat`, { method: 'POST', body: formData });
            const data = await res.json();
            if (res.ok) {
                addMessage('ai', data.response, true);
                if (data.plot_url) renderChartInWorkspace(data.plot_url);
            } else addMessage('error', data.error);
        } catch (e) { addMessage('error', 'Service Offline'); }
        finally { setLoading(false); }
    }
});

// 3. TAB SWITCHING LOGIC (Updated to include EDA)
tabStats.addEventListener('click', () => {
    tabStats.className = 'text-blue-600 font-black border-b-2 border-blue-600 pb-1 text-xs uppercase tracking-wider';
    tabCharts.className = 'text-gray-400 font-black pb-1 text-xs uppercase tracking-wider hover:text-gray-600 transition';
    if(tabEDA) tabEDA.className = 'text-gray-400 font-black pb-1 text-xs uppercase tracking-wider hover:text-gray-600 transition';
    
    statsContainer.classList.remove('hidden');
    chartContainer.classList.add('hidden');
    if(edaContainer) edaContainer.classList.add('hidden');
});

tabCharts.addEventListener('click', () => {
    tabCharts.className = 'text-blue-600 font-black border-b-2 border-blue-600 pb-1 text-xs uppercase tracking-wider';
    tabStats.className = 'text-gray-400 font-black pb-1 text-xs uppercase tracking-wider hover:text-gray-600 transition';
    if(tabEDA) tabEDA.className = 'text-gray-400 font-black pb-1 text-xs uppercase tracking-wider hover:text-gray-600 transition';
    
    chartContainer.classList.remove('hidden');
    statsContainer.classList.add('hidden');
    if(edaContainer) edaContainer.classList.add('hidden');
});

if (tabEDA) {
    tabEDA.addEventListener('click', () => {
        tabEDA.className = 'text-blue-600 font-black border-b-2 border-blue-600 pb-1 text-xs uppercase tracking-wider';
        tabStats.className = 'text-gray-400 font-black pb-1 text-xs uppercase tracking-wider hover:text-gray-600 transition';
        tabCharts.className = 'text-gray-400 font-black pb-1 text-xs uppercase tracking-wider hover:text-gray-600 transition';
        
        edaContainer.classList.remove('hidden');
        statsContainer.classList.add('hidden');
        chartContainer.classList.add('hidden');

        // Fetch the report from the backend if it hasn't been loaded yet
        if (!edaFrame.src || edaFrame.src === '') {
            edaFrame.src = `${API_BASE_URL}/generate-eda`;
        }
    });
}

downloadBtn.addEventListener('click', () => { window.location.href = `${API_BASE_URL}/download`; });

function renderChartInWorkspace(plotUrl) {
    tabCharts.click();
    const url = `${API_BASE_URL}${plotUrl}`;
    const div = document.createElement('div');
    div.className = 'bg-white p-4 rounded-xl shadow-sm border border-gray-200 flex flex-col items-center';
    div.innerHTML = `<img src="${url}" class="max-w-full rounded-lg mb-2"><a href="${url}" target="_blank" class="text-[10px] font-bold text-blue-500 uppercase hover:underline">Download Image</a>`;
    chartContainer.prepend(div);
}