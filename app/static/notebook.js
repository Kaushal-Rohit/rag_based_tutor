/**
 * Adaptive Notebook — Client-side Logic
 * ======================================
 * Handles: SSE streaming, file upload (drag & drop + click), source panel,
 *          adaptive state display, and session management.
 */

(function () {
  'use strict';

  // ── Session ──
  let SESSION_ID = localStorage.getItem('notebook_session_id');
  if (!SESSION_ID) {
    SESSION_ID = crypto.randomUUID();
    localStorage.setItem('notebook_session_id', SESSION_ID);
  }

  // ── DOM refs ──
  const messagesContainer = document.getElementById('messages');
  const queryInput = document.getElementById('query-input');
  const sendBtn = document.getElementById('send-btn');
  const fileInput = document.getElementById('file-input');
  const uploadZone = document.getElementById('upload-zone');
  const sourceList = document.getElementById('source-list');

  // Adaptive state refs
  const sentimentEl = document.getElementById('sentiment-value');
  const stateEl = document.getElementById('state-badge');
  const kEl = document.getElementById('k-value');

  let isStreaming = false;

  // ── Initialize ──
  loadSessionSources();

  // ── Input handling ──
  queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendQuery();
    }
  });

  queryInput.addEventListener('input', () => {
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 150) + 'px';
  });

  sendBtn.addEventListener('click', sendQuery);

  // ── Upload: click ──
  uploadZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) uploadFile(e.target.files[0]);
  });

  // ── Upload: drag & drop ──
  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
  });
  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
  });
  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
  });


  // ═══════════════════════════════════════
  // Query & Streaming
  // ═══════════════════════════════════════

  async function sendQuery() {
    const query = queryInput.value.trim();
    if (!query || isStreaming) return;

    // Add user message
    addMessage('user', query);
    queryInput.value = '';
    queryInput.style.height = 'auto';
    isStreaming = true;
    sendBtn.disabled = true;

    // Show typing indicator
    const typingEl = addTypingIndicator();

    try {
      const res = await fetch('/api/v1/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify({
          query: query,
          session_id: SESSION_ID,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.message || `HTTP ${res.status}`);
      }

      // Read SSE stream
      typingEl.remove();
      const msgEl = addMessage('assistant', '');
      const contentEl = msgEl.querySelector('.message-text');

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let fullText = '';
      let metadata = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep incomplete line

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            var eventType = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            const data = line.slice(6);

            if (eventType === 'metadata') {
              metadata = JSON.parse(data);
              updateAdaptiveState(metadata);
            } else if (eventType === 'token') {
              fullText += data;
              contentEl.textContent = fullText;
              scrollToBottom();
            } else if (eventType === 'done') {
              // Render sources if metadata has them
              if (metadata && metadata.sources) {
                addSourceCitations(msgEl, metadata.sources);
              }
            }
          }
        }
      }

      // If we didn't get SSE (non-streaming fallback)
      if (!fullText && res.headers.get('content-type')?.includes('json')) {
        const json = JSON.parse(decoder.decode());
        contentEl.textContent = json.answer || '';
        if (json.sources) addSourceCitations(msgEl, json.sources);
        updateAdaptiveState(json);
      }

    } catch (err) {
      typingEl.remove();
      addMessage('assistant', `⚠️ Error: ${err.message}`);
      showToast(err.message, 'error');
    } finally {
      isStreaming = false;
      sendBtn.disabled = false;
      queryInput.focus();
    }
  }


  // ═══════════════════════════════════════
  // File Upload
  // ═══════════════════════════════════════

  async function uploadFile(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      showToast('Please upload a PDF file.', 'error');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', SESSION_ID);

    showToast(`Uploading ${file.name}...`, 'info');

    try {
      const res = await fetch('/api/v1/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.message || `Upload failed: ${res.status}`);
      }

      showToast(`${file.name} queued for processing`, 'success');
      pollJobStatus(data.job_id);

    } catch (err) {
      showToast(err.message, 'error');
    }

    fileInput.value = '';
  }

  async function pollJobStatus(jobId) {
    const poll = async () => {
      try {
        const res = await fetch(`/api/v1/upload/${jobId}/status`);
        const data = await res.json();

        updateSourceItem(data);

        if (data.status === 'ready') {
          showToast(`${data.filename} — ${data.chunks_created} chunks indexed`, 'success');
          return;
        } else if (data.status === 'failed') {
          showToast(`${data.filename} failed: ${data.error}`, 'error');
          return;
        }

        setTimeout(poll, 1500);
      } catch { setTimeout(poll, 3000); }
    };

    poll();
  }


  // ═══════════════════════════════════════
  // Source Panel
  // ═══════════════════════════════════════

  async function loadSessionSources() {
    try {
      const res = await fetch(`/api/v1/upload/session/${SESSION_ID}`);
      const sources = await res.json();
      sourceList.innerHTML = '';
      sources.forEach(s => updateSourceItem(s));
    } catch { /* ignore on first load */ }
  }

  function updateSourceItem(data) {
    let el = document.getElementById(`source-${data.job_id}`);

    if (!el) {
      el = document.createElement('div');
      el.id = `source-${data.job_id}`;
      el.className = 'source-item selected';
      el.onclick = () => el.classList.toggle('selected');
      sourceList.prepend(el);
    }

    const statusClass = data.status === 'ready' ? 'ready' :
                         data.status === 'failed' ? 'failed' : 'processing';

    el.innerHTML = `
      <span class="source-icon">📄</span>
      <div class="source-info">
        <div class="source-name" title="${data.filename}">${data.filename}</div>
        <div class="source-meta">
          <span class="source-status ${statusClass}">${data.status}</span>
          ${data.chunks_created ? `<span>${data.chunks_created} chunks</span>` : ''}
          ${data.pages_processed ? `<span>${data.pages_processed} pages</span>` : ''}
        </div>
      </div>
    `;
  }


  // ═══════════════════════════════════════
  // UI Helpers
  // ═══════════════════════════════════════

  function addMessage(role, text) {
    // Remove welcome message if present
    const welcome = document.getElementById('welcome');
    if (welcome) welcome.remove();

    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.innerHTML = `
      <div class="message-content">
        <div class="message-text">${escapeHtml(text)}</div>
      </div>
    `;
    messagesContainer.appendChild(div);
    scrollToBottom();
    return div;
  }

  function addSourceCitations(msgEl, sources) {
    if (!sources || !sources.length) return;

    const bar = document.createElement('div');
    bar.className = 'sources-bar';

    const uniqueSources = new Map();
    sources.forEach(s => {
      const key = (s.filename || s.chunk_id || 'Unknown') + (s.page ? `:p${s.page}` : '');
      if (!uniqueSources.has(key)) uniqueSources.set(key, s);
    });

    uniqueSources.forEach(s => {
      const tag = document.createElement('span');
      tag.className = 'source-tag';
      const name = s.filename ? s.filename.replace(/^.*[\\/]/, '').slice(0, 30) : s.chunk_id;
      tag.innerHTML = `📎 ${escapeHtml(name)}${s.page ? ` <span class="page-num">p.${s.page}</span>` : ''}`;
      bar.appendChild(tag);
    });

    const content = msgEl.querySelector('.message-content');
    if (content) content.appendChild(bar);
  }

  function addTypingIndicator() {
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.innerHTML = `<div class="typing-indicator"><span></span><span></span><span></span></div>`;
    messagesContainer.appendChild(div);
    scrollToBottom();
    return div;
  }

  function updateAdaptiveState(data) {
    if (data.sentiment_score !== undefined && sentimentEl) {
      sentimentEl.textContent = data.sentiment_score.toFixed(2);
    }
    if (data.user_state && stateEl) {
      stateEl.textContent = data.user_state;
      stateEl.className = `badge ${data.user_state}`;
    }
    if (data.k_used !== undefined && kEl) {
      kEl.textContent = data.k_used;
    }
  }

  function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  function showToast(msg, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(40px)';
      toast.style.transition = '0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

})();
