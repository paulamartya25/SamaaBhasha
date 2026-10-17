// background.js — SamaaBhasha Service Worker

const BACKEND_WS_URL = 'ws://YOUR_ORACLE_VM_IP:8000/ws/caption';
// ⚠️ Replace with your Oracle Cloud VM IP when deployed
// For local dev: 'ws://localhost:8000/ws/caption'

let ws = null;
let activeTabId = null;
let offscreenCreated = false;

// ── Message router ─────────────────────────────────────────────────────────
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  switch (msg.type) {
    case 'START_CAPTURE':
      startCapture(msg.tabId, msg.targetLang);
      break;
    case 'STOP_CAPTURE':
      stopCapture();
      break;
    case 'AUDIO_CHUNK':
      // Received audio chunk from offscreen — send to backend
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          audio: msg.audio,      // base64 encoded audio
          targetLang: msg.targetLang,
        }));
      }
      break;
  }
});

// ── Start capturing tab audio ──────────────────────────────────────────────
async function startCapture(tabId, targetLang) {
  activeTabId = tabId;

  // 1. Open WebSocket to backend
  connectWebSocket(targetLang);

  // 2. Create offscreen document for audio capture
  await setupOffscreen();

  // 3. Get tab media stream and pass to offscreen
  const streamId = await new Promise((resolve) => {
    chrome.tabCapture.getMediaStreamId({ targetTabId: tabId }, resolve);
  });

  chrome.runtime.sendMessage({
    type: 'START_RECORDING',
    streamId,
    targetLang,
  });
}

// ── Stop capture ───────────────────────────────────────────────────────────
function stopCapture() {
  chrome.runtime.sendMessage({ type: 'STOP_RECORDING' });
  if (ws) { ws.close(); ws = null; }
}

// ── WebSocket connection to FastAPI backend ────────────────────────────────
function connectWebSocket(targetLang) {
  ws = new WebSocket(`${BACKEND_WS_URL}?lang=${targetLang}`);

  ws.onopen  = () => console.log('[SamaaBhasha] WebSocket connected');
  ws.onclose = () => console.log('[SamaaBhasha] WebSocket closed');
  ws.onerror = (e) => console.error('[SamaaBhasha] WS error:', e);

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    // Forward caption to content script
    if (activeTabId) {
      chrome.tabs.sendMessage(activeTabId, {
        type: 'NEW_CAPTION',
        text: data.caption,
        original: data.original,
        lang: data.detected_lang,
      }).catch(() => {});
    }
  };
}

// ── Create offscreen document (audio recording lives here) ─────────────────
async function setupOffscreen() {
  if (offscreenCreated) return;
  await chrome.offscreen.createDocument({
    url: chrome.runtime.getURL('offscreen/offscreen.html'),
    reasons: ['USER_MEDIA'],
    justification: 'Capture tab audio for real-time transcription',
  });
  offscreenCreated = true;
}
