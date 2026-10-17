// content.js — SamaaBhasha Caption Overlay (injected into every page)

let captionBox   = null;
let transcript   = [];
let fontSize     = 18;
let position     = 'bottom';

// ── Create caption overlay UI ──────────────────────────────────────────────
function createCaptionOverlay() {
  if (captionBox) return;

  captionBox = document.createElement('div');
  captionBox.id = 'samaa-caption-box';
  captionBox.innerHTML = `
    <div id="samaa-lang-badge"></div>
    <div id="samaa-caption-text">🌐 SamaaBhasha ready...</div>
  `;
  document.body.appendChild(captionBox);
  applyStyle();
}

function applyStyle() {
  if (!captionBox) return;
  Object.assign(captionBox.style, {
    position:      'fixed',
    [position === 'bottom' ? 'bottom' : 'top']: '20px',
    left:          '50%',
    transform:     'translateX(-50%)',
    zIndex:        '999999',
    maxWidth:      '80vw',
    minWidth:      '300px',
    background:    'rgba(0,0,0,0.82)',
    color:         '#fff',
    padding:       '12px 20px',
    borderRadius:  '12px',
    fontSize:      `${fontSize}px`,
    lineHeight:    '1.5',
    textAlign:     'center',
    backdropFilter:'blur(6px)',
    border:        '1px solid rgba(124,58,237,0.4)',
    boxShadow:     '0 4px 24px rgba(0,0,0,0.5)',
    transition:    'all 0.3s ease',
    fontFamily:    '"Noto Sans", "Segoe UI", sans-serif',
  });
}

function removeCaptionOverlay() {
  if (captionBox) { captionBox.remove(); captionBox = null; }
}

// ── Update caption text ────────────────────────────────────────────────────
function showCaption(text, originalText, detectedLang) {
  if (!captionBox) createCaptionOverlay();

  const captionEl = document.getElementById('samaa-caption-text');
  const badgeEl   = document.getElementById('samaa-lang-badge');

  captionEl.textContent = text;
  if (detectedLang) {
    badgeEl.textContent = `🔊 ${detectedLang}`;
    Object.assign(badgeEl.style, {
      fontSize: '10px', color: '#a78bfa', marginBottom: '4px',
    });
  }

  // Save to transcript
  transcript.push({
    time:     new Date().toLocaleTimeString(),
    original: originalText,
    caption:  text,
    lang:     detectedLang,
  });

  // Auto-fade after 6 seconds
  captionEl.style.opacity = '1';
  clearTimeout(captionBox._fadeTimer);
  captionBox._fadeTimer = setTimeout(() => {
    captionEl.style.opacity = '0.4';
  }, 6000);
}

// ── Download transcript as .txt ────────────────────────────────────────────
function downloadTranscript() {
  const lines = transcript.map(
    t => `[${t.time}] (${t.lang}) ${t.original}\n→ ${t.caption}`
  ).join('\n\n');

  const blob = new Blob([lines], { type: 'text/plain' });
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = `SamaaBhasha_transcript_${Date.now()}.txt`;
  a.click();
}

// ── Listen for messages from background/popup ──────────────────────────────
chrome.runtime.onMessage.addListener((msg) => {
  switch (msg.type) {
    case 'NEW_CAPTION':
      showCaption(msg.text, msg.original, msg.lang);
      break;
    case 'UPDATE_STYLE':
      fontSize = parseInt(msg.fontSize);
      applyStyle();
      break;
    case 'UPDATE_POSITION':
      position = msg.position;
      applyStyle();
      break;
    case 'UPDATE_LANG':
      // Language changed — just acknowledge
      break;
    case 'DOWNLOAD_TRANSCRIPT':
      downloadTranscript();
      break;
    case 'STOP_CAPTION':
      removeCaptionOverlay();
      break;
  }
});
