// popup.js — SamaaBhasha Extension Popup Logic

const toggleBtn       = document.getElementById('toggleBtn');
const statusDot       = document.getElementById('statusDot');
const statusText      = document.getElementById('statusText');
const targetLanguage  = document.getElementById('targetLanguage');
const fontSizeSlider  = document.getElementById('fontSize');
const fontSizeVal     = document.getElementById('fontSizeVal');
const downloadBtn     = document.getElementById('downloadBtn');
const modeBtns        = document.querySelectorAll('.mode-btn');
const captionPosition = document.getElementById('captionPosition');

let isActive = false;

// ── Load saved settings ────────────────────────────────────────────────────
chrome.storage.local.get(
  ['targetLang', 'fontSize', 'isActive', 'captionPosition', 'mode'],
  (data) => {
    if (data.targetLang)      targetLanguage.value   = data.targetLang;
    if (data.fontSize)        { fontSizeSlider.value = data.fontSize; fontSizeVal.textContent = data.fontSize + 'px'; }
    if (data.captionPosition) captionPosition.value  = data.captionPosition;
    if (data.mode)            setMode(data.mode);
    if (data.isActive)        setActiveState(data.isActive);
  }
);

// ── Mode toggle ────────────────────────────────────────────────────────────
modeBtns.forEach(btn => {
  btn.addEventListener('click', () => setMode(btn.dataset.mode));
});

function setMode(mode) {
  modeBtns.forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
  chrome.storage.local.set({ mode });
}

// ── Font size slider ───────────────────────────────────────────────────────
fontSizeSlider.addEventListener('input', () => {
  fontSizeVal.textContent = fontSizeSlider.value + 'px';
  chrome.storage.local.set({ fontSize: fontSizeSlider.value });
  sendToContent({ type: 'UPDATE_STYLE', fontSize: fontSizeSlider.value });
});

// ── Language change ────────────────────────────────────────────────────────
targetLanguage.addEventListener('change', () => {
  chrome.storage.local.set({ targetLang: targetLanguage.value });
  sendToContent({ type: 'UPDATE_LANG', lang: targetLanguage.value });
});

// ── Caption position ───────────────────────────────────────────────────────
captionPosition.addEventListener('change', () => {
  chrome.storage.local.set({ captionPosition: captionPosition.value });
  sendToContent({ type: 'UPDATE_POSITION', position: captionPosition.value });
});

// ── Main toggle button ─────────────────────────────────────────────────────
toggleBtn.addEventListener('click', async () => {
  isActive = !isActive;
  setActiveState(isActive);
  chrome.storage.local.set({ isActive });

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  chrome.runtime.sendMessage({
    type: isActive ? 'START_CAPTURE' : 'STOP_CAPTURE',
    tabId: tab.id,
    targetLang: targetLanguage.value,
  });
});

function setActiveState(active) {
  isActive = active;
  toggleBtn.textContent = active ? '⏹ Stop Captions' : '▶ Start Captions';
  toggleBtn.classList.toggle('active', active);
  statusDot.className   = 'status-dot' + (active ? ' active' : '');
  statusText.textContent = active ? 'Live — Captioning...' : 'Ready';
}

// ── Download transcript ────────────────────────────────────────────────────
downloadBtn.addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  chrome.tabs.sendMessage(tab.id, { type: 'DOWNLOAD_TRANSCRIPT' });
});

// ── Helper: send message to active tab ────────────────────────────────────
async function sendToContent(msg) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab?.id) chrome.tabs.sendMessage(tab.id, msg).catch(() => {});
}
