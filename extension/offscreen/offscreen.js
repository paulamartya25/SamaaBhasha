// offscreen.js — Runs in hidden offscreen document; handles actual audio recording

let mediaRecorder = null;
let targetLang    = 'eng_Latn';
const CHUNK_MS    = 3000; // Send audio every 3 seconds

// ── Listen for commands from background ───────────────────────────────────
chrome.runtime.onMessage.addListener(async (msg) => {
  if (msg.type === 'START_RECORDING') {
    await startRecording(msg.streamId, msg.targetLang);
  } else if (msg.type === 'STOP_RECORDING') {
    stopRecording();
  }
});

// ── Start recording from tab's audio stream ────────────────────────────────
async function startRecording(streamId, lang) {
  targetLang = lang;

  // Get tab audio stream using the streamId from background
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      mandatory: {
        chromeMediaSource: 'tab',
        chromeMediaSourceId: streamId,
      },
    },
    video: false,
  });

  // Use MediaRecorder to capture chunks
  mediaRecorder = new MediaRecorder(stream, {
    mimeType: 'audio/webm;codecs=opus',
  });

  mediaRecorder.ondataavailable = async (event) => {
    if (event.data.size === 0) return;

    // Convert blob to base64 and send to background
    const base64 = await blobToBase64(event.data);
    chrome.runtime.sendMessage({
      type:       'AUDIO_CHUNK',
      audio:      base64,
      targetLang: targetLang,
    });
  };

  mediaRecorder.start(CHUNK_MS); // Fire ondataavailable every 3 seconds
  console.log('[SamaaBhasha Offscreen] Recording started');
}

// ── Stop recording ─────────────────────────────────────────────────────────
function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());
    console.log('[SamaaBhasha Offscreen] Recording stopped');
  }
}

// ── Helper: Blob → Base64 ──────────────────────────────────────────────────
function blobToBase64(blob) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result.split(',')[1]);
    reader.readAsDataURL(blob);
  });
}
