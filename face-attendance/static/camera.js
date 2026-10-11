// Shared camera helpers for the enrol and scan pages.
const Cam = {
  stream: null,
  async start(video) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Camera is not available. Open this site with https:// in a normal browser tab.");
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 1280 } }, audio: false,
    });
    video.srcObject = this.stream;
    await video.play();
  },
  stop() { if (this.stream) this.stream.getTracks().forEach(t => t.stop()); this.stream = null; },
  frame(video, maxW = 640, quality = 0.85) {
    const w = Math.min(video.videoWidth, maxW), h = Math.round(video.videoHeight * w / video.videoWidth);
    const c = document.createElement("canvas"); c.width = w; c.height = h;
    c.getContext("2d").drawImage(video, 0, 0, w, h);   // unmirrored, as the server expects
    return c.toDataURL("image/jpeg", quality);
  },
  sleep(ms) { return new Promise(r => setTimeout(r, ms)); },
};
async function postJSON(url, body) {
  const r = await fetch(url, { method: "POST", credentials: "same-origin",
    headers: { "Content-Type": "application/json", "X-CSRF-Token": window.CSRF },
    body: JSON.stringify(body || {}) });
  let data = {}; try { data = await r.json(); } catch (e) { data = { ok: false, error: "Server error (" + r.status + ")." }; }
  return data;
}
function cameraError(e) {
  if (e && e.name === "NotAllowedError") return "Camera permission was blocked. Allow the camera for this site and reload.";
  if (e && e.name === "NotFoundError") return "No camera found on this device.";
  return (e && e.message) || "Could not start the camera.";
}
