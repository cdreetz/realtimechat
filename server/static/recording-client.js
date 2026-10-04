/* Observations only: nothing here injects UI focus or edits into model prompts. */
class TrainingRecorder {
  constructor({socket, metadata = () => ({}), state = () => null, defaultEnabled = false, key}) {
    Object.assign(this, {socket, metadata, state, key});
    this.wanted = localStorage.getItem(key) === null ? defaultEnabled : localStorage.getItem(key) === "true";
    this.active = this.available = false;
    this.recordingId = null;
    this.clientSessionId = crypto.randomUUID();
    this.pendingLabels = new Map();
  }
  controls(parent) {
    const row = document.createElement("div");
    row.style.cssText = "display:flex;align-items:center;gap:8px;padding:4px;flex-wrap:wrap;font-size:12px";
    this.button = document.createElement("button");
    this.status = document.createElement("span");
    this.status.setAttribute("role", "status");
    row.append(this.button, this.status);
    parent.appendChild(row);
    this.button.onclick = () => {
      this.wanted = !this.wanted;
      localStorage.setItem(this.key, String(this.wanted));
      this.configure();
    };
    this.render("Connect to start recording");
  }
  render(message) {
    if (!this.button) return;
    this.button.textContent = this.wanted ? "Pause recording" : "Record session";
    this.button.disabled = !this.available;
    this.status.textContent = message || (this.active ? `● Recording audio + activity · ${this.recordingId.slice(0, 8)}` : "Recording paused");
    this.status.style.color = this.active ? "#c03030" : "inherit";
    this.status.title = "Saved on the speech server: microphone audio, text, model/tools and app observations. Feedback is optional.";
  }
  configure() {
    if (!this.available) return;
    this.render(this.wanted ? "Starting recording…" : "Saving recording…");
    this.socket().send(JSON.stringify({type: "set_recording", enabled: this.wanted,
      client: {client_session_id: this.clientSessionId, time_origin_ms: performance.timeOrigin,
        monotonic_ms: performance.now(), wall_time_ms: Date.now(), ...this.metadata()}}));
  }
  handle(m) {
    if (m.type === "ready") {
      this.available = m.recording_available === true;
      if (this.available) this.configure();
      else this.render("Recording unavailable on this server");
    } else if (m.type === "recording_status") {
      this.active = m.enabled;
      this.recordingId = m.recording_id;
      this.lastState = null;
      this.render(m.error ? `Recording stopped: ${m.error}` : null);
      if (this.active) { this.event("clock", this.metadata()); this.snapshot("recording_start"); }
    } else if (m.type === "clock_sync") {
      this.event("clock_reply", {...m, client_received_ms: performance.now()});
    } else if (m.type === "feedback_saved") {
      const row = this.pendingLabels.get(m.client_event_id);
      if (row) row.textContent = m.event_id ? "Feedback saved" : "Feedback not saved: recording paused or changed";
      this.pendingLabels.delete(m.client_event_id);
    }
  }
  disconnected() {
    this.active = this.available = false;
    for (const row of this.pendingLabels.values()) row.textContent = "Feedback save unconfirmed: disconnected";
    this.pendingLabels.clear();
    this.render("Recording disconnected");
  }
  event(event, data = {}) {
    const ws = this.socket();
    if (!this.active || !ws || ws.readyState !== WebSocket.OPEN) return null;
    const id = crypto.randomUUID();
    ws.send(JSON.stringify({...data, type: "client_event", event,
      recording_id: this.recordingId, client_session_id: this.clientSessionId,
      client_event_id: id, monotonic_ms: performance.now(), wall_time_ms: Date.now()}));
    return id;
  }
  snapshot(reason, extra = {}) {
    if (!this.active) return;
    const state = this.state(), encoded = JSON.stringify(state);
    if (reason === "periodic" && encoded === this.lastState) return;
    this.lastState = encoded;
    this.event("app_state", {reason, state, ...extra});
  }
  feedback(parent, target) {
    if (!this.active) return;
    const recordingId = this.recordingId, row = document.createElement("div");
    row.style.cssText = "display:flex;gap:4px;font-size:11px;flex-wrap:wrap;margin:3px 0";
    const submit = (label, correction = null) => {
      if (!this.active || recordingId !== this.recordingId) {
        row.textContent = "This recording has ended; label it offline."; return;
      }
      const id = this.event("feedback", {target, label, correction, source: "explicit_human"});
      if (id) { row.textContent = "Saving feedback…"; this.pendingLabels.set(id, row); }
    };
    for (const [label, caption] of [["accepted", "Correct"], ["incorrect", "Incorrect"], ["revised", "Correct / revise…"]]) {
      const button = document.createElement("button");
      button.textContent = caption;
      button.style.cssText = "font:inherit;padding:2px 5px";
      button.onclick = () => {
        if (!this.active || recordingId !== this.recordingId) {
          row.textContent = "This recording has ended; label it offline."; return;
        }
        if (label === "revised") {
          const original = [...row.childNodes];
          const input = document.createElement("textarea");
          input.placeholder = "What should it have said or done?";
          input.setAttribute("aria-label", "Desired correction");
          input.style.cssText = "width:100%;min-height:55px;font:inherit;box-sizing:border-box";
          const save = document.createElement("button"), cancel = document.createElement("button");
          save.textContent = "Save correction"; cancel.textContent = "Cancel";
          save.onclick = () => submit(label, input.value);
          cancel.onclick = () => row.replaceChildren(...original);
          row.replaceChildren(input, save, cancel);
          input.focus();
          return;
        }
        submit(label);
      };
      row.appendChild(button);
    }
    parent.appendChild(row);
  }
  playback(source, context, start, samples, rate, target) {
    const recordingId = this.recordingId;
    this.event("playback_scheduled", {...target, audio_context_s: context.currentTime,
      scheduled_start_s: start, samples, sample_rate: rate, context_state: context.state});
    source.traceStopped = false;
    source.onended = () => {
      if (recordingId !== this.recordingId) return;
      this.event("playback_ended", {...target, stopped: source.traceStopped,
        audio_context_s: context.currentTime, context_state: context.state,
        estimated_played_samples: Math.min(samples, Math.max(0, Math.floor(((source.traceStopTime ?? context.currentTime) - start) * rate))),
        evidence: "Web Audio timing estimate; not proof of audible output"});
    };
  }
}
