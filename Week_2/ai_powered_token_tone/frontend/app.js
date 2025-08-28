document.addEventListener("DOMContentLoaded", () => {
  const API_BASE = "http://127.0.0.1:8000";

  const $ = (id) => document.getElementById(id);
  const btn = $("analyzeBtn");
  const err = $("error");
  const res = $("result");
  const msg = $("message");
  const ticket = $("ticket");
  const emotion = $("emotion");
  const conf = $("confidence");
  const theme = $("theme");
  const sourceEl = $("source");

  btn.addEventListener("click", async () => {
    err.hidden = true;
    res.hidden = true;
    btn.disabled = true;

    const text = $("problem").value.trim();
    if (text.length < 5) {
      err.textContent = "Please enter a bit more detail.";
      err.hidden = false;
      btn.disabled = false;
      return;
    }

    try {
      const r = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ problem: text }),
      });

      if (!r.ok) {
        const j = await r.json().catch(() => ({ detail: "Unknown error" }));
        throw new Error(j.detail || `Request failed: ${r.status}`);
      }

      const data = await r.json();

      // Populate UI
      msg.textContent = data.message;
      ticket.textContent = data.ticket_id;
      emotion.textContent = data.emotion;
      conf.textContent = data.confidence ? `${(data.confidence * 100).toFixed(1)}%` : "N/A";

      // Source ID with color
      sourceEl.textContent = data.source || "N/A";
      if (data.source === "OPENAI") {
        sourceEl.className = "source-openai";
      } else {
        sourceEl.className = "source-internal";
      }

      // Populate themes
      theme.innerHTML = "";
      const themeData = Array.isArray(data.theme) ? data.theme : [data.theme];
      themeData.forEach((t) => {
        const li = document.createElement("li");
        li.textContent = t;
        theme.appendChild(li);
      });

      res.hidden = false;
    } catch (e) {
      err.textContent = e.message;
      err.hidden = false;
    } finally {
      btn.disabled = false;
    }
  });
});
