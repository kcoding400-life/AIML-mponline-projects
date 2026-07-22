const chatEl = document.getElementById("chat");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const statusDot = document.getElementById("status-dot");
const sourceLabel = document.getElementById("source-label");

let history = []; // {role, content}

function addMessage(role, contentHtml) {
  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = contentHtml;
  wrap.appendChild(bubble);
  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
  return bubble;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderSources(sources) {
  if (!sources || sources.length === 0) return "";
  const items = sources
    .map(
      (s) =>
        `<div class="excerpt"><strong>Excerpt ${s.rank}</strong> (relevance ${s.score}) — ${escapeHtml(
          s.excerpt
        )}...</div>`
    )
    .join("");
  return `<details class="sources"><summary>View source excerpts (${sources.length})</summary>${items}</details>`;
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.index_ready && data.has_api_key) {
      statusDot.className = "status-dot ok";
      statusDot.title = "Ready";
      if (data.meta && data.meta.source_label) {
        sourceLabel.textContent = `Source: ${data.meta.source_label}`;
      }
    } else if (!data.has_api_key) {
      statusDot.className = "status-dot bad";
      statusDot.title = "Missing GEMINI_API_KEY";
    } else {
      statusDot.className = "status-dot bad";
      statusDot.title = "Index not ready yet";
    }
  } catch (e) {
    statusDot.className = "status-dot bad";
    statusDot.title = "Server unreachable";
  }
}

formEl.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = inputEl.value.trim();
  if (!message) return;

  addMessage("user", escapeHtml(message));
  inputEl.value = "";
  sendBtn.disabled = true;

  const thinkingBubble = addMessage("assistant", `<span class="typing">Thinking...</span>`);

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history }),
    });
    const data = await res.json();

    if (!res.ok) {
      thinkingBubble.innerHTML = `<span style="color:#e5534b">Error: ${escapeHtml(
        data.detail || "Something went wrong."
      )}</span>`;
      sendBtn.disabled = false;
      return;
    }

    thinkingBubble.innerHTML = escapeHtml(data.answer).replace(/\n/g, "<br>") + renderSources(data.sources);

    history.push({ role: "user", content: message });
    history.push({ role: "assistant", content: data.answer });
    // Keep history bounded so requests don't grow unbounded
    if (history.length > 20) history = history.slice(-20);
  } catch (err) {
    thinkingBubble.innerHTML = `<span style="color:#e5534b">Network error — please try again.</span>`;
  }
  sendBtn.disabled = false;
  inputEl.focus();
});

checkHealth();
setInterval(checkHealth, 15000);
