let selectedModel = "";
let modelsList = [];
let recognition = null;
let isListening = false;

function renderModelDropdown(models) {
  const dropdown = document.getElementById('model-dropdown');
  dropdown.innerHTML = models.map(model => `
    <option value="${model.id}"${model.id === selectedModel ? ' selected' : ''}>${model.name}</option>
  `).join('');
}

function getTimeGreeting() {
  const hour = new Date().getHours();
  if (hour < 12) return "Good morning";
  if (hour < 18) return "Good afternoon";
  return "Good evening";
}

function setBotGreeting() {
  const botname = document.getElementById('botname');
  if (botname) botname.textContent = `${getTimeGreeting()}, ${window.username || ''}!`;
}

document.getElementById('model-dropdown')?.addEventListener('change', function(e) {
  selectedModel = e.target.value;
  // Remove model name from botname, set greeting instead
  setBotGreeting();
});

function paragraphToBullets(text) {
  // Prefer splitting on newlines if present
  let lines = text.split('\n').map(s => s.trim()).filter(Boolean);
  // If only one line, split by period
  if (lines.length === 1) {
    lines = text.split('.').map(s => s.trim()).filter(Boolean);
  }
  return '<ul>' + lines.map(s => `<li>${escapeHtml(s)}</li>`).join('') + '</ul>';
}

function addMessage(text, isUser, options = {}) {
  const chatBody = document.getElementById('chat-body');
  const msgRow = document.createElement('div');
  msgRow.className = 'message-row ' + (isUser ? 'user' : 'bot');
  const avatarSrc = isUser ? '/static/user.png' : '/static/bot.png';

  if (!isUser) {
    // Always render bot messages as markdown
    msgRow.innerHTML = `${!isUser ? `<img src="${avatarSrc}" class="avatar" />` : ''}<div class="message-bubble bot">${marked.parse(text)}</div>`;
  } else {
    msgRow.innerHTML = `${!isUser ? `<img src="${avatarSrc}" class="avatar" />` : ''}<div class="message-bubble user">${escapeHtml(text)}</div>${isUser ? `<img src="${avatarSrc}" class="avatar" />` : ''}`;
  }
  chatBody.appendChild(msgRow);

  // Feedback for bot messages (optional)
  if (!isUser && options.feedback) {
    const feedbackRow = document.createElement('div');
    feedbackRow.className = 'feedback-row';
    feedbackRow.innerHTML = `
      <button class="feedback-btn" title="Helpful">👍</button>
      <button class="feedback-btn" title="Not helpful">👎</button>
    `;
    let selected = null;
    feedbackRow.querySelectorAll('.feedback-btn').forEach((btn, idx) => {
      btn.onclick = function() {
        feedbackRow.querySelectorAll('.feedback-btn').forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
        selected = idx === 0 ? 'yes' : 'no';

        // Find the last user and bot message
        const chatBody = document.getElementById('chat-body');
        const userMessages = chatBody.querySelectorAll('.message-row.user .message-bubble');
        const botMessages = chatBody.querySelectorAll('.message-row.bot .message-bubble');
        const lastUserMsg = userMessages[userMessages.length - 1]?.textContent || '';
        const lastBotMsg = botMessages[botMessages.length - 1]?.textContent || '';

        // Send feedback to backend
        fetch('/feedback', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            question: lastUserMsg,
            answer: lastBotMsg,
            feedback: selected
          })
        });
      };
    });
    chatBody.appendChild(feedbackRow);
  }

  chatBody.scrollTop = chatBody.scrollHeight;
}

function escapeHtml(text) {
  var div = document.createElement('div');
  div.innerText = text;
  return div.innerHTML;
}

function showThinking() {
  const chatBody = document.getElementById('chat-body');
  const thinkingRow = document.createElement('div');
  thinkingRow.className = 'thinking-row';
  thinkingRow.id = 'thinking-row';
  thinkingRow.innerHTML = `
    <span class="thinking-dot"></span>
    <span class="thinking-dot"></span>
    <span class="thinking-dot"></span>
  `;
  chatBody.appendChild(thinkingRow);
  chatBody.scrollTop = chatBody.scrollHeight;
}

function removeThinking() {
  const row = document.getElementById('thinking-row');
  if (row) row.remove();
}

function sendMessage() {
  const input = document.getElementById('user-input');
  const text = input.value.trim();
  if (!text) return false;
  addMessage(text, true, {question: true});
  input.value = '';
  showThinking();
  fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question: text, model_id: selectedModel})
  })
  .then(res => res.json())
  .then(data => {
    removeThinking();
    addMessage(data.answer, false, {feedback: true});
  })
  .catch(() => {
    removeThinking();
    addMessage('Sorry, I could not reach the server.', false);
  });
  return false;
}

function uploadFile() {
  const input = document.getElementById('file-input');
  const status = document.getElementById('upload-status');
  if (!input.files.length) {
    status.textContent = "Please select a file.";
    return;
  }
  const file = input.files[0];
  const formData = new FormData();
  formData.append('file', file);

  status.textContent = "Uploading...";
  fetch('/upload', {
    method: 'POST',
    body: formData
  })
  .then(res => res.json())
  .then(data => {
    status.textContent = data.message;
    if (data.status === 'ok') {
      input.value = '';
    }
  })
  .catch(() => {
    status.textContent = "Upload failed.";
  });
}

function initSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    console.warn('Speech recognition not supported in this browser');
    return;
  }
  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = 'en-US';

  recognition.onstart = () => {
    isListening = true;
    const voiceBtn = document.getElementById('voice-btn');
    if (voiceBtn) voiceBtn.classList.add('listening');
  };

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    const userInput = document.getElementById('user-input');
    if (userInput) userInput.value = transcript;
    const voiceBtn = document.getElementById('voice-btn');
    if (voiceBtn) voiceBtn.classList.remove('listening');
    isListening = false;
  };

  recognition.onerror = (event) => {
    console.error('Speech recognition error:', event.error);
    const voiceBtn = document.getElementById('voice-btn');
    if (voiceBtn) voiceBtn.classList.remove('listening');
    isListening = false;
  };

  recognition.onend = () => {
    const voiceBtn = document.getElementById('voice-btn');
    if (voiceBtn) voiceBtn.classList.remove('listening');
    isListening = false;
  };
}

function toggleVoiceInput() {
  if (!recognition) return;
  if (isListening) {
    recognition.stop();
  } else {
    recognition.start();
  }
}

window.onload = function() {
  fetch('/models').then(res => res.json()).then(models => {
    modelsList = models;
    // Set selectedModel to the first model in the list by default
    selectedModel = models.length > 0 ? models[0].id : "";
    renderModelDropdown(models);
    const dropdown = document.getElementById('model-dropdown');
    dropdown.value = selectedModel;
    // Set greeting in top bar
    setBotGreeting();
  });
  // Add personalized greeting to chat area
  addMessage(`Hello, ${window.username || "there"}! How can I assist you today?`, false);

  // Voice input setup
  initSpeechRecognition();
  const voiceBtn = document.getElementById('voice-btn');
  if (voiceBtn && recognition) {
    voiceBtn.addEventListener('click', toggleVoiceInput);
  }
};

// Emoji picker (basic, just insert emoji)
document.querySelectorAll('.footer-icon')[1].onclick = function() {
  const input = document.getElementById('user-input');
  input.value += '😊';
  input.focus();
};

function focusInput() {
  document.getElementById('user-input').focus();
} 