document.addEventListener('DOMContentLoaded', (event) => {
    const messageArea = document.querySelector('.message-area');
    const promptForm = document.querySelector('.prompt-form');
    const promptInput = promptForm.querySelector('input[name="prompt"]');
    const sendButton = promptForm.querySelector('.send-btn');
    const micBtn = document.getElementById('mic-btn');
    const audioPreview = document.getElementById('audio-preview');
    const fileUploadInput = document.getElementById('file-upload');

    // Debug function
    const debug = (message, data = '') => {
        console.log(`[DEBUG] ${message}`, data);
    };

    const scrollToBottom = () => {
        if (messageArea) {
            messageArea.scrollTop = messageArea.scrollHeight;
        }
    };

    const applySyntaxHighlighting = (element) => {
        if (typeof hljs !== 'undefined') {
            element.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightBlock(block);
            });
        }
    };

    const renderMarkdown = (element, markdown) => {
        if (!markdown || !markdown.trim()) {
            element.textContent = 'Sorry, I couldn\'t generate a response.';
            return;
        }
        if (typeof marked !== 'undefined') {
            try {
                element.innerHTML = marked.parse(markdown);
            } catch (e) {
                console.error('Markdown parsing error:', e);
                element.textContent = markdown;
            }
        } else {
            element.textContent = markdown;
        }
    };

    // Initialize messages
    document.querySelectorAll('.message.assistant .message-content').forEach(element => {
        renderMarkdown(element, element.textContent || '');
        applySyntaxHighlighting(element);
    });
    scrollToBottom();

    // --- File Upload AJAX Handling ---
    if (fileUploadInput) {
        fileUploadInput.addEventListener('change', () => {
            if (fileUploadInput.files.length > 0) {
                debug(`File selected: ${fileUploadInput.files[0].name}. Submitting form.`);
                // Automatically submit the form when a file is chosen.
                promptForm.dispatchEvent(new SubmitEvent('submit', { bubbles: true, cancelable: true }));
            }
        });
    }

    // --- Voice Recording and Upload Logic ---
    let audioContext, mediaStream, sourceNode, processorNode;
    let recordedBuffers = [];
    let recording = false;
    const TARGET_SAMPLE_RATE = 16000; // The sample rate the server expects.

    // Helper function to convert float audio data to 16-bit PCM format
    function floatTo16BitPCM(float32Array) {
        const l = float32Array.length;
        const buffer = new ArrayBuffer(l * 2);
        const view = new DataView(buffer);
        for (let i = 0; i < l; i++) {
            const s = Math.max(-1, Math.min(1, float32Array[i]));
            view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
        return view;
    }

    // Helper function to write a string to a DataView
    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    // Main function to encode raw audio data into a WAV file blob
    function encodeWAV(float32Array, sampleRate) {
        const pcmView = floatTo16BitPCM(float32Array);
        const buffer = new ArrayBuffer(44 + pcmView.byteLength);
        const view = new DataView(buffer);

        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + pcmView.byteLength, true);
        writeString(view, 8, 'WAVE');
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true); // PCM
        view.setUint16(22, 1, true); // Mono
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true); // Byte rate
        view.setUint16(32, 2, true); // Block align
        view.setUint16(34, 16, true); // 16-bit
        writeString(view, 36, 'data');
        view.setUint32(40, pcmView.byteLength, true);

        // Copy PCM data after the header
        for (let i = 0; i < pcmView.byteLength; i++) {
            view.setUint8(44 + i, pcmView.getUint8(i));
        }

        return new Blob([view], { type: 'audio/wav' });
    }

    async function startRecording() {
        debug("Attempting to start recording...");
        recordedBuffers = [];
        audioPreview.hidden = true;

        try {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (err) {
            console.error("Microphone access denied:", err);
            promptInput.placeholder = "Microphone access is required.";
            return;
        }

        sourceNode = audioContext.createMediaStreamSource(mediaStream);
        const bufferSize = 4096;
        processorNode = audioContext.createScriptProcessor(bufferSize, 1, 1);

        processorNode.onaudioprocess = (e) => {
            recordedBuffers.push(new Float32Array(e.inputBuffer.getChannelData(0)));
        };

        sourceNode.connect(processorNode);
        processorNode.connect(audioContext.destination);

        recording = true;
        micBtn.classList.add("listening");
        micBtn.title = "Stop recording";
        promptInput.placeholder = "Recording...";
        debug("Recording started.");
    }

    async function stopRecordingAndUpload() {
        debug("Stopping recording and preparing upload...");
        if (!recording) return;

        // Stop audio processing
        try {
            processorNode.disconnect();
            sourceNode.disconnect();
            mediaStream.getTracks().forEach(t => t.stop());
            await audioContext.close();
        } catch (err) {
            console.warn("Error stopping audio nodes:", err);
        }

        recording = false;
        micBtn.classList.remove("listening");
        micBtn.title = "Use microphone";
        promptInput.placeholder = "Transcribing...";

        // Merge and downsample audio buffers
        const totalLength = recordedBuffers.reduce((sum, b) => sum + b.length, 0);
        const merged = new Float32Array(totalLength);
        let offset = 0;
        for (const buffer of recordedBuffers) {
            merged.set(buffer, offset);
            offset += buffer.length;
        }

        // Create WAV blob
        const wavBlob = encodeWAV(merged, audioContext.sampleRate);
        debug("WAV blob created", { size: wavBlob.size });

        // Show audio preview
        audioPreview.src = URL.createObjectURL(wavBlob);
        audioPreview.hidden = false;

        // Upload for transcription
        const formData = new FormData();
        formData.append("audio", wavBlob, "recording.wav");

        try {
            const csrfToken = promptForm.querySelector('input[name="csrfmiddlewaretoken"]').value;
            const response = await fetch("/speech-to-text/", {
                method: "POST",
                headers: { "X-CSRFToken": csrfToken },
                body: formData
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || "Transcription failed.");
            }

            debug("Transcription successful:", data.text);
            promptInput.value = data.text;
            promptInput.placeholder = "Ask anything...";
            audioPreview.hidden = true; // Hide preview on success

            // Automatically submit the form with the transcribed text
            if (data.text.trim()) {
                promptForm.dispatchEvent(new SubmitEvent('submit', { bubbles: true, cancelable: true }));
            }

        } catch (err) {
            console.error("Upload/transcription error:", err);
            promptInput.placeholder = err.message;
        }
    }

    if (micBtn) {
        micBtn.addEventListener("click", () => {
            if (recording) {
                stopRecordingAndUpload();
            } else {
                startRecording();
            }
        });
    } else {
        debug("Microphone button not found.");
    }

    // Form submission handler
    promptForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const file = fileUploadInput ? fileUploadInput.files[0] : undefined;
        const promptText = promptInput.value.trim();

        // Do not submit if there is no text and no file.
        if (!promptText && !file) {
            debug("Submission cancelled: No prompt text or file.");
            return;
        }

        const welcomeMessage = document.querySelector('.welcome-message');
        if (welcomeMessage) welcomeMessage.remove();

        // Display user message (prompt text or file name)
        const userMessage = document.createElement('div');
        userMessage.className = 'message user';
        const userContent = document.createElement('div');
        userContent.className = 'message-content';
        userContent.textContent = file ? `📎 Uploaded: ${file.name}` : promptText;
        userMessage.appendChild(userContent);
        messageArea.appendChild(userMessage);
        scrollToBottom();

        // Prepare assistant's response
        const assistantMessage = document.createElement('div');
        assistantMessage.className = 'message assistant';
        const assistantContent = document.createElement('div');
        assistantContent.className = 'message-content';
        assistantContent.innerHTML = '<span class="thinking-indicator"></span>';
        assistantMessage.appendChild(assistantContent);
        messageArea.appendChild(assistantMessage);
        scrollToBottom();

        // Prepare form data
        const formData = new FormData(promptForm);
        promptInput.value = '';
        if (fileUploadInput) fileUploadInput.value = ''; // Clear file input after getting it into formData
        promptInput.disabled = true;
        sendButton.disabled = true;
        if (micBtn) micBtn.disabled = true;

        let rawResponse = '';
        try {
            const response = await fetch(promptForm.action, {
                method: 'POST',
                headers: {
                    'X-CSRFToken': promptForm.querySelector('input[name="csrfmiddlewaretoken"]').value,
                },
                body: formData
            });

            if (!response.ok) {
                // Try to parse a JSON error message from the server first
                try {
                    const errData = await response.json();
                    throw new Error(errData.error || `Server error: ${response.status}`);
                } catch (e) {
                    // If the error response wasn't JSON, use the status text
                    throw new Error(`Server error: ${response.status} ${response.statusText}`);
                }
            }

            const contentType = response.headers.get("content-type");

            if (contentType && contentType.indexOf("application/json") !== -1) {
                // Handle JSON response from file upload
                const data = await response.json();
                renderMarkdown(assistantContent, data.system_message);

                if (data.new_session_id) {
                    updateChatSessionUI(data.new_session_id, data.new_session_title, promptForm);
                }
            } else {
                // Handle streaming response from text prompt
                const newSessionId = response.headers.get('X-Chat-Session-Id');
                const newSessionTitle = response.headers.get('X-Chat-Session-Title');

                if (newSessionId) {
                    updateChatSessionUI(newSessionId, newSessionTitle, promptForm);
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                assistantContent.innerHTML = '';

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    rawResponse += decoder.decode(value, { stream: true });
                    renderMarkdown(assistantContent, rawResponse);
                    scrollToBottom();
                }
            }

        } catch (error) {
            console.error('Error in form submission:', error);
            assistantContent.innerHTML = `<div class="error-message">Error: ${error.message || 'An error occurred. Please try again.'}</div>`;
        } finally {
            applySyntaxHighlighting(assistantContent);
            // Only show fallback if the response wasn't a file upload and the stream was empty
            if (rawResponse === '' && !assistantContent.textContent.trim() && !assistantContent.querySelector('.error-message')) {
                renderMarkdown(assistantContent, '');
            }
            promptInput.disabled = false;
            sendButton.disabled = false;
            if (micBtn) micBtn.disabled = false;
            promptInput.focus();
            scrollToBottom();
        }
    });

    function updateChatSessionUI(sessionId, title, form) {
        const chatUrlTemplate = form.dataset.chatUrlTemplate;
        const newUrl = chatUrlTemplate.replace('99999999', sessionId);
        window.history.pushState({path: newUrl}, '', newUrl);
        form.action = newUrl;

        const chatList = document.querySelector('.chat-list');
        const noChatsMessage = chatList.querySelector('li[style*="color: #888"]');
        if (noChatsMessage) noChatsMessage.remove();

        const currentActive = chatList.querySelector('.chat-list-item.active');
        if (currentActive) currentActive.classList.remove('active');

        const newChatItem = document.createElement('li');
        newChatItem.className = 'chat-list-item active';
        const deleteUrlTemplate = form.dataset.deleteUrlTemplate;
        const deleteUrl = deleteUrlTemplate.replace('99999999', sessionId);
        const csrfToken = form.querySelector('input[name="csrfmiddlewaretoken"]').value;

        newChatItem.innerHTML = `
            <a href="${newUrl}">${title || 'New Chat'}</a>
            <form action="${deleteUrl}" method="post" class="delete-chat-form">
                <input type="hidden" name="csrfmiddlewaretoken" value="${csrfToken}">
                <button type="submit" title="Delete chat" onclick="return confirm('Are you sure you want to delete this chat?');">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="3 6 5 6 21 6"></polyline>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        <line x1="10" y1="11" x2="10" y2="17"></line>
                        <line x1="14" y1="11" x2="14" y2="17"></line>
                    </svg>
                </button>
            </form>`;
        chatList.prepend(newChatItem);
    }
});