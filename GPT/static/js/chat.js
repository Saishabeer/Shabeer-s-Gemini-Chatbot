document.addEventListener('DOMContentLoaded', (event) => {
    const messageArea = document.querySelector('.message-area');
    const promptForm = document.querySelector('.prompt-form');
    const promptInput = promptForm.querySelector('input[name="prompt"]');
    const sendButton = promptForm.querySelector('button[type="submit"]');

    // --- Theme Toggling Logic ---
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        const htmlElement = document.documentElement;
        const applyTheme = (theme) => {
            htmlElement.classList.toggle('dark-theme', theme === 'dark');
        };

        const savedTheme = localStorage.getItem('theme');
        const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        const currentTheme = savedTheme || (prefersDark ? 'dark' : 'light');

        applyTheme(currentTheme);

        themeToggle.addEventListener('click', () => {
            const newTheme = htmlElement.classList.contains('dark-theme') ? 'light' : 'dark';
            applyTheme(newTheme);
            localStorage.setItem('theme', newTheme);
        });
    }

    // --- Helper Functions ---
    const scrollToBottom = () => {
        if (messageArea) {
            messageArea.scrollTop = messageArea.scrollHeight;
        }
    };

    const renderContent = (element) => {
        const rawMarkdown = element.textContent || '';
        if (!rawMarkdown.trim()) {
            element.textContent = 'Sorry, I couldn\'t generate a response.';
            return;
        }
        if (typeof marked !== 'undefined') {
            element.innerHTML = marked.parse(rawMarkdown);
        } else {
            console.error("marked.js is not loaded. Cannot render markdown.");
            element.innerHTML = rawMarkdown.replace(/\n/g, '<br>');
        }
        if (typeof hljs !== 'undefined') {
            element.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightBlock(block);
            });
        }
    };

    // --- Initial Page Load ---
    document.querySelectorAll('.message.assistant .message-content').forEach(renderContent);
    scrollToBottom();

    // --- Handle Form Submission (AJAX for Streaming) ---
    promptForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const promptText = promptInput.value.trim();
        if (!promptText) return;

        // 1. Display User Message
        const welcomeMessage = document.querySelector('.welcome-message');
        if (welcomeMessage) welcomeMessage.remove();

        const userMessage = document.createElement('div');
        userMessage.className = 'message user';
        const userContent = document.createElement('div');
        userContent.className = 'message-content';
        userContent.textContent = promptText;
        userMessage.appendChild(userContent);
        messageArea.appendChild(userMessage);
        scrollToBottom();

        // 2. Prepare Assistant's Message Bubble
        const assistantMessage = document.createElement('div');
        assistantMessage.className = 'message assistant';
        const assistantContent = document.createElement('div');
        assistantContent.className = 'message-content';
        assistantContent.innerHTML = '<span class="thinking-indicator"></span>';
        assistantMessage.appendChild(assistantContent);
        messageArea.appendChild(assistantMessage);
        scrollToBottom();

        // 3. Disable Form and Clear Input
        const formData = new FormData(promptForm);
        promptInput.value = '';
        promptInput.disabled = true;
        sendButton.disabled = true;

        // 4. Fetch and Process Stream
        try {
            const response = await fetch(promptForm.action, {
                method: 'POST',
                headers: {
                    'X-CSRFToken': promptForm.querySelector('input[name="csrfmiddlewaretoken"]').value,
                },
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }

            const newSessionId = response.headers.get('X-Chat-Session-Id');
            const newSessionTitle = response.headers.get('X-Chat-Session-Title');

            if (newSessionId) {
                const chatUrlTemplate = promptForm.dataset.chatUrlTemplate;
                const newUrl = chatUrlTemplate.replace('99999999', newSessionId);
                window.history.pushState({path: newUrl}, '', newUrl);
                promptForm.action = newUrl;

                const chatList = document.querySelector('.chat-list');
                const noChatsMessage = chatList.querySelector('li[style*="color: #888"]');
                if (noChatsMessage) noChatsMessage.remove();

                const currentActive = chatList.querySelector('.chat-list-item.active');
                if (currentActive) currentActive.classList.remove('active');

                const newChatItem = document.createElement('li');
                newChatItem.className = 'chat-list-item active';
                const deleteUrlTemplate = promptForm.dataset.deleteUrlTemplate;
                const deleteUrl = deleteUrlTemplate.replace('99999999', newSessionId);
                const csrfToken = promptForm.querySelector('input[name="csrfmiddlewaretoken"]').value;
                newChatItem.innerHTML = `
                    <a href="${newUrl}">${newSessionTitle || 'New Chat'}</a>
                    <form action="${deleteUrl}" method="post" class="delete-chat-form">
                        <input type="hidden" name="csrfmiddlewaretoken" value="${csrfToken}">
                        <button type="submit" title="Delete chat" onclick="return confirm('Are you sure you want to delete this chat?');"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg></button>
                    </form>`;
                chatList.prepend(newChatItem);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            assistantContent.innerHTML = ''; // Clear thinking indicator

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                assistantContent.textContent += decoder.decode(value, { stream: true });
                scrollToBottom();
            }

        } catch (error) {
            console.error('Streaming or request failed:', error);
            assistantContent.textContent = 'Sorry, an error occurred while getting the response.';
        } finally {
            renderContent(assistantContent);
            promptInput.disabled = false;
            sendButton.disabled = false;
            promptInput.focus();
            scrollToBottom();
        }
    });
});