document.addEventListener('DOMContentLoaded', function() {
    let sessionId = null;
    let progressChart = null;
    
    // DOM Elements
    const chatForm = document.getElementById('chatForm');
    const messageInput = document.getElementById('messageInput');
    const chatContainer = document.getElementById('chatContainer');
    const profileForm = document.getElementById('profileForm');
    const sessionIdElement = document.getElementById('sessionId');
    const summaryTextElement = document.getElementById('summaryText');
    const commandButtons = document.querySelectorAll('.command-btn');
    
    // Initialize
    startNewSession();
    
    // Event Listeners
    chatForm.addEventListener('submit', handleChatSubmit);
    profileForm.addEventListener('submit', handleProfileSubmit);
    commandButtons.forEach(btn => btn.addEventListener('click', handleCommandClick));
    
    // Functions
    function startNewSession() {
        sessionId = generateSessionId();
        sessionIdElement.textContent = `Oturum: ${sessionId.slice(0, 8)}`;
        chatContainer.innerHTML = `
            <div class="welcome-message text-center mt-5">
                <h4>SFBT Terapi Botuna Hoş Geldiniz</h4>
                <p class="text-muted">Nasıl yardımcı olabilirim?</p>
            </div>
        `;
    }
    
    async function handleChatSubmit(e) {
        e.preventDefault();
        const message = messageInput.value.trim();
        if (!message) return;
        
        addMessageToChat('user', message);
        messageInput.value = '';
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message, session_id: sessionId })
            });
            
            if (!response.ok) throw new Error(await response.text());
            
            const data = await response.json();
            addMessageToChat('bot', data.response);
            
            if (message.toLowerCase() === 'özet') {
                updateSummaryText(data.response);
            }
        } catch (error) {
            console.error('Error:', error);
            addMessageToChat('bot', 'Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin.');
        }
    }
    
    async function handleProfileSubmit(e) {
        e.preventDefault();
        const formData = new FormData(profileForm);
        const profile = {
            name: formData.get('name'),
            age: formData.get('age'),
            primary_concerns: Array.from(document.querySelectorAll('#profileForm input[type="checkbox"]:checked')).map(c => c.value)
        };
        
        try {
            await fetch(`/api/profile/${sessionId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(profile)
            });
        } catch (error) {
            console.error('Profile error:', error);
        }
    }
    
    function handleCommandClick() {
        messageInput.value = this.getAttribute('data-command');
        chatForm.dispatchEvent(new Event('submit'));
    }
    
    function addMessageToChat(sender, message) {
        const messageElement = document.createElement('div');
        messageElement.className = `chat-message ${sender}-message`;
        messageElement.innerHTML = `
            <div>${message.replace(/\n/g, '<br>')}</div>
            <span class="message-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        `;
        
        const welcome = chatContainer.querySelector('.welcome-message');
        if (welcome) welcome.remove();
        
        chatContainer.appendChild(messageElement);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    function updateSummaryText(text) {
        if (summaryTextElement) {
            summaryTextElement.innerHTML = text.replace(/\n/g, '<br>');
        }
    }
    
    function generateSessionId() {
        return 'session-' + Math.random().toString(36).substr(2, 9);
    }
});