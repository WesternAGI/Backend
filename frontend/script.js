// Basic placeholder for script.js
// We will add functionality for registration, login, and chat here.

document.addEventListener('DOMContentLoaded', () => {
    console.log('Frontend JavaScript Loaded');

    const registerForm = document.getElementById('register-form');
    const loginForm = document.getElementById('login-form');
    const registerMessage = document.getElementById('register-message');
    const loginMessage = document.getElementById('login-message');

    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const chatHistory = document.getElementById('chat-history');
    const logoutButton = document.getElementById('logout-button');

    const API_BASE_URL = 'http://localhost:8000'; // Adjust if your backend runs on a different port

    // Handle Registration
    if (registerForm) {
        registerForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            const username = document.getElementById('reg-username').value; // Use reg-username for username
            // const email = document.getElementById('reg-email').value; // Email is not in RegisterRequest schema
            const password = document.getElementById('reg-password').value;

            try {
                const response = await fetch(`${API_BASE_URL}/register`, { // Updated endpoint
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ username, password }), // Sending username and password
                });
                const data = await response.json();
                if (response.ok) {
                    registerMessage.textContent = 'Registration successful! You can now login.';
                    registerMessage.style.color = 'green';
                    registerForm.reset();
                } else {
                    registerMessage.textContent = data.detail || 'Registration failed.';
                    registerMessage.style.color = 'red';
                }
            } catch (error) {
                console.error('Registration error:', error);
                registerMessage.textContent = 'An error occurred. Please try again.';
                registerMessage.style.color = 'red';
            }
        });
    }

    // Handle Login
    if (loginForm) {
        loginForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            const email = document.getElementById('login-email').value; // This will be sent as 'username' for OAuth2
            const password = document.getElementById('login-password').value;

            const formData = new URLSearchParams();
            formData.append('username', email); // FastAPI's OAuth2PasswordRequestForm expects 'username'
            formData.append('password', password);

            try {
                const response = await fetch(`${API_BASE_URL}/login`, { // Updated endpoint
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: formData,
                });
                const data = await response.json();
                if (response.ok) {
                    loginMessage.textContent = 'Login successful!';
                    loginMessage.style.color = 'green';
                    localStorage.setItem('accessToken', data.access_token);
                    // Redirect to chat page
                    window.location.href = 'chat.html';
                } else {
                    loginMessage.textContent = data.detail || 'Login failed. Check credentials.';
                    loginMessage.style.color = 'red';
                }
            } catch (error) {
                console.error('Login error:', error);
                loginMessage.textContent = 'An error occurred. Please try again.';
                loginMessage.style.color = 'red';
            }
        });
    }

    // Check if on chat page
    if (window.location.pathname.endsWith('chat.html')) {
        const accessToken = localStorage.getItem('accessToken');
        if (!accessToken) {
            window.location.href = 'index.html'; // Redirect to login if no token
            return;
        }

        // Function to add a message to chat history
        function addMessageToChat(message, sender) {
            if (!chatHistory) return;
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message', sender);
            messageDiv.textContent = message;
            chatHistory.appendChild(messageDiv);
            chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll to bottom
        }

        // Load initial user info / greeting
        async function fetchUserInfo() {
            try {
                const response = await fetch(`${API_BASE_URL}/profile`, { // Updated endpoint
                    headers: {
                        'Authorization': `Bearer ${accessToken}`
                    }
                });
                if (response.ok) {
                    const user = await response.json();
                    addMessageToChat(`Welcome, ${user.username}!`, 'bot');
                    // Chat history loading removed as endpoint is not available
                } else {
                    console.error('Failed to fetch user info');
                    addMessageToChat('Could not load your user information. Please try logging in again.', 'bot');
                    if (response.status === 401) { // Unauthorized
                        localStorage.removeItem('accessToken');
                        window.location.href = 'index.html';
                    }
                }
            } catch (error) {
                console.error('Error fetching user info:', error);
                addMessageToChat('Error connecting to the server.', 'bot');
            }
        }
        
        fetchUserInfo();

        // Chat history functionality removed as backend endpoint is not available.
        // Original fetchConversationHistory function is deleted.

        // Handle Send Message
        if (sendButton && messageInput) {
            sendButton.addEventListener('click', sendMessage);
            messageInput.addEventListener('keypress', (event) => {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            });

            async function sendMessage() {
                const messageText = messageInput.value.trim();
                if (!messageText) return;

                addMessageToChat(messageText, 'user');
                messageInput.value = '';

                try {
                    const response = await fetch(`${API_BASE_URL}/query`, { // Updated endpoint
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${accessToken}`,
                        },
                        body: JSON.stringify({ query: messageText }), // Backend expects 'query'
                    });
                    const data = await response.json();
                    if (response.ok) {
                        // Assuming the backend response for /query has a 'response' field with the AI's message
                        addMessageToChat(data.response, 'bot'); 
                    } else {
                        addMessageToChat(data.detail || 'Error sending message.', 'bot');
                    }
                } catch (error) {
                    console.error('Send message error:', error);
                    addMessageToChat('Could not connect to the chat server.', 'bot');
                }
            }
        }

        // Handle Logout
        if (logoutButton) {
            logoutButton.addEventListener('click', () => {
                localStorage.removeItem('accessToken');
                window.location.href = 'index.html';
            });
        }
    }
});
