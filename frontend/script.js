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

    const API_BASE_URL = ''; // API calls will use relative paths

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
            loginMessage.textContent = ''; // Clear previous messages
            const email = document.getElementById('login-email').value; // This will be sent as 'username' for OAuth2
            const password = document.getElementById('login-password').value;

            const formData = new URLSearchParams();
            formData.append('username', email);
            formData.append('password', password);

            try {
                // Step 1: Get the access token
                const tokenResponse = await fetch(`${API_BASE_URL}/login`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: formData,
                });

                const tokenData = await tokenResponse.json();

                if (!tokenResponse.ok) {
                    loginMessage.textContent = tokenData.detail || 'Login failed. Check credentials.';
                    loginMessage.style.color = 'red';
                    return; // Stop if token request failed
                }

                const accessToken = tokenData.access_token;
                localStorage.setItem('accessToken', accessToken);

                // Step 2: Fetch user profile to check role
                try {
                    const profileResponse = await fetch(`${API_BASE_URL}/profile`, {
                        headers: {
                            'Authorization': `Bearer ${accessToken}`
                        }
                    });

                    if (!profileResponse.ok) {
                         // Handle profile fetch error (e.g., token became invalid quickly?)
                        localStorage.removeItem('accessToken'); // Clean up token
                        loginMessage.textContent = 'Login succeeded, but failed to fetch profile. Please try again.';
                        loginMessage.style.color = 'red';
                        return;
                    }

                    const user = await profileResponse.json();

                    // Step 3: Redirect based on role
                    if (user.role === 0) { // Assuming 0 is the student role
                        loginMessage.textContent = 'Login successful! Redirecting...';
                        loginMessage.style.color = 'green';
                        window.location.href = 'chat.html'; // Redirect students to chat
                    } else { // Admin or other roles
                        loginMessage.textContent = 'Admin login successful. Dashboard not available yet.';
                        loginMessage.style.color = 'blue';
                        // Stay on the login page for admins for now
                    }

                } catch (profileError) {
                    console.error('Profile fetch error:', profileError);
                    localStorage.removeItem('accessToken'); // Clean up token
                    loginMessage.textContent = 'Error fetching user profile after login.';
                    loginMessage.style.color = 'red';
                }

            } catch (loginError) {
                console.error('Login error:', loginError);
                loginMessage.textContent = 'An error occurred during login. Please try again.';
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
