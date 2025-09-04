# Shabeer's Conversational AI Chatbot 🤖✨

This project is a sophisticated, full-stack conversational AI chatbot built with Python and Django. It leverages the power of Google's Gemini models to provide intelligent, context-aware, and real-time streaming responses. The application features a robust Retrieval-Augmented Generation (RAG) pipeline, allowing users to upload documents and ask questions based on their content.

## 🚀 Key Features

-   *🧠 Intelligent Chat*: Powered by Google's Gemini for high-quality conversational abilities.
-   *📄 Document Q&A (RAG)*: Upload your PDF and TXT files to create a personalized, user-specific knowledge base. The AI can answer questions based on the content of any document you've ever uploaded, across all chat sessions.
-   *🌐 Live Web Search*: For questions about recent events or topics not in its internal knowledge, the chatbot can perform a live web search using DuckDuckGo.
-   *⚡ Real-Time Streaming*: Responses are streamed word-by-word for a dynamic and engaging user experience.
-   *🔑 Resilient API Key Management*: Automatically rotates through a list of Gemini API keys to handle rate limits and ensure high availability.
-   *👤 User Authentication*: Secure user registration and login system.
-   *💬 Conversational Memory*: The AI is designed to remember context within a conversation and across sessions for a personalized feel.
-   *🔌 REST API*: A set of API endpoints for programmatic interaction with the chatbot.

## 🛠 Tech Stack

-   *Backend*: Python, Django
-   *AI & LLM*: Google Gemini Pro
-   *Vector Database*: ChromaDB for RAG
-   *Document Processing*: LangChain, PyPDF, Unstructured
-   *Web Search*: DuckDuckGo Search
-   *Frontend*: HTML, CSS, Vanilla JavaScript
-   *Database*: SQLite (default for Django)

## ⚙ Setup and Installation

Follow these steps to get the project running on your local machine.

### 1. Prerequisites

-   Python 3.11+
-   Git

### 2. Clone the Repository

### 3. Create and Activate Virtual Environment

### 4. Install Dependencies

### 5. Configure Environment Variables

Create a file named .env in the root directory of the project. Add your Google Gemini API keys to this file. The application supports key rotation, so you can add multiple keys separated by commas.

### 6. Run Database Migrations

This will set up the necessary database tables for users, chat sessions, and messages.

💡 How to Use1.Navigate to

1. http://127.0.0.1:8000/
2. Register a new account or Login with your existing credentials
3. Click "✨ New Chat" to start a conversation.
4. Use the paperclip icon (📎) to upload a document (.pdf, .txt, etc.).
5. The AI will now have knowledge of this document for all your future chats.5.Type your questions in the prompt box and get real-time answers!

 📡 API Endpoints
 
 The application provides a REST API for programmatic access.
 Authentication is required for all endpoints.•POST /api/upload/: 
 Upload a document.•POST /api/chat/<session_id>/query/: 
 Send a prompt to a specific chat session.•GET /api/chat/<session_id>/history/:
 Retrieve the message history for a session.
 DELETE /api/chat/<session_id>/delete/: Delete a chat session.
 
 
 Happy coding! 🚀