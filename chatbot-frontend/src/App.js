import './App.css';
import React, { useState } from 'react';

function App() {
  const [input, setInput] = useState('');
  const [response, setResponse] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    setResponse(`You typed: ${input}`);
    setInput('');
  };

  return (
    <div className="App" style={{ minHeight: "100vh", display: "flex", flexDirection: "column", background: "#222" }}>
      <header className="App-header" style={{ flex: 1 }}>
        <h1 style={{ fontSize: "2.5rem", marginBottom: "1rem" }}>AudiPro</h1>
        <p>
          Welcome to <b>AudiPro</b>! Choose to type or speak to interact with your assistant.
        </p>
        {response && (
          <div style={{ marginTop: "2rem", color: "#fff" }}>
            <strong>Assistant:</strong> {response}
          </div>
        )}
      </header>
      <form 
        onSubmit={handleSubmit} 
        style={{
          maxWidth: 700,
          margin: "0 auto",
          background: "#fff",
          padding: "16px 0",
          boxShadow: "0 -2px 8px rgba(0,0,0,0.08)",
          display: "flex",
          alignItems: "center",
          position: "fixed",
          left: 0,
          right: 0,
          bottom: 0,
          zIndex: 120,
        }}
      >
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Type your message..."
          style={{
            flex: 1,
            marginLeft: 16,
            marginRight: 8,
            padding: "12px 16px",
            borderRadius: 24,
            border: "1px solid #ccc",
            fontSize: "1rem",
            outline: "none",
            background: "#f7f7f8",
            color: "#222"
          }}
        />
        <button 
          type="submit"
          style={{
            marginRight: 16,
            padding: "10px 24px",
            borderRadius: 24,
            border: "none",
            background: "#10a37f",
            color: "#fff",
            fontWeight: "bold",
            fontSize: "1rem",
            cursor: "pointer"
          }}
        >
          Send
        </button>
      </form>
    </div>
  );
}

export default App;
