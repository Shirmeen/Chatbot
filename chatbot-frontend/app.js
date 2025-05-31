import React, { useState } from "react";
import axios from "axios";

function App() {
  const [mode, setMode] = useState("text");
  const [input, setInput] = useState("");
  const [response, setResponse] = useState("");
  const [recording, setRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [audioChunks, setAudioChunks] = useState([]);

  // Handle text submit
  const handleTextSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    setResponse("...");
    try {
      const res = await axios.post("http://localhost:5000/chat", { message: input });
      setResponse(res.data.response);
    } catch (err) {
      setResponse("Error: " + err.message);
    }
  };

  // Handle audio recording
  const startRecording = async () => {
    setResponse("");
    setAudioChunks([]);
    setRecording(true);
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new window.MediaRecorder(stream);
    setMediaRecorder(recorder);
    recorder.start();

    recorder.ondataavailable = (e) => {
      setAudioChunks((prev) => [...prev, e.data]);
    };

    recorder.onstop = async () => {
      setRecording(false);
      const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
      const formData = new FormData();
      formData.append("audio", audioBlob, "audio.wav");
      setResponse("Processing...");
      // Send audio to backend (optional: implement /chat/audio in Flask)
      try {
        const res = await axios.post("http://localhost:5000/chat/audio", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        setResponse(res.data.response);
      } catch (err) {
        setResponse("Error: " + err.message);
      }
    };
  };

  const stopRecording = () => {
    if (mediaRecorder) {
      mediaRecorder.stop();
    }
  };

  return (
    <div style={{ maxWidth: 500, margin: "40px auto", padding: 20, border: "1px solid #ddd", borderRadius: 8 }}>
      <h2>Shirmeen Chatbot</h2>
      <div style={{ marginBottom: 20 }}>
        <button onClick={() => setMode("text")} disabled={mode === "text"}>Type</button>
        <button onClick={() => setMode("audio")} disabled={mode === "audio"} style={{ marginLeft: 10 }}>Speak</button>
      </div>
      {mode === "text" ? (
        <form onSubmit={handleTextSubmit}>
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Type your message..."
            style={{ width: "80%", padding: 8 }}
          />
          <button type="submit" style={{ marginLeft: 10 }}>Send</button>
        </form>
      ) : (
        <div>
          <button onClick={recording ? stopRecording : startRecording}>
            {recording ? "Stop Recording" : "Start Recording"}
          </button>
        </div>
      )}
      <div style={{ marginTop: 30, minHeight: 40 }}>
        <strong>Assistant:</strong>
        <div>{response}</div>
      </div>
    </div>
  );
}

export default App;