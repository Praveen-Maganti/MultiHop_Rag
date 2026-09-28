import React, { useState } from 'react';

const ChatInterface = ({ onAsk }) => {
  const [question, setQuestion] = useState('');
  const [groundTruth, setGroundTruth] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (question.trim()) {
      onAsk(question, groundTruth.trim() || null);
    }
  };

  return (
    <div className="glass-panel">
      <h2 className="panel-title">Ask a Question</h2>
      <form onSubmit={handleSubmit} className="chat-input-container" style={{ flexDirection: 'column', gap: '0.75rem' }}>
        <input
          type="text"
          className="chat-input"
          placeholder="e.g., What is Praveen's CGPA?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <input
            type="text"
            className="chat-input"
            placeholder="Expected answer (ground truth) — for F1 evaluation, e.g., Praveen"
            value={groundTruth}
            onChange={(e) => setGroundTruth(e.target.value)}
            style={{ flex: 1 }}
          />
          <button type="submit" className="btn-primary" disabled={!question.trim()}>
            Ask
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatInterface;
