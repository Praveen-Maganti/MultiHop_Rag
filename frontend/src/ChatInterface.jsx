import React, { useState } from 'react';

const ChatInterface = ({ onAsk }) => {
  const [question, setQuestion] = useState('');
  const [groundTruth, setGroundTruth] = useState('');
  const [mode, setMode] = useState('socratic');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (question.trim()) {
      onAsk(question, groundTruth.trim() || null, mode);
    }
  };

  return (
    <div className="glass-panel">
      <h2 className="panel-title">Ask a Question (State-Aware RAG)</h2>
      <form onSubmit={handleSubmit} className="chat-input-container" style={{ flexDirection: 'column', gap: '0.75rem' }}>
        <input
          type="text"
          className="chat-input"
          placeholder="e.g., What company is Praveen working for and what is their domain?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', flexWrap: 'wrap' }}>
          <input
            type="text"
            className="chat-input"
            placeholder="Expected answer (ground truth) — for evaluation (optional)"
            value={groundTruth}
            onChange={(e) => setGroundTruth(e.target.value)}
            style={{ flex: 1, minWidth: '220px' }}
          />

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Algorithm Mode:</label>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value)}
              style={{
                background: 'rgba(22, 27, 34, 0.9)',
                color: 'var(--text-primary)',
                border: '1px solid rgba(88, 166, 255, 0.3)',
                borderRadius: '6px',
                padding: '0.45rem 0.75rem',
                fontSize: '0.82rem',
                cursor: 'pointer'
              }}
            >
              <option value="socratic">State-Aware RAG (Socratic Planning)</option>
              <option value="mcts">State-Aware RAG (MCTS Tree Search)</option>
              <option value="proposed">Evidence-Aware Memory RAG</option>
            </select>
          </div>

          <button type="submit" className="btn-primary" disabled={!question.trim()}>
            Run Query
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatInterface;
