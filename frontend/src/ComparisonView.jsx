import React from 'react';

const ComparisonView = ({ results }) => {
  if (!results) return null;

  return (
    <div className="glass-panel">
      <h2 className="panel-title">Analysis Results</h2>
      
      <div className="comparison-grid">
        {/* Standard RAG Side */}
        <div>
          <h3 style={{ textAlign: 'center', marginBottom: '1.5rem', color: '#8b949e' }}>STANDARD RAG</h3>
          
          <div className="result-section">
            <h3>Answer</h3>
            <div className="result-box">
              {results.standard.answer}
            </div>
          </div>

          <div className="result-section">
            <h3>Retrieved Context</h3>
            <div className="result-box">
              {results.standard.evidence.map((ev, i) => (
                <div key={i} style={{ marginBottom: '1rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.5rem' }}>
                  <strong>{ev.id}</strong>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>{ev.text}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="result-section">
            <h3>Metrics</h3>
            <table className="metrics-table">
              <tbody>
                <tr>
                  <td>Retrieval Calls</td>
                  <td>{results.standard.metrics.retrieval_calls}</td>
                </tr>
                <tr>
                  <td>Tokens Used</td>
                  <td>{results.standard.metrics.estimated_tokens || results.standard.metrics.tokens || 0}</td>
                </tr>
                <tr>
                  <td>Latency</td>
                  <td>{results.standard.metrics.latency_seconds || results.standard.metrics.latency || 0}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Proposed RAG Side */}
        <div>
          <h3 style={{ textAlign: 'center', marginBottom: '1.5rem', color: 'var(--accent-color)' }}>PROPOSED RAG</h3>
          
          <div className="result-section">
            <h3>Answer</h3>
            <div className="result-box">
              {results.proposed.answer}
            </div>
          </div>

          <div className="result-section">
            <h3>Working Memory</h3>
            <div className="result-box">
              {results.proposed.memory.map((mem, i) => (
                <div key={i} style={{ marginBottom: '1rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.5rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <strong>{mem.claim}</strong>
                    {mem.conflict_score > 0.5 && <span className="badge danger">Conflict</span>}
                  </div>
                  <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
                    <span className="badge">Rel: {mem.relevance_score || mem.relevance || 0}</span>
                    <span className="badge">Sup: {mem.evidence_score || mem.support || 0}</span>
                    <span className="badge success">Fut: {mem.utility_score || mem.future_utility || 0}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {results.proposed.hop_trace && results.proposed.hop_trace.length > 0 && (
            <div className="result-section">
              <h3>Reasoning Hops</h3>
              <div className="result-box">
                {results.proposed.hop_trace.map((hop, i) => (
                  <div key={i} style={{ marginBottom: '1rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.75rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
                      <span className="badge success" style={{ fontSize: '0.8rem' }}>Hop {hop.hop}</span>
                      <span className="badge" style={{ fontSize: '0.75rem', background: hop.action === 'ANSWER' ? 'rgba(63,185,80,0.2)' : 'rgba(88,166,255,0.2)' }}>
                        {hop.action}
                      </span>
                    </div>
                    <div style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', marginBottom: '0.3rem' }}>
                      <strong>Query:</strong> {hop.query}
                    </div>
                    {hop.evidence && hop.evidence.length > 0 && (
                      <div style={{ color: '#8b949e', fontSize: '0.78rem' }}>
                        <strong>Evidence:</strong> {hop.evidence[0]}
                      </div>
                    )}
                    {hop.next_query && (
                      <div style={{ color: 'var(--accent-color)', fontSize: '0.78rem', marginTop: '0.3rem' }}>
                        ↓ Next: {hop.next_query}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="result-section">
            <h3>Metrics</h3>
            <table className="metrics-table">
              <tbody>
                <tr>
                  <td>Retrieval Calls</td>
                  <td>{results.proposed.metrics.retrieval_calls}</td>
                </tr>
                <tr>
                  <td>Tokens Used</td>
                  <td>{results.proposed.metrics.estimated_tokens || results.proposed.metrics.tokens || 0}</td>
                </tr>
                <tr>
                  <td>Latency</td>
                  <td>{results.proposed.metrics.latency_seconds || results.proposed.metrics.latency || 0}</td>
                </tr>
                <tr>
                  <td>Memory Items</td>
                  <td>{results.proposed.metrics.memory_items}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ComparisonView;
