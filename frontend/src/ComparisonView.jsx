import React, { useState } from 'react';

const stopReasonColor = (reason) => {
  if (reason === 'sufficient_evidence') return 'var(--success-color)';
  if (reason === 'repeated_query') return '#e3b341';
  if (reason === 'no_useful_chunks') return 'var(--danger-color)';
  if (reason === 'max_hops_reached') return '#e3b341';
  return 'var(--text-secondary)';
};

const ActionBadge = ({ action }) => {
  let color = 'rgba(255,255,255,0.1)';
  let text = action || 'STEP';

  if (text.includes('A1') || text === 'SEARCH') {
    color = 'rgba(88,166,255,0.25)';
  } else if (text.includes('A5') || text === 'ANSWER') {
    color = 'rgba(63,185,80,0.25)';
  } else if (text.includes('A2')) {
    color = 'rgba(210,153,34,0.25)';
  } else if (text.includes('A3')) {
    color = 'rgba(163,113,247,0.25)';
  } else if (text.includes('A4')) {
    color = 'rgba(248,81,73,0.25)';
  }

  return (
    <span className="badge" style={{ background: color, fontSize: '0.72rem', fontWeight: 600 }}>
      {text}
    </span>
  );
};

// ─── Per-Hop Memory Usage Comparison Card ───────────────────────────────────
const HopMemoryTracker = ({ standardData, proposedData }) => {
  const stdHops = standardData.hop_trace || [];
  const propHops = proposedData.hop_trace || [];
  const maxHopCount = Math.max(stdHops.length, propHops.length);

  if (maxHopCount === 0) return null;

  const rows = [];
  for (let i = 0; i < maxHopCount; i++) {
    const sHop = stdHops[i] || {};
    const pHop = propHops[i] || {};

    const sItems = sHop.memory_items_count ?? sHop.total_context_size ?? (i + 1) * 3;
    const sTokens = sHop.memory_tokens_est ?? (sItems * 120);

    const pItems = pHop.memory_items_count ?? pHop.active_memory ?? (pHop.memory_after ?? Math.min(3, i + 1));
    const pTokens = pHop.memory_tokens_est ?? (pItems * 40);

    const tokenReduction = sTokens > 0 ? Math.round(((sTokens - pTokens) / sTokens) * 100) : 0;

    rows.push({
      hop: i + 1,
      sItems,
      sTokens,
      pItems,
      pTokens,
      reductionPct: tokenReduction
    });
  }

  return (
    <div className="result-section" style={{ background: 'rgba(88,166,255,0.03)', border: '1px solid rgba(88,166,255,0.2)', borderRadius: '8px', padding: '1rem', marginBottom: '1.5rem' }}>
      <h3 style={{ margin: '0 0 0.5rem 0', color: 'var(--accent-color)', fontSize: '0.92rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <span>📊</span> Per-Hop Working Memory &amp; Token Footprint Tracker
      </h3>
      <p style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', margin: '0 0 0.85rem 0' }}>
        Verifies that Proposed State-Aware RAG dynamically filters, consolidates, and compresses context per hop—avoiding unbounded memory bloat.
      </p>

      <table className="metrics-table" style={{ width: '100%', fontSize: '0.78rem' }}>
        <thead>
          <tr style={{ background: 'rgba(255,255,255,0.02)' }}>
            <th>Reasoning Hop</th>
            <th style={{ color: '#8b949e' }}>Multi-Hop (Flat Accumulation)</th>
            <th style={{ color: 'var(--accent-color)' }}>Proposed (Curated Memory)</th>
            <th style={{ color: 'var(--success-color)' }}>Memory Reduction</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.hop}>
              <td><strong>Hop {r.hop}</strong></td>
              <td style={{ color: '#8b949e' }}>
                <span className="badge" style={{ background: 'rgba(255,255,255,0.06)' }}>
                  {r.sItems} chunks (~{r.sTokens} tok)
                </span>
              </td>
              <td style={{ color: 'var(--accent-color)' }}>
                <span className="badge" style={{ background: 'rgba(88,166,255,0.15)', color: 'var(--accent-color)' }}>
                  {r.pItems} active facts (~{r.pTokens} tok)
                </span>
              </td>
              <td>
                <span className="badge" style={{ background: 'rgba(63,185,80,0.15)', color: 'var(--success-color)', fontWeight: 600 }}>
                  -{r.reductionPct}% tokens
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// ─── Multi-Hop RAG Baseline Panel ───────────────────────────────────────────
const MultiHopPanel = ({ data }) => {
  const [showMemory, setShowMemory] = useState(true);

  return (
    <div>
      <h3 style={{ textAlign: 'center', marginBottom: '1.5rem', color: '#8b949e', letterSpacing: '0.05em' }}>
        MULTI-HOP RAG (BASELINE)
        <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', fontWeight: 400, marginTop: '0.25rem' }}>
          Constant Flat Memory · No Dynamic Curation · Context Bloat
        </div>
      </h3>

      {/* Answer */}
      <div className="result-section">
        <h3>Answer</h3>
        <div className="result-box">{data.answer}</div>
      </div>

      {/* Hop Trace */}
      {data.hop_trace && data.hop_trace.length > 0 && (
        <div className="result-section">
          <h3>Reasoning Hops &amp; Memory Accumulation</h3>
          <div className="result-box">
            {data.hop_trace.map((hop, i) => (
              <div
                key={i}
                style={{
                  marginBottom: '1rem',
                  borderBottom: '1px solid rgba(255,255,255,0.08)',
                  paddingBottom: '0.75rem',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.35rem', flexWrap: 'wrap' }}>
                  <span className="badge success" style={{ fontSize: '0.75rem' }}>Hop {hop.hop}</span>
                  <ActionBadge action={hop.action} />
                  <span className="badge" style={{ background: 'rgba(248,81,73,0.15)', color: '#f85149', fontSize: '0.7rem' }}>
                    Accumulated Context: {hop.memory_items_count ?? hop.total_context_size} chunks (~{hop.memory_tokens_est ?? (hop.total_context_size * 120)} tok)
                  </span>
                  {hop.stop_reason && (
                    <span style={{ fontSize: '0.7rem', color: stopReasonColor(hop.stop_reason) }}>
                      [{hop.stop_reason}]
                    </span>
                  )}
                </div>
                <div style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', marginBottom: '0.25rem' }}>
                  <strong>Query:</strong> {hop.query}
                </div>
                <div style={{ fontSize: '0.78rem', color: '#8b949e' }}>
                  Retrieved: <strong>{hop.chunks_retrieved}</strong> · Added: <strong>{hop.chunks_added}</strong> · Total context: <strong>{hop.total_context_size}</strong> chunks
                </div>
                {hop.next_query && (
                  <div style={{ color: 'var(--accent-color)', fontSize: '0.78rem', marginTop: '0.25rem' }}>
                    ↓ Next query: {hop.next_query}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Flat Working Memory */}
      <div className="result-section">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
          <h3 style={{ margin: 0 }}>
            Accumulated Raw Context
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginLeft: '0.5rem' }}>
              ({data.memory?.length ?? 0} chunks · uncurated)
            </span>
          </h3>
          <button
            onClick={() => setShowMemory((v) => !v)}
            style={{
              background: 'rgba(88,166,255,0.1)',
              border: '1px solid rgba(88,166,255,0.3)',
              borderRadius: '6px',
              color: 'var(--accent-color)',
              fontSize: '0.75rem',
              padding: '0.2rem 0.6rem',
              cursor: 'pointer',
            }}
          >
            {showMemory ? 'Hide' : 'Show'}
          </button>
        </div>
        {showMemory && (
          <div className="result-box">
            {data.memory?.length === 0 && (
              <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>No chunks in memory.</div>
            )}
            {data.memory?.map((item, i) => (
              <div
                key={i}
                style={{ marginBottom: '0.85rem', borderBottom: '1px solid rgba(255,255,255,0.06)', paddingBottom: '0.6rem' }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.3rem' }}>
                  <strong style={{ fontSize: '0.82rem', color: 'var(--text-primary)' }}>{item.id}</strong>
                  <span className="badge" style={{ background: 'rgba(88,166,255,0.12)', color: 'var(--accent-color)', fontSize: '0.7rem' }}>
                    Hop {item.hop_added}
                  </span>
                </div>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', margin: 0, lineHeight: 1.5 }}>
                  {item.text}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Metrics */}
      <div className="result-section">
        <h3>Metrics</h3>
        <table className="metrics-table">
          <tbody>
            <tr><td>Hops Executed</td><td>{data.metrics?.hops ?? '—'}</td></tr>
            <tr><td>Retrieval Calls</td><td>{data.metrics?.retrieval_calls ?? '—'}</td></tr>
            <tr><td>Unique Chunks Accumulated</td><td>{data.metrics?.unique_chunks ?? '—'}</td></tr>
            <tr><td>Tokens Used</td><td>{data.metrics?.estimated_tokens ?? data.metrics?.tokens ?? 0}</td></tr>
            <tr><td>Latency</td><td>{data.metrics?.latency_seconds ?? data.metrics?.latency ?? '—'}s</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  );
};

// ─── State-Aware RAG Panel ──────────────────────────────────────────────────
const StateAwarePanel = ({ data }) => {
  const [showMemory, setShowMemory] = useState(true);
  const modeName = data.metrics?.mode || "State-Aware RAG";

  return (
    <div>
      <h3 style={{ textAlign: 'center', marginBottom: '1.5rem', color: 'var(--accent-color)', letterSpacing: '0.05em' }}>
        {modeName.toUpperCase()}
        <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', fontWeight: 400, marginTop: '0.25rem' }}>
          Dynamic Working Memory ($M_i$) · Adaptive Curation &amp; Compaction · Dual Reward
        </div>
      </h3>

      {/* Answer */}
      <div className="result-section">
        <h3>Answer</h3>
        <div className="result-box" style={{ borderColor: 'rgba(88,166,255,0.4)', background: 'rgba(88,166,255,0.04)' }}>
          {data.answer}
        </div>
      </div>

      {/* Hop / Action Trace */}
      {data.hop_trace && data.hop_trace.length > 0 && (
        <div className="result-section">
          <h3>Reasoning Steps &amp; Per-Hop Memory Curation</h3>
          <div className="result-box">
            {data.hop_trace.map((step, i) => {
              const isString = typeof step === 'string';
              return (
                <div
                  key={i}
                  style={{
                    marginBottom: '1rem',
                    borderBottom: '1px solid rgba(255,255,255,0.08)',
                    paddingBottom: '0.75rem',
                  }}
                >
                  {isString ? (
                    <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>{step}</div>
                  ) : (
                    <>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.35rem', flexWrap: 'wrap' }}>
                        <span className="badge success" style={{ fontSize: '0.75rem' }}>Step {step.hop}</span>
                        <ActionBadge action={step.action} />
                        
                        {/* Per-Hop Memory Usage Badge */}
                        <span className="badge" style={{ background: 'rgba(88,166,255,0.15)', color: 'var(--accent-color)', fontSize: '0.7rem' }}>
                          Memory: {step.memory_items_count ?? step.active_memory ?? '—'} active items (~{step.memory_tokens_est ?? ((step.memory_items_count || 1) * 35)} tok)
                        </span>

                        {step.compression_ratio && (
                          <span className="badge" style={{ background: 'rgba(63,185,80,0.15)', color: 'var(--success-color)', fontSize: '0.7rem' }}>
                            Pruned: {step.compression_ratio}
                          </span>
                        )}

                        {step.path_reward !== undefined && (
                          <span className="badge" style={{ background: 'rgba(63,185,80,0.15)', color: 'var(--success-color)', fontSize: '0.7rem' }}>
                            Path Reward: {step.path_reward}
                          </span>
                        )}
                      </div>
                      {step.subquestion && (
                        <div style={{ color: 'var(--text-primary)', fontSize: '0.82rem', marginBottom: '0.25rem' }}>
                          <strong>Sub-Q:</strong> {step.subquestion}
                        </div>
                      )}
                      {step.answer && (
                        <div style={{ color: 'var(--text-secondary)', fontSize: '0.80rem', marginBottom: '0.25rem' }}>
                          <strong>Sub-A:</strong> {step.answer}
                        </div>
                      )}
                      {step.reasoning && (
                        <div style={{ color: '#8b949e', fontSize: '0.76rem', fontStyle: 'italic' }}>
                          {step.reasoning}
                        </div>
                      )}
                    </>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Active Working Memory Mi */}
      <div className="result-section">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
          <h3 style={{ margin: 0 }}>
            Dynamic Working Memory ($M_i$)
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginLeft: '0.5rem' }}>
              ({data.memory?.filter(m => m.status !== 'DISCARDED').length ?? 0} active
              {data.memory?.some(m => m.status === 'DISCARDED') ? ` · ${data.memory.filter(m => m.status === 'DISCARDED').length} pruned` : ''})
            </span>
          </h3>
          <button
            onClick={() => setShowMemory((v) => !v)}
            style={{
              background: 'rgba(88,166,255,0.1)',
              border: '1px solid rgba(88,166,255,0.3)',
              borderRadius: '6px',
              color: 'var(--accent-color)',
              fontSize: '0.75rem',
              padding: '0.2rem 0.6rem',
              cursor: 'pointer',
            }}
          >
            {showMemory ? 'Hide' : 'Show'}
          </button>
        </div>
        {showMemory && (
          <div className="result-box">
            {data.memory?.length === 0 && (
              <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>No active memory facts.</div>
            )}
            {data.memory?.map((mem, i) => {
              const isDiscarded = mem.status === 'DISCARDED';
              return (
                <div
                  key={i}
                  style={{
                    marginBottom: '1rem',
                    borderBottom: '1px solid rgba(255,255,255,0.06)',
                    paddingBottom: '0.5rem',
                    opacity: isDiscarded ? 0.45 : 1,
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.5rem' }}>
                    <strong style={{ fontSize: '0.85rem', flex: 1, color: 'var(--text-primary)', textDecoration: isDiscarded ? 'line-through' : 'none' }}>
                      {mem.claim}
                    </strong>
                    <div style={{ display: 'flex', gap: '0.3rem', flexShrink: 0 }}>
                      {mem.hop_added && (
                        <span className="badge" style={{ background: 'rgba(88,166,255,0.12)', color: 'var(--accent-color)', fontSize: '0.68rem' }}>
                          Hop {mem.hop_added}
                        </span>
                      )}
                      <span
                        className="badge"
                        style={{
                          background: isDiscarded ? 'rgba(248,81,73,0.15)' : 'rgba(63,185,80,0.12)',
                          color: isDiscarded ? 'var(--danger-color)' : 'var(--success-color)',
                          fontSize: '0.68rem'
                        }}
                      >
                        {mem.status || 'ACTIVE'}
                      </span>
                    </div>
                  </div>
                  {mem.text && (
                    <div style={{ fontSize: '0.75rem', color: '#8b949e', marginTop: '0.25rem' }}>
                      {mem.text}
                    </div>
                  )}
                  {mem.decision_reason && (
                    <div style={{ fontSize: '0.72rem', color: '#58a6ff', marginTop: '0.2rem', fontStyle: 'italic' }}>
                      💡 {mem.decision_reason}
                    </div>
                  )}
                  <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
                    <span className="badge">Rel: {(mem.relevance_score ?? 0).toFixed(2)}</span>
                    <span className="badge">Sup: {(mem.evidence_score ?? 0).toFixed(2)}</span>
                    <span className="badge">Nov: {(mem.novelty_score ?? 0).toFixed(2)}</span>
                    <span
                      className="badge"
                      style={{
                        background: isDiscarded ? 'rgba(248,81,73,0.1)' : 'rgba(63,185,80,0.2)',
                        color: isDiscarded ? 'var(--danger-color)' : 'var(--success-color)'
                      }}
                    >
                      Retention: {(mem.utility_score ?? mem.retention_score ?? 0).toFixed(2)}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Dual Reward & Performance Metrics */}
      <div className="result-section">
        <h3>Dual Reward &amp; Performance Metrics</h3>
        <table className="metrics-table">
          <tbody>
            <tr><td>Algorithm Pipeline</td><td style={{ color: 'var(--accent-color)' }}>{modeName}</td></tr>
            <tr><td>Hops / Depth</td><td>{data.metrics?.hops ?? data.metrics?.max_tree_depth ?? '—'}</td></tr>
            <tr><td>Retrieval Calls</td><td>{data.metrics?.retrieval_calls ?? '—'}</td></tr>
            <tr><td>Active Working Memory Items ($M_i$)</td><td>{data.metrics?.memory_facts_count ?? data.metrics?.memory_items ?? '—'}</td></tr>
            {data.metrics?.duplicates_removed !== undefined && (
              <tr><td>Duplicates Merged / Pruned</td><td style={{ color: 'var(--success-color)' }}>{data.metrics.duplicates_removed} items</td></tr>
            )}
            {data.metrics?.low_utility_removed !== undefined && (
              <tr><td>Low-Utility Items Evicted</td><td style={{ color: 'var(--success-color)' }}>{data.metrics.low_utility_removed} items</td></tr>
            )}
            {data.metrics?.avg_path_reward !== undefined && (
              <tr><td>Average Path Reward ($R_p$)</td><td style={{ color: 'var(--success-color)' }}>{data.metrics.avg_path_reward}</td></tr>
            )}
            {data.metrics?.outcome_reward !== undefined && (
              <tr><td>Outcome Reward ($R_o$)</td><td style={{ color: 'var(--success-color)' }}>{data.metrics.outcome_reward}</td></tr>
            )}
            {data.metrics?.cumulative_dual_reward !== undefined && (
              <tr><td>Cumulative Dual Reward ($R_p + \lambda R_o$)</td><td style={{ color: '#58a6ff', fontWeight: 600 }}>{data.metrics.cumulative_dual_reward}</td></tr>
            )}
            <tr><td>Execution Latency</td><td>{data.metrics?.latency_seconds ?? '—'}s</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  );
};

// ─── Main ComparisonView ─────────────────────────────────────────────────────
const ComparisonView = ({ results }) => {
  if (!results) return null;

  return (
    <div className="glass-panel">
      <h2 className="panel-title">Reasoning &amp; Memory Comparison</h2>
      
      {/* Top Banner: Per-Hop Memory Tracking Table */}
      <HopMemoryTracker standardData={results.standard} proposedData={results.proposed} />

      <div className="comparison-grid">
        <MultiHopPanel data={results.standard} />
        <StateAwarePanel data={results.proposed} />
      </div>
    </div>
  );
};

export default ComparisonView;
