import React from 'react';

const fmt = (val, decimals = 3) => {
  if (val === null || val === undefined || val === '--') return '—';
  const n = parseFloat(val);
  return isNaN(n) ? '—' : n.toFixed(decimals);
};

const diffCell = (stdVal, propVal, lowerIsBetter = false) => {
  const s = parseFloat(stdVal);
  const p = parseFloat(propVal);
  if (isNaN(s) || isNaN(p) || s === 0) return <td style={{ color: 'var(--text-secondary)' }}>—</td>;
  const pct = (((p - s) / Math.abs(s)) * 100).toFixed(1);
  const better = lowerIsBetter ? p <= s : p >= s;
  return (
    <td style={{ color: better ? 'var(--success-color)' : '#f85149', fontWeight: 600 }}>
      {pct >= 0 ? `+${pct}%` : `${pct}%`}
    </td>
  );
};

const MetricsPanel = ({ results }) => {
  if (!results) return null;

  const mh = results.standard;   // Multi-Hop RAG
  const pr = results.proposed;   // Proposed RAG

  const mhTokens = parseFloat(mh.metrics?.estimated_tokens ?? mh.metrics?.tokens ?? 0);
  const prTokens = parseFloat(pr.metrics?.estimated_tokens ?? pr.metrics?.tokens ?? 0);
  const tokenDiff = mhTokens > 0
    ? (((prTokens - mhTokens) / mhTokens) * 100).toFixed(1)
    : null;

  const mhLatency = parseFloat(mh.metrics?.latency_seconds ?? mh.metrics?.latency ?? 0);
  const prLatency = parseFloat(pr.metrics?.latency_seconds ?? pr.metrics?.latency ?? 0);

  return (
    <div className="glass-panel" style={{ marginTop: '2rem' }}>
      <h2 className="panel-title">Overall Performance Comparison</h2>

      <table className="metrics-table" style={{ marginTop: '1rem' }}>
        <thead>
          <tr>
            <th>Metric</th>
            <th style={{ color: '#8b949e' }}>Multi-Hop RAG</th>
            <th style={{ color: 'var(--accent-color)' }}>Proposed RAG</th>
            <th>Δ Improvement</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Answer Quality (F1)</td>
            <td>{fmt(mh.evaluation?.f1)}</td>
            <td>{fmt(pr.evaluation?.f1)}</td>
            {diffCell(mh.evaluation?.f1, pr.evaluation?.f1)}
          </tr>
          <tr>
            <td>Evidence Coverage</td>
            <td>{fmt(mh.evaluation?.coverage)}</td>
            <td>{fmt(pr.evaluation?.coverage)}</td>
            {diffCell(mh.evaluation?.coverage, pr.evaluation?.coverage)}
          </tr>
          <tr>
            <td>Unsupported Claims ↓</td>
            <td>{fmt(mh.evaluation?.unsupported)}</td>
            <td>{fmt(pr.evaluation?.unsupported)}</td>
            {diffCell(mh.evaluation?.unsupported, pr.evaluation?.unsupported, true)}
          </tr>
          <tr>
            <td>Hops Executed</td>
            <td>{mh.metrics?.hops ?? '—'}</td>
            <td>{pr.metrics?.hops ?? '—'}</td>
            <td style={{ color: 'var(--text-secondary)' }}>—</td>
          </tr>
          <tr>
            <td>Retrieval Calls</td>
            <td>{mh.metrics?.retrieval_calls ?? '—'}</td>
            <td>{pr.metrics?.retrieval_calls ?? '—'}</td>
            <td style={{ color: 'var(--text-secondary)' }}>—</td>
          </tr>
          <tr>
            <td>Context / Memory Size</td>
            <td>{mh.metrics?.unique_chunks ?? '—'} chunks</td>
            <td>{pr.metrics?.memory_items ?? '—'} items</td>
            <td style={{ color: 'var(--text-secondary)' }}>—</td>
          </tr>
          <tr>
            <td>LLM Tokens Used</td>
            <td>{mhTokens}</td>
            <td>{prTokens}</td>
            <td style={{ color: tokenDiff !== null && parseFloat(tokenDiff) <= 0 ? 'var(--success-color)' : '#f85149', fontWeight: 600 }}>
              {tokenDiff !== null ? (parseFloat(tokenDiff) >= 0 ? `+${tokenDiff}%` : `${tokenDiff}%`) : '—'}
            </td>
          </tr>
          <tr>
            <td>Latency (s)</td>
            <td>{mhLatency.toFixed(2)}s</td>
            <td>{prLatency.toFixed(2)}s</td>
            {diffCell(mhLatency, prLatency, true)}
          </tr>
          <tr>
            <td>Memory Pruning</td>
            <td style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>None (flat)</td>
            <td style={{ color: 'var(--success-color)', fontSize: '0.85rem' }}>
              {pr.metrics?.duplicates_removed ?? 0} dup + {pr.metrics?.low_utility_removed ?? 0} low-util removed
            </td>
            <td style={{ color: 'var(--text-secondary)' }}>—</td>
          </tr>
          <tr>
            <td>Conflict Detection</td>
            <td style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>None</td>
            <td style={{ color: 'var(--success-color)', fontSize: '0.85rem' }}>
              {pr.metrics?.conflict_llm_calls ?? 0} calls · {pr.metrics?.conflict_calls_saved ?? 0} saved
            </td>
            <td style={{ color: 'var(--text-secondary)' }}>—</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default MetricsPanel;
