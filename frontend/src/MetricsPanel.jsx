import React from 'react';

const MetricsPanel = ({ results }) => {
  if (!results) return null;

  return (
    <div className="glass-panel" style={{ marginTop: '2rem' }}>
      <h2 className="panel-title">Overall Performance Comparison</h2>
      
      <table className="metrics-table" style={{ marginTop: '1rem' }}>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Standard RAG</th>
            <th>Proposed RAG</th>
            <th>Improvement</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Answer Quality (F1)</td>
            <td>{results.standard.evaluation.f1}</td>
            <td>{results.proposed.evaluation.f1}</td>
            <td style={{ color: results.proposed.evaluation.f1 >= results.standard.evaluation.f1 ? 'var(--success-color)' : '#f85149' }}>
              {results.standard.evaluation.f1 === 0 && results.proposed.evaluation.f1 === 0
                ? '—'
                : results.standard.evaluation.f1 === 0
                  ? `+${(results.proposed.evaluation.f1 * 100).toFixed(1)}% pts`
                  : `${(((results.proposed.evaluation.f1 - results.standard.evaluation.f1) / results.standard.evaluation.f1) * 100).toFixed(1)}%`}
            </td>
          </tr>
          <tr>
            <td>Evidence Coverage</td>
            <td>{results.standard.evaluation.coverage}</td>
            <td>{results.proposed.evaluation.coverage}</td>
            <td style={{ color: results.proposed.evaluation.coverage >= results.standard.evaluation.coverage ? 'var(--success-color)' : '#f85149' }}>
              {results.standard.evaluation.coverage === 0 && results.proposed.evaluation.coverage === 0
                ? '—'
                : results.standard.evaluation.coverage === 0
                  ? `+${(results.proposed.evaluation.coverage * 100).toFixed(1)}% pts`
                  : `${(((results.proposed.evaluation.coverage - results.standard.evaluation.coverage) / results.standard.evaluation.coverage) * 100).toFixed(1)}%`}
            </td>
          </tr>
          <tr>
            <td>Unsupported Claims</td>
            <td>{(results.standard.evaluation.unsupported * 100).toFixed(1)}%</td>
            <td>{(results.proposed.evaluation.unsupported * 100).toFixed(1)}%</td>
            <td style={{ color: results.proposed.evaluation.unsupported <= results.standard.evaluation.unsupported ? 'var(--success-color)' : '#f85149' }}>
              {results.standard.evaluation.unsupported === 0 && results.proposed.evaluation.unsupported === 0
                ? '—'
                : results.standard.evaluation.unsupported === 0
                  ? `+${((results.proposed.evaluation.unsupported - results.standard.evaluation.unsupported) * 100).toFixed(1)}% pts`
                  : `-${(((results.standard.evaluation.unsupported - results.proposed.evaluation.unsupported) / results.standard.evaluation.unsupported) * 100).toFixed(1)}%`}
            </td>
          </tr>
          <tr>
            <td>LLM Tokens Used</td>
            <td>{results.standard.metrics.estimated_tokens || results.standard.metrics.tokens || 0}</td>
            <td>{results.proposed.metrics.estimated_tokens || results.proposed.metrics.tokens || 0}</td>
            <td style={{ color: 'var(--success-color)' }}>
              {(() => {
                const std = parseFloat(results.standard.metrics.estimated_tokens || results.standard.metrics.tokens || 0);
                const prop = parseFloat(results.proposed.metrics.estimated_tokens || results.proposed.metrics.tokens || 0);
                if (std === 0) return '—';
                const diff = (((std - prop) / std) * 100).toFixed(1);
                return diff >= 0 ? `-${diff}%` : `+${Math.abs(diff)}%`;
              })()}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default MetricsPanel;
