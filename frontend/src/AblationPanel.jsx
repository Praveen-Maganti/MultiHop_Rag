import React, { useState } from 'react';

const VARIANT_ORDER = [
  'full',
  'no_reranker',
  'no_filter',
  'no_claim_extractor',
  'no_conflict',
  'no_compressor',
];

const VARIANT_COLORS = {
  full:               'var(--accent-color)',
  no_reranker:        '#e3b341',
  no_filter:          '#f0883e',
  no_claim_extractor: '#f85149',
  no_conflict:        '#bc8cff',
  no_compressor:      '#8b949e',
};

const fmt = (v, d = 3) => {
  if (v === null || v === undefined) return '—';
  const n = parseFloat(v);
  return isNaN(n) ? '—' : n.toFixed(d);
};

const DeltaBadge = ({ delta }) => {
  if (delta === null || delta === undefined || delta === 0) return null;
  const n = parseFloat(delta);
  if (isNaN(n) || n === 0) return null;
  const better = n >= 0;
  return (
    <span style={{
      fontSize: '0.7rem',
      fontWeight: 700,
      color: better ? 'var(--success-color)' : 'var(--danger-color)',
      marginLeft: '0.4rem',
    }}>
      {better ? `+${n.toFixed(3)}` : n.toFixed(3)}
    </span>
  );
};

const AblationPanel = ({ uploadedFiles }) => {
  const [questions, setQuestions]     = useState([{ question: '', ground_truth: '' }]);
  const [loading, setLoading]         = useState(false);
  const [results, setResults]         = useState(null);
  const [error, setError]             = useState(null);
  const [activeTab, setActiveTab]     = useState('summary');
  const [activeQuestion, setActiveQuestion] = useState(0);

  const addQuestion = () =>
    setQuestions(prev => [...prev, { question: '', ground_truth: '' }]);

  const removeQuestion = (i) =>
    setQuestions(prev => prev.filter((_, idx) => idx !== i));

  const updateQuestion = (i, field, val) =>
    setQuestions(prev => prev.map((q, idx) => idx === i ? { ...q, [field]: val } : q));

  const handleRun = async () => {
    if (!uploadedFiles.length) { setError('Upload a document first.'); return; }
    const valid = questions.filter(q => q.question.trim());
    if (!valid.length) { setError('Enter at least one question.'); return; }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const body = {
        document_id: uploadedFiles[0].document_id,
        questions: valid.map(q => ({
          question: q.question.trim(),
          ground_truth: q.ground_truth.trim() || null,
        })),
      };
      const res = await fetch('http://127.0.0.1:8000/api/ablation/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Ablation failed');
      }
      const data = await res.json();
      setResults(data);
      setActiveQuestion(0);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  // ── Summary table ──────────────────────────────────────────────────────
  const SummaryTable = ({ summary }) => {
    const fullF1 = summary?.full?.f1 ?? 0;
    return (
      <div style={{ overflowX: 'auto' }}>
        <table className="metrics-table" style={{ minWidth: '900px' }}>
          <thead>
            <tr>
              <th style={{ minWidth: '200px' }}>Variant</th>
              <th>F1 ↑</th>
              <th>Precision ↑</th>
              <th>Recall ↑</th>
              <th>Coverage ↑</th>
              <th>Unsupported ↓</th>
              <th>Hops</th>
              <th>Tokens</th>
              <th>Latency (s)</th>
              <th>Mem Items</th>
            </tr>
          </thead>
          <tbody>
            {VARIANT_ORDER.map(v => {
              const row = summary?.[v];
              if (!row) return null;
              const isFull = v === 'full';
              const delta = isFull ? null : row.f1_delta;
              return (
                <tr key={v} style={{ background: isFull ? 'rgba(88,166,255,0.06)' : 'transparent' }}>
                  <td>
                    <span style={{
                      display: 'inline-block',
                      width: '10px', height: '10px',
                      borderRadius: '50%',
                      background: VARIANT_COLORS[v],
                      marginRight: '0.5rem',
                    }} />
                    <span style={{ fontWeight: isFull ? 700 : 400, fontSize: '0.85rem' }}>
                      {row.label}
                    </span>
                  </td>
                  <td>
                    <span style={{ fontWeight: isFull ? 700 : 400 }}>{fmt(row.f1)}</span>
                    {!isFull && <DeltaBadge delta={delta} />}
                  </td>
                  <td>{fmt(row.precision)}</td>
                  <td>{fmt(row.recall)}</td>
                  <td>{fmt(row.evidence_coverage)}</td>
                  <td style={{ color: row.unsupported_claim_rate > 0.2 ? 'var(--danger-color)' : 'inherit' }}>
                    {fmt(row.unsupported_claim_rate)}
                  </td>
                  <td>{fmt(row.hops, 1)}</td>
                  <td>{fmt(row.tokens, 0)}</td>
                  <td>{fmt(row.latency, 2)}</td>
                  <td>{fmt(row.memory_items, 1)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  };

  // ── Component contribution bar chart (text-based) ─────────────────────
  const ContributionChart = ({ summary }) => {
    const fullF1 = summary?.full?.f1 ?? 0;
    const variants = VARIANT_ORDER.filter(v => v !== 'full');
    return (
      <div style={{ marginTop: '1rem' }}>
        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
          F1 drop when component is removed (larger drop = more important component)
        </div>
        {variants.map(v => {
          const row = summary?.[v];
          if (!row) return null;
          const drop = fullF1 - row.f1;
          const pct  = fullF1 > 0 ? Math.abs(drop / fullF1) * 100 : 0;
          const componentName = row.label.replace('No ', '');
          return (
            <div key={v} style={{ marginBottom: '0.75rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.2rem' }}>
                <span style={{ fontSize: '0.82rem', color: 'var(--text-primary)' }}>{componentName}</span>
                <span style={{ fontSize: '0.82rem', color: drop > 0 ? 'var(--danger-color)' : 'var(--success-color)' }}>
                  {drop > 0 ? `-${drop.toFixed(3)} F1` : `+${Math.abs(drop).toFixed(3)} F1`}
                </span>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.06)', borderRadius: '4px', height: '8px', overflow: 'hidden' }}>
                <div style={{
                  width: `${Math.min(pct, 100)}%`,
                  height: '100%',
                  background: drop > 0 ? 'var(--danger-color)' : 'var(--success-color)',
                  borderRadius: '4px',
                  transition: 'width 0.5s ease',
                }} />
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  // ── Per-question detail view ───────────────────────────────────────────
  const PerQuestionView = ({ perQuestion }) => {
    const qData = perQuestion?.[activeQuestion];
    if (!qData) return null;
    return (
      <div>
        {/* Question selector */}
        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '1rem' }}>
          {perQuestion.map((pq, i) => (
            <button key={i} onClick={() => setActiveQuestion(i)} style={{
              background: activeQuestion === i ? 'rgba(88,166,255,0.2)' : 'rgba(255,255,255,0.05)',
              border: `1px solid ${activeQuestion === i ? 'var(--accent-color)' : 'rgba(255,255,255,0.1)'}`,
              borderRadius: '6px', color: 'var(--text-primary)',
              padding: '0.3rem 0.75rem', cursor: 'pointer', fontSize: '0.82rem',
            }}>
              Q{i + 1}
            </button>
          ))}
        </div>

        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
          <strong style={{ color: 'var(--text-primary)' }}>Question:</strong> {qData.question}
          {qData.ground_truth && (
            <span style={{ marginLeft: '1rem' }}>
              <strong style={{ color: 'var(--text-primary)' }}>Expected:</strong> {qData.ground_truth}
            </span>
          )}
        </div>

        {/* Per-variant answers */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {VARIANT_ORDER.map(v => {
            const vdata = qData.variants?.[v];
            if (!vdata) return null;
            return (
              <div key={v} style={{
                background: 'rgba(0,0,0,0.2)',
                border: `1px solid ${v === 'full' ? 'rgba(88,166,255,0.3)' : 'rgba(255,255,255,0.06)'}`,
                borderRadius: '8px', padding: '0.75rem',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                  <span style={{ fontWeight: 600, fontSize: '0.82rem', color: VARIANT_COLORS[v] }}>
                    {vdata.variant === 'full' ? '✦ ' : ''}{qData.variants[v] && VARIANT_ORDER.includes(v) ? v.replace(/_/g, ' ').toUpperCase() : v}
                  </span>
                  <div style={{ display: 'flex', gap: '0.4rem' }}>
                    <span className="badge">F1: {fmt(vdata.f1)}</span>
                    <span className="badge">Cov: {fmt(vdata.evidence_coverage)}</span>
                    <span className="badge">{fmt(vdata.latency, 1)}s</span>
                  </div>
                </div>
                <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', margin: 0, lineHeight: 1.5 }}>
                  {vdata.answer}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className="glass-panel">
      <h2 className="panel-title">Ablation Study</h2>
      <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1.25rem' }}>
        Runs the Proposed RAG 6 times — once with all components ON, then with each component
        disabled individually — to measure each component's contribution to F1 and accuracy.
      </div>

      {/* Question inputs */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', marginBottom: '1rem' }}>
        {questions.map((q, i) => (
          <div key={i} style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}>
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              <input
                className="chat-input"
                placeholder={`Question ${i + 1}`}
                value={q.question}
                onChange={e => updateQuestion(i, 'question', e.target.value)}
                style={{ fontSize: '0.9rem' }}
              />
              <input
                className="chat-input"
                placeholder="Expected answer (ground truth) — optional but needed for F1"
                value={q.ground_truth}
                onChange={e => updateQuestion(i, 'ground_truth', e.target.value)}
                style={{ fontSize: '0.85rem', opacity: 0.8 }}
              />
            </div>
            {questions.length > 1 && (
              <button onClick={() => removeQuestion(i)} style={{
                background: 'rgba(248,81,73,0.15)', border: '1px solid rgba(248,81,73,0.3)',
                borderRadius: '6px', color: 'var(--danger-color)',
                padding: '0.4rem 0.6rem', cursor: 'pointer', fontSize: '0.8rem', marginTop: '0.2rem',
              }}>✕</button>
            )}
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.5rem' }}>
        <button onClick={addQuestion} style={{
          background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.15)',
          borderRadius: '6px', color: 'var(--text-secondary)',
          padding: '0.4rem 1rem', cursor: 'pointer', fontSize: '0.85rem',
        }}>+ Add Question</button>
        <button onClick={handleRun} disabled={loading} className="btn-primary"
          style={{ padding: '0.4rem 1.5rem', opacity: loading ? 0.6 : 1 }}>
          {loading ? 'Running Ablation…' : 'Run Ablation Study'}
        </button>
      </div>

      {loading && (
        <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--accent-color)' }}>
          <div style={{ fontSize: '0.9rem', marginBottom: '0.5rem' }}>
            Running 6 variants × {questions.filter(q => q.question.trim()).length} question(s)…
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            This may take several minutes due to LLM calls per variant.
          </div>
        </div>
      )}

      {error && (
        <div style={{ color: 'var(--danger-color)', fontSize: '0.85rem', marginBottom: '1rem' }}>
          ⚠ {error}
        </div>
      )}

      {results && (
        <div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
            Ablation ID: <code style={{ color: 'var(--accent-color)' }}>{results.ablation_id}</code>
          </div>

          {/* Tabs */}
          <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.25rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.5rem' }}>
            {['summary', 'contribution', 'per_question'].map(tab => (
              <button key={tab} onClick={() => setActiveTab(tab)} style={{
                background: activeTab === tab ? 'rgba(88,166,255,0.15)' : 'transparent',
                border: `1px solid ${activeTab === tab ? 'var(--accent-color)' : 'transparent'}`,
                borderRadius: '6px', color: activeTab === tab ? 'var(--accent-color)' : 'var(--text-secondary)',
                padding: '0.3rem 0.9rem', cursor: 'pointer', fontSize: '0.85rem',
              }}>
                {tab === 'summary' ? 'Summary Table' : tab === 'contribution' ? 'Component Impact' : 'Per Question'}
              </button>
            ))}
          </div>

          {activeTab === 'summary'      && <SummaryTable    summary={results.summary} />}
          {activeTab === 'contribution' && <ContributionChart summary={results.summary} />}
          {activeTab === 'per_question' && <PerQuestionView  perQuestion={results.per_question} />}
        </div>
      )}
    </div>
  );
};

export default AblationPanel;
