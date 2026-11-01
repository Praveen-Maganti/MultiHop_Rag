import React, { useState } from 'react';
import UploadPanel from './UploadPanel';
import ChatInterface from './ChatInterface';
import ComparisonView from './ComparisonView';
import MetricsPanel from './MetricsPanel';
import AblationPanel from './AblationPanel';
import './index.css';

function App() {
  // List of uploaded files: [{document_id, filename, chunks_created}]
  const [activeMode, setActiveMode] = useState('chat'); // 'chat' | 'ablation'
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [currentMode, setCurrentMode] = useState('socratic');

  const handleFileUpload = async (file) => {
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/documents/upload', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error('Upload failed');
      const data = await response.json();
      // Avoid duplicates by filename
      setUploadedFiles(prev => {
        const exists = prev.find(f => f.filename === data.filename);
        if (exists) return prev;
        return [...prev, { document_id: data.document_id, filename: data.filename, chunks_created: data.chunks_created }];
      });
    } catch (error) {
      console.error(error);
      alert(`Failed to upload ${file.name}`);
    } finally {
      setLoading(false);
    }
  };

  const handleRemove = async (documentId) => {
    try {
      await fetch(`http://127.0.0.1:8000/api/documents/${documentId}`, {
        method: 'DELETE',
      });
    } catch (err) {
      console.warn('Backend remove failed (may already be gone):', err);
    }
    setUploadedFiles(prev => prev.filter(f => f.document_id !== documentId));
    setResults(null);
  };

  const handleAsk = async (question, groundTruth, mode = 'socratic') => {
    if (uploadedFiles.length === 0) {
      alert('Please upload at least one document first.');
      return;
    }

    setLoading(true);
    setCurrentMode(mode);
    try {
      const primaryDocId = uploadedFiles[0].document_id;

      const body = {
        document_id: primaryDocId,
        question: question,
        mode: mode
      };
      if (groundTruth) body.ground_truth = groundTruth;

      const response = await fetch('http://127.0.0.1:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!response.ok) throw new Error('Network response was not ok');

      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error('Failed to fetch results:', error);
      alert('Error processing query. Ensure the backend is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>State-Aware Multi-Hop RAG</h1>
        <p className="subtitle">Dynamic Working Memory &amp; Cognitive Workspace for Complex Reasoning</p>
      </header>

      <UploadPanel
        uploadedFiles={uploadedFiles}
        onUpload={handleFileUpload}
        onRemove={handleRemove}
      />

      {/* Mode switcher */}
      {uploadedFiles.length > 0 && (
        <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'center' }}>
          <button
            onClick={() => setActiveMode('chat')}
            style={{
              background: activeMode === 'chat' ? 'rgba(88,166,255,0.2)' : 'rgba(255,255,255,0.05)',
              border: `1px solid ${activeMode === 'chat' ? 'var(--accent-color)' : 'rgba(255,255,255,0.15)'}`,
              borderRadius: '8px', color: activeMode === 'chat' ? 'var(--accent-color)' : 'var(--text-secondary)',
              padding: '0.5rem 1.5rem', cursor: 'pointer', fontSize: '0.95rem', fontWeight: 600,
            }}
          >
            Chat Comparison
          </button>
          <button
            onClick={() => setActiveMode('ablation')}
            style={{
              background: activeMode === 'ablation' ? 'rgba(88,166,255,0.2)' : 'rgba(255,255,255,0.05)',
              border: `1px solid ${activeMode === 'ablation' ? 'var(--accent-color)' : 'rgba(255,255,255,0.15)'}`,
              borderRadius: '8px', color: activeMode === 'ablation' ? 'var(--accent-color)' : 'var(--text-secondary)',
              padding: '0.5rem 1.5rem', cursor: 'pointer', fontSize: '0.95rem', fontWeight: 600,
            }}
          >
            Ablation Study
          </button>
        </div>
      )}

      {activeMode === 'chat' && uploadedFiles.length > 0 && (
        <ChatInterface onAsk={handleAsk} />
      )}

      {activeMode === 'ablation' && uploadedFiles.length > 0 && (
        <AblationPanel uploadedFiles={uploadedFiles} />
      )}

      {activeMode === 'chat' && loading && (
        <div style={{ textAlign: 'center', margin: '2rem', color: 'var(--accent-color)' }}>
          Running Multi-Hop RAG &amp; Proposed RAG across {uploadedFiles.length} document{uploadedFiles.length !== 1 ? 's' : ''}...
        </div>
      )}

      {activeMode === 'chat' && !loading && results && (
        <>
          <ComparisonView results={results} mode={currentMode} />
          <MetricsPanel results={results} />
        </>
      )}
    </div>
  );
}

export default App;
