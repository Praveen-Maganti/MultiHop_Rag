import React, { useState } from 'react';
import UploadPanel from './UploadPanel';
import ChatInterface from './ChatInterface';
import ComparisonView from './ComparisonView';
import MetricsPanel from './MetricsPanel';
import './index.css';

function App() {
  // List of uploaded files: [{document_id, filename, chunks_created}]
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

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

  const handleAsk = async (question, groundTruth) => {
    if (uploadedFiles.length === 0) {
      alert('Please upload at least one document first.');
      return;
    }

    setLoading(true);
    try {
      // Use the first uploaded document_id as the primary reference.
      // The proposed RAG will search_all_documents automatically.
      const primaryDocId = uploadedFiles[0].document_id;

      const body = {
        document_id: primaryDocId,
        question: question,
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
        <h1>Adaptive Memory RAG</h1>
        <p className="subtitle">Evidence- and Utility-Aware Working Memory Evaluation</p>
      </header>

      <UploadPanel
        uploadedFiles={uploadedFiles}
        onUpload={handleFileUpload}
        onRemove={handleRemove}
      />

      {uploadedFiles.length > 0 && (
        <ChatInterface onAsk={handleAsk} />
      )}

      {loading && (
        <div style={{ textAlign: 'center', margin: '2rem', color: 'var(--accent-color)' }}>
          Processing RAG pipelines across {uploadedFiles.length} document{uploadedFiles.length !== 1 ? 's' : ''}...
        </div>
      )}

      {!loading && results && (
        <>
          <ComparisonView results={results} />
          <MetricsPanel results={results} />
        </>
      )}
    </div>
  );
}

export default App;
