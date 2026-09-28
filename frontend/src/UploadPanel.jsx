import React, { useRef } from 'react';

const UploadPanel = ({ uploadedFiles, onUpload, onRemove }) => {
  const inputRef = useRef(null);

  const handleDragOver = (e) => e.preventDefault();

  const handleDrop = (e) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files).filter(f =>
      f.name.endsWith('.pdf') || f.name.endsWith('.docx') || f.name.endsWith('.txt')
    );
    files.forEach(f => onUpload(f));
  };

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files);
    files.forEach(f => onUpload(f));
    // Reset input so same file can be re-selected
    e.target.value = '';
  };

  return (
    <div className="glass-panel">
      <h2 className="panel-title">Upload Documents</h2>

      {/* Drop zone */}
      <div
        className="upload-dropzone"
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        style={{ marginBottom: uploadedFiles.length > 0 ? '1rem' : '0' }}
      >
        <div className="upload-icon">📂</div>
        <div>
          <h3>Drag &amp; Drop PDF / DOCX / TXT</h3>
          <p className="subtitle">Click to browse — multiple files supported</p>
        </div>
        <input
          ref={inputRef}
          type="file"
          style={{ display: 'none' }}
          onChange={handleFileChange}
          accept=".pdf,.docx,.txt"
          multiple
        />
      </div>

      {/* File list */}
      {uploadedFiles.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {uploadedFiles.map((f, i) => (
            <div
              key={f.document_id}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                background: 'rgba(255,255,255,0.04)',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '8px',
                padding: '0.6rem 1rem',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <span style={{ fontSize: '1.2rem' }}>✅</span>
                <div>
                  <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>{f.filename}</div>
                  <div style={{ color: '#8b949e', fontSize: '0.75rem' }}>
                    ID: {f.document_id} · {f.chunks_created} chunk{f.chunks_created !== 1 ? 's' : ''}
                  </div>
                </div>
              </div>
              <button
                onClick={() => onRemove(f.document_id)}
                style={{
                  background: 'rgba(248,81,73,0.15)',
                  border: '1px solid rgba(248,81,73,0.4)',
                  color: '#f85149',
                  borderRadius: '6px',
                  padding: '0.3rem 0.7rem',
                  cursor: 'pointer',
                  fontSize: '0.8rem',
                }}
              >
                ✕ Remove
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default UploadPanel;
