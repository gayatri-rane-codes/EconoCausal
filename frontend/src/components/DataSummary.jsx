import React from 'react';

export default function DataSummary({ fileInfo }) {
  if (!fileInfo) {
    return null;
  }

  const formatBytes = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <section className="card">
      <h2 className="card-title">Dataset Inspection & Summary</h2>
      <p className="card-description">
        Client-side summary of the loaded CSV file. 
        <em> (Causal effect estimation and DML modelling are not run on the client.)</em>
      </p>

      <div className="summary-grid">
        <div className="summary-metric">
          <span className="metric-label">File Name</span>
          <span className="metric-value">{fileInfo.name}</span>
        </div>
        <div className="summary-metric">
          <span className="metric-label">File Size</span>
          <span className="metric-value">{formatBytes(fileInfo.sizeBytes)}</span>
        </div>
        <div className="summary-metric">
          <span className="metric-label">Recorded Rows</span>
          <span className="metric-value">{fileInfo.rowCount.toLocaleString()}</span>
        </div>
        <div className="summary-metric">
          <span className="metric-label">Detected Columns</span>
          <span className="metric-value">{fileInfo.headers.length}</span>
        </div>
      </div>

      <div className="columns-container">
        <h3 className="columns-title">Parsed Columns:</h3>
        <div className="tags-list">
          {fileInfo.headers.map((col, idx) => (
            <span key={idx} className="column-tag">
              {col}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}
