import React, { useRef } from 'react';

export default function DataUpload({
  fileInfo,
  setFileInfo,
  uploadError,
  setUploadError,
  isLoading,
  setIsLoading,
}) {
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setUploadError('');

    if (!file) {
      return;
    }

    if (!file.name.toLowerCase().endsWith('.csv')) {
      setUploadError('Invalid file type: Please select a file with a .csv extension.');
      setFileInfo(null);
      return;
    }

    setIsLoading(true);

    const reader = new FileReader();

    reader.onload = (event) => {
      try {
        const text = event.target.result;
        const lines = text.split(/\r\n|\n/).filter((l) => l.trim().length > 0);

        if (lines.length < 2) {
          setUploadError('CSV file is empty or missing a data row.');
          setFileInfo(null);
          setIsLoading(false);
          return;
        }

        const headers = lines[0].split(',').map((h) => h.trim().replace(/^["']|["']$/g, ''));
        const rowCount = lines.length - 1;

        setFileInfo({
          name: file.name,
          sizeBytes: file.size,
          rowCount: rowCount,
          headers: headers,
        });
      } catch (err) {
        setUploadError(`Failed to parse CSV client-side: ${err.message}`);
        setFileInfo(null);
      } finally {
        setIsLoading(false);
      }
    };

    reader.onerror = () => {
      setUploadError('Error reading file from disk.');
      setFileInfo(null);
      setIsLoading(false);
    };

    reader.readAsText(file);
  };

  const handleRemoveFile = () => {
    setFileInfo(null);
    setUploadError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <section className="card">
      <h2 className="card-title">Historical Campaign Data</h2>
      <p className="card-description">
        Select a campaign dataset (.csv) for causal evaluation. 
        <em> Note: Processing is performed purely client-side in browser memory. No data is sent to an external server.</em>
      </p>

      <div className="format-guidelines">
        <strong>Expected Schema:</strong> Pre-treatment confounders (e.g., <code>historical_spend</code>, <code>cart_abandonment_rate</code>), treatment arm (<code>treatment_arm</code>), and binary conversion outcome (<code>purchased</code>).
      </div>

      <div className="file-input-container">
        <label htmlFor="csv-upload" className="visually-hidden">Choose CSV file</label>
        <input
          id="csv-upload"
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          className="file-input"
        />
        {fileInfo && (
          <button
            type="button"
            onClick={handleRemoveFile}
            className="btn btn-secondary"
            aria-label="Remove selected CSV file"
          >
            Remove File
          </button>
        )}
      </div>

      {isLoading && <div className="feedback-banner info-banner">Reading and parsing CSV in browser...</div>}

      {uploadError && (
        <div className="feedback-banner error-banner" role="alert">
          {uploadError}
        </div>
      )}
    </section>
  );
}
