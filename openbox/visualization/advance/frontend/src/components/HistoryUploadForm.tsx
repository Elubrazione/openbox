import React, { useState } from 'react';

interface HistoryUploadFormProps {
  onSubmit: (historyFiles: File[]) => void;
  onBack: () => void;
  uploading: boolean;
}

const HistoryUploadForm: React.FC<HistoryUploadFormProps> = ({ onSubmit, onBack, uploading }) => {
  const [historyFiles, setHistoryFiles] = useState<File[]>([]);
  const [dragActive, setDragActive] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setHistoryFiles(Array.from(e.target.files));
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const files = Array.from(e.dataTransfer.files).filter((file) =>
        file.name.endsWith('.json')
      );
      if (files.length > 0) {
        setHistoryFiles(files);
      } else {
        alert('Please upload JSON files only');
      }
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (historyFiles.length === 0) {
      alert('Please upload at least one history file');
      return;
    }
    onSubmit(historyFiles);
  };

  const removeFile = (index: number) => {
    setHistoryFiles(historyFiles.filter((_, i) => i !== index));
  };

  return (
    <form onSubmit={handleSubmit} className="history-upload-form">
      <div className="form-header">
        <h3>Step 3: Upload History Data</h3>
        <p className="form-description">Upload one or more history JSON files</p>
      </div>

      <div className="form-content">
        <div
          className={`dropzone ${dragActive ? 'active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <div className="dropzone-content">
            <div className="upload-icon">📁</div>
            <p className="dropzone-text">
              Drag and drop history files here, or click to select
            </p>
            <input
              type="file"
              accept=".json"
              multiple
              onChange={handleFileChange}
              className="file-input-hidden"
              id="history-file-input"
              disabled={uploading}
            />
            <label htmlFor="history-file-input" className="btn btn-secondary">
              Select Files
            </label>
            <p className="dropzone-hint">Supports multiple JSON files</p>
          </div>
        </div>

        {historyFiles.length > 0 && (
          <div className="file-list">
            <h4>Selected Files ({historyFiles.length})</h4>
            {historyFiles.map((file, index) => (
              <div key={index} className="file-item">
                <div className="file-info">
                  <span className="file-name">📄 {file.name}</span>
                  <span className="file-size">({(file.size / 1024).toFixed(2)} KB)</span>
                </div>
                <button
                  type="button"
                  className="btn-remove-file"
                  onClick={() => removeFile(index)}
                  disabled={uploading}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="info-box">
          <h4>History File Format</h4>
          <p>Each history file should be a JSON array containing observation objects:</p>
          <pre>{`[
  {
    "config": {
      "param1": value1,
      "param2": value2
    },
    "objectives": {
      "metric": value
    }
  },
  ...
]`}</pre>
        </div>
      </div>

      <div className="form-actions">
        <button type="button" className="btn btn-secondary" onClick={onBack} disabled={uploading}>
          Back
        </button>
        <button
          type="submit"
          className="btn btn-primary"
          disabled={historyFiles.length === 0 || uploading}
        >
          {uploading ? 'Uploading & Processing...' : 'Submit & Run Compression'}
        </button>
      </div>
    </form>
  );
};

export default HistoryUploadForm;
