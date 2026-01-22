import React, { useState } from 'react';
import './FileUpload.css';

interface FileUploadProps {
  onUploadSuccess: () => void;
}

interface ValidationResult {
  valid: boolean;
  error?: string;
}

interface HistoryValidation extends ValidationResult {
  file: string;
}

interface UploadResponse {
  success: boolean;
  saved_files?: string[];
  validation_results?: {
    config_space: ValidationResult;
    steps: ValidationResult;
    history: HistoryValidation[];
  };
  all_valid?: boolean;
  message?: string;
  compression?: {
    status: string;
    message: string;
    result?: any;
    error?: string;
  };
  error?: string;
}

const FileUpload: React.FC<FileUploadProps> = ({ onUploadSuccess }) => {
  const [configSpaceFile, setConfigSpaceFile] = useState<File | null>(null);
  const [stepsFile, setStepsFile] = useState<File | null>(null);
  const [historyFiles, setHistoryFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [showModal, setShowModal] = useState(false);

  const handleFileChange = (
    e: React.ChangeEvent<HTMLInputElement>,
    fileType: 'config_space' | 'steps' | 'history'
  ) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    if (fileType === 'config_space') {
      setConfigSpaceFile(files[0]);
    } else if (fileType === 'steps') {
      setStepsFile(files[0]);
    } else if (fileType === 'history') {
      setHistoryFiles(Array.from(files));
    }
  };

  const handleUpload = async () => {
    if (!configSpaceFile || !stepsFile || historyFiles.length === 0) {
      alert('Please select all required files: config_space.json, steps.json, and at least one history file');
      return;
    }

    setUploading(true);
    setUploadResult(null);

    try {
      const formData = new FormData();
      formData.append('config_space', configSpaceFile);
      formData.append('steps', stepsFile);
      historyFiles.forEach((file) => {
        formData.append('history', file);
      });

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      const data: UploadResponse = await response.json();
      setUploadResult(data);

      if (data.success && data.all_valid) {
        // Wait a moment for compression to complete if it was started
        setTimeout(() => {
          onUploadSuccess();
          setShowModal(false);
          // Reset form
          setConfigSpaceFile(null);
          setStepsFile(null);
          setHistoryFiles([]);
        }, 500);
      }
    } catch (error) {
      console.error('Upload error:', error);
      setUploadResult({
        success: false,
        error: error instanceof Error ? error.message : 'Upload failed',
      });
    } finally {
      setUploading(false);
    }
  };

  const handleClear = () => {
    setConfigSpaceFile(null);
    setStepsFile(null);
    setHistoryFiles([]);
    setUploadResult(null);
  };

  return (
    <>
      <button className="upload-button" onClick={() => setShowModal(true)}>
        Upload Files
      </button>

      {showModal && (
        <div className="modal-overlay" onClick={() => !uploading && setShowModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Upload Configuration Files</h2>
              <button
                className="close-button"
                onClick={() => !uploading && setShowModal(false)}
                disabled={uploading}
              >
                ×
              </button>
            </div>

            <div className="upload-form">
              <div className="upload-section">
                <h3>Step 1: Upload Config Space</h3>
                <p className="upload-description">
                  Configuration space definition (JSON file)
                </p>
                <div className="file-input-wrapper">
                  <label className="file-label">
                    <input
                      type="file"
                      accept=".json"
                      onChange={(e) => handleFileChange(e, 'config_space')}
                      disabled={uploading}
                    />
                    <span className="file-label-text">
                      {configSpaceFile ? configSpaceFile.name : 'Choose config_space.json'}
                    </span>
                  </label>
                  {configSpaceFile && (
                    <span className="file-size">
                      ({(configSpaceFile.size / 1024).toFixed(2)} KB)
                    </span>
                  )}
                </div>
                {uploadResult?.validation_results?.config_space && (
                  <div
                    className={`validation-result ${uploadResult.validation_results.config_space.valid ? 'success' : 'error'
                      }`}
                  >
                    {uploadResult.validation_results.config_space.valid
                      ? '✓ Valid'
                      : `✗ ${uploadResult.validation_results.config_space.error}`}
                  </div>
                )}
              </div>

              <div className="upload-section">
                <h3>Step 2: Upload Steps Configuration</h3>
                <p className="upload-description">
                  Compression steps configuration (JSON file)
                </p>
                <div className="file-input-wrapper">
                  <label className="file-label">
                    <input
                      type="file"
                      accept=".json"
                      onChange={(e) => handleFileChange(e, 'steps')}
                      disabled={uploading}
                    />
                    <span className="file-label-text">
                      {stepsFile ? stepsFile.name : 'Choose steps.json'}
                    </span>
                  </label>
                  {stepsFile && (
                    <span className="file-size">({(stepsFile.size / 1024).toFixed(2)} KB)</span>
                  )}
                </div>
                {uploadResult?.validation_results?.steps && (
                  <div
                    className={`validation-result ${uploadResult.validation_results.steps.valid ? 'success' : 'error'
                      }`}
                  >
                    {uploadResult.validation_results.steps.valid
                      ? '✓ Valid'
                      : `✗ ${uploadResult.validation_results.steps.error}`}
                  </div>
                )}
              </div>

              <div className="upload-section">
                <h3>Step 3: Upload History Files</h3>
                <p className="upload-description">
                  One or more history files (JSON files, can select multiple)
                </p>
                <div className="file-input-wrapper">
                  <label className="file-label">
                    <input
                      type="file"
                      accept=".json"
                      multiple
                      onChange={(e) => handleFileChange(e, 'history')}
                      disabled={uploading}
                    />
                    <span className="file-label-text">
                      {historyFiles.length > 0
                        ? `${historyFiles.length} file(s) selected`
                        : 'Choose history.json file(s)'}
                    </span>
                  </label>
                </div>
                {historyFiles.length > 0 && (
                  <div className="file-list">
                    {historyFiles.map((file, idx) => (
                      <div key={idx} className="file-item">
                        {file.name} ({(file.size / 1024).toFixed(2)} KB)
                      </div>
                    ))}
                  </div>
                )}
                {uploadResult?.validation_results?.history &&
                  uploadResult.validation_results.history.map((result, idx) => (
                    <div
                      key={idx}
                      className={`validation-result ${result.valid ? 'success' : 'error'}`}
                    >
                      {result.file}: {result.valid ? '✓ Valid' : `✗ ${result.error}`}
                    </div>
                  ))}
              </div>

              {uploadResult && (
                <div className={`upload-result ${uploadResult.success ? 'success' : 'error'}`}>
                  <h4>{uploadResult.success ? 'Upload Successful' : 'Upload Failed'}</h4>
                  <p>{uploadResult.message || uploadResult.error}</p>
                  {uploadResult.compression && (
                    <div className="compression-status">
                      <h5>Compression Status: {uploadResult.compression.status}</h5>
                      <p>{uploadResult.compression.message}</p>
                      {uploadResult.compression.result && (
                        <div className="compression-details">
                          <p>
                            Original Dimension: {uploadResult.compression.result.original_dim}
                          </p>
                          <p>
                            Surrogate Dimension: {uploadResult.compression.result.surrogate_dim}
                          </p>
                          <p>
                            Compression Ratio:{' '}
                            {(uploadResult.compression.result.compression_ratio * 100).toFixed(2)}%
                          </p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              <div className="modal-actions">
                <button className="btn btn-secondary" onClick={handleClear} disabled={uploading}>
                  Clear
                </button>
                <button
                  className="btn btn-primary"
                  onClick={handleUpload}
                  disabled={uploading || !configSpaceFile || !stepsFile || historyFiles.length === 0}
                >
                  {uploading ? 'Uploading & Processing...' : 'Upload & Run Compression'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default FileUpload;
