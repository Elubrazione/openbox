import React, { useState, useCallback } from 'react';
import ConfigSpaceForm from './ConfigSpaceForm';
import StepsForm, { StepsConfig } from './StepsForm';
import HistoryUploadForm from './HistoryUploadForm';
import { useUploadWizard } from '../hooks/useUploadWizard';
import './MultiStepUpload.css';

interface MultiStepUploadProps {
  onUploadSuccess: () => void;
}

const StepIndicator: React.FC<{ currentStep: number }> = ({ currentStep }) => (
  <div className="step-indicator">
    <div className={`step ${currentStep >= 1 ? 'active' : ''} ${currentStep > 1 ? 'completed' : ''}`}>
      <div className="step-number">1</div>
      <div className="step-label">Config Space</div>
    </div>
    <div className="step-line"></div>
    <div className={`step ${currentStep >= 2 ? 'active' : ''} ${currentStep > 2 ? 'completed' : ''}`}>
      <div className="step-number">2</div>
      <div className="step-label">Steps</div>
    </div>
    <div className="step-line"></div>
    <div className={`step ${currentStep >= 3 ? 'active' : ''}`}>
      <div className="step-number">3</div>
      <div className="step-label">History</div>
    </div>
  </div>
);

const UploadResult: React.FC<{ result: NonNullable<ReturnType<typeof useUploadWizard>['context']['uploadResult']> }> = ({ result }) => (
  <div className={`upload-result ${result.success ? 'success' : 'error'}`}>
    <h4>{result.success ? '✓ Success!' : '✗ Error'}</h4>
    <p>{result.message || result.error}</p>
    {result.compression?.result && (
      <div className="compression-details">
        <p><strong>Original Dimension:</strong> {result.compression.result.original_dim}</p>
        <p><strong>Surrogate Dimension:</strong> {result.compression.result.surrogate_dim}</p>
        <p><strong>Compression Ratio:</strong> {(result.compression.result.compression_ratio * 100).toFixed(2)}%</p>
      </div>
    )}
  </div>
);

const MultiStepUploadFSM: React.FC<MultiStepUploadProps> = ({ onUploadSuccess }) => {
  const [showModal, setShowModal] = useState(false);
  const {
    state,
    step,
    context,
    isUploading,
    next,
    back,
    submit,
    reset,
    retry,
  } = useUploadWizard({ onUploadSuccess });
  const handleClose = useCallback(() => {
    if (!isUploading) {
      setShowModal(false);
      reset();
    }
  }, [isUploading, reset]);
  const handleOpen = useCallback(() => {
    setShowModal(true);
  }, []);
  const handleConfigSpaceNext = useCallback((config: Record<string, any>) => {
    next(config);
  }, [next]);
  const handleStepsNext = useCallback((steps: StepsConfig) => {
    next(steps);
  }, [next]);
  const handleHistorySubmit = useCallback((files: File[]) => {
    submit(files);
  }, [submit]);
  const renderStepContent = () => {
    switch (state) {
      case 'configSpace':
        return (
          <ConfigSpaceForm
            onNext={handleConfigSpaceNext}
            initialData={context.configSpace || undefined}
          />
        );

      case 'stepsConfig':
        return (
          <StepsForm
            onNext={handleStepsNext}
            onBack={back}
            initialData={context.stepsConfig || undefined}
            configSpace={context.configSpace || undefined}
          />
        );

      case 'historyUpload':
        return (
          <HistoryUploadForm
            onSubmit={handleHistorySubmit}
            onBack={back}
            uploading={false}
          />
        );

      case 'uploading':
        return (
          <div className="uploading-state">
            <div className="spinner"></div>
            <p>Uploading and processing...</p>
          </div>
        );

      case 'success':
      case 'error':
        return null;

      default:
        return null;
    }
  };

  return (
    <>
      <button className="upload-button" onClick={handleOpen}>
        Configure & Upload
      </button>

      {showModal && (
        <div className="modal-overlay" onClick={handleClose}>
          <div className="modal-content multi-step" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Compression Configuration</h2>
              <div className="modal-header-badge">
                <span className="fsm-badge">FSM</span>
              </div>
              <button className="close-button" onClick={handleClose} disabled={isUploading}>
                ×
              </button>
            </div>
            <StepIndicator currentStep={step} />
            <div className="modal-body">
              {renderStepContent()}
              {context.uploadResult && (
                <UploadResult result={context.uploadResult} />
              )}
              {state === 'error' && (
                <div className="error-actions">
                  <button className="btn-retry" onClick={retry}>
                    Retry Upload
                  </button>
                  <button className="btn-reset" onClick={reset}>
                    Start Over
                  </button>
                </div>
              )}
              {state === 'success' && (
                <div className="success-actions">
                  <button className="btn-close" onClick={handleClose}>
                    Close
                  </button>
                </div>
              )}
            </div>
            {process.env.NODE_ENV === 'development' && (
              <div className="fsm-debug">
                <span>State: <code>{state}</code></span>
                <span>Step: <code>{step}</code></span>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
};

export default MultiStepUploadFSM;
