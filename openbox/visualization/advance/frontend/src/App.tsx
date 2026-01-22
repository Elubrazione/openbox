import React from 'react';
import { useCompressionData } from './hooks';
import CompressionSummary from './components/CompressionSummary';
import MultiStepUpload from './components/MultiStepUpload';
import {
  RangeCompressionContainer,
  ParameterImportanceContainer,
  DimensionEvolutionContainer,
  MultiTaskHeatmapContainer,
  SourceSimilaritiesContainer,
} from './components/containers';
import './App.css';

const LoadingState: React.FC<{ onUploadSuccess: () => void }> = ({ onUploadSuccess }) => (
  <div className="app-container">
    <MultiStepUpload onUploadSuccess={onUploadSuccess} />
    <div className="loading">Loading compression history...</div>
  </div>
);

const ErrorState: React.FC<{
  error: string | null;
  onUploadSuccess: () => void;
  onRetry: () => void;
}> = ({ error, onUploadSuccess, onRetry }) => (
  <div className="app-container">
    <MultiStepUpload onUploadSuccess={onUploadSuccess} />
    <div className="error-container">
      <div className="error">Error: {error || 'No data available'}</div>
      <button className="refresh-button" onClick={onRetry}>Retry</button>
    </div>
  </div>
);

const App: React.FC = () => {
  const { data, isLoading, error, refetch, hasData } = useCompressionData();

  if (isLoading) return <LoadingState onUploadSuccess={refetch} />;
  if (error || !hasData) return <ErrorState error={error} onUploadSuccess={refetch} onRetry={refetch} />;

  return (
    <div className="app-container">
      <MultiStepUpload onUploadSuccess={refetch} />
      <button className="refresh-button-main" onClick={refetch} title="Refresh">⟳ Refresh</button>
      <main className="app-main">
        <CompressionSummary data={data!} />
        {/* <LargeDataChartDemo /> */}

        <RangeCompressionContainer data={data!} />
        <ParameterImportanceContainer data={data!} />
        <DimensionEvolutionContainer data={data!} />
        <MultiTaskHeatmapContainer data={data!} />
        <SourceSimilaritiesContainer data={data!} />
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>OpenBox Visualization Dashboard v1.0 | Powered by React + TypeScript + ECharts</p>
      </footer>
    </div>
  );
};

export default App;
