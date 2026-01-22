import type { CompressionHistory, CompressionEvent } from './api';
import type { Pipeline, PipelineStep } from './pipeline';
import type { ChartVisibilityConfig } from './chart';

// ============================================
// ============================================

export interface UseCompressionDataReturn {
  
  data: CompressionHistory | null;
  
  isLoading: boolean;
  
  error: string | null;
  
  refetch: () => Promise<void>;
  
  hasData: boolean;
}

// ============================================
// ============================================

export interface CompressionStats {
  
  originalDim: number;
  
  finalDim: number;
  
  ratio: number;
  
  stepCount: number;
  
  dimensionFlow: number[];
}

export interface UseCompressionPipelineReturn {
  
  event: CompressionEvent | null;
  
  pipeline: Pipeline | null;
  
  activeSteps: PipelineStep[];
  
  hasStepType: (typePrefix: string) => boolean;
  
  getCompressionStats: () => CompressionStats | null;
  
  hasSHAP: boolean;
  
  hasCorrelation: boolean;
  
  hasAdaptive: boolean;
  
  hasImportanceBasedDimension: boolean;
  
  dimensionStepWithCalculator: PipelineStep | undefined;
}

// ============================================
// ============================================

export interface ChartMockData {
  
  paramImportances: number[] | null;
  
  iterations: number[] | null;
  
  dimensions: number[] | null;
  
  multiTaskImportances: number[][] | null;
  
  taskNames: string[] | null;
  
  sourceSimilarities: Record<string, number> | null;
}

export interface RangeCompressionStepInfo {
  step: PipelineStep;
  index: number;
}

export interface UseChartVisibilityReturn extends ChartVisibilityConfig {
  
  chartData: ChartMockData;
  
  rangeCompressionSteps: RangeCompressionStepInfo[];
}

// ============================================
// ============================================

export interface UseChartConfigParams<T = unknown> {
  
  type: string;
  
  data: T;
  
  options?: Record<string, unknown>;
  
  deps?: unknown[];
}

export interface UseChartConfigReturn {
  
  option: Record<string, unknown>;
  
  dimensions: {
    width?: string | number;
    height: string | number;
  };
  
  isValid: boolean;
  
  error: string | null;
}

// ============================================
// ============================================

export interface UseLazyChartOptions {
  
  threshold?: number;
  
  rootMargin?: string;
  
  onVisible?: () => void;
  
  disabled?: boolean;
}

export interface UseLazyChartReturn {
  
  ref: React.RefObject<HTMLDivElement>;
  
  isVisible: boolean;
  
  hasLoaded: boolean;
}

// ============================================
// ============================================

export interface UseChunkedDataOptions {
  
  chunkSize?: number;
  
  enabled?: boolean;
  
  threshold?: number;
  
  onComplete?: () => void;
  
  idleTimeout?: number;
}

export interface UseChunkedDataReturn<T> {
  
  displayData: T[];
  
  progress: number;
  
  isComplete: boolean;
  
  isLoading: boolean;
  
  reset: () => void;
}
