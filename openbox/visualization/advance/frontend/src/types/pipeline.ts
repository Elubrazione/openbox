// ============================================
// ============================================

export interface ParameterCompression {
  
  name: string;
  
  type: string;
  
  original_range: number[];
  
  compressed_range: number[];
  
  compression_ratio: number;
  
  original_num_values?: number;
  
  quantized_num_values?: number;
}

export interface CompressionInfo {
  
  compressed_params: ParameterCompression[];
  
  unchanged_params: string[];
  
  avg_compression_ratio: number;
}

// ============================================
// ============================================

export interface PipelineStep {
  
  name: string;
  
  type: string;
  
  step_index: number;
  
  input_space_params: number;
  
  output_space_params: number;
  
  supports_adaptive_update: boolean;
  
  uses_progressive_compression: boolean;
  
  compression_ratio?: number;
  
  compression_info?: CompressionInfo;
  
  selected_parameters?: string[];
  
  selected_indices?: number[];
  
  calculator?: string;
  
  topk?: number;
  
  top_ratio?: number;
  
  sigma?: number;
  
  enable_mixed_sampling?: boolean;
  
  initial_prob?: number;
}

export interface Pipeline {
  
  n_steps: number;
  
  steps: PipelineStep[];
  
  sampling_strategy?: string;
}

// ============================================
// ============================================

export enum StepType {
  DIMENSION_SELECTION = 'dimension_selection',
  RANGE_COMPRESSION = 'range_compression',
  QUANTIZATION = 'quantization',
  PROJECTION = 'projection',
  NONE = 'none',
}

export enum CalculatorType {
  SHAP = 'SHAP',
  CORRELATION = 'Correlation',
  MUTUAL_INFO = 'MutualInfo',
  VARIANCE = 'Variance',
}

export function isStepOfType(step: PipelineStep, type: StepType): boolean {
  return step.type.toLowerCase().includes(type.toLowerCase()) ||
         step.name.toLowerCase().includes(type.toLowerCase());
}

export function hasCalculator(step: PipelineStep, calculator: CalculatorType): boolean {
  return step.calculator?.includes(calculator) ?? false;
}
