// ============================================
// ============================================

export enum EventType {
  INITIAL_COMPRESSION = 'initial_compression',
  ADAPTIVE_UPDATE = 'adaptive_update',
  PROGRESSIVE_COMPRESSION = 'progressive_compression',
}

export enum SpaceType {
  ORIGINAL = 'original',
  SAMPLE = 'sample',
  SURROGATE = 'surrogate',
}

// ============================================
// ============================================

export interface Space {
  
  n_parameters: number;
  
  parameters: string[];
  
  space_type?: SpaceType;
}

export interface SpaceSnapshot {
  
  original: Space;
  
  sample: Space;
  
  surrogate: Space;
}

// ============================================
// ============================================

export interface PerformanceMetrics {
  
  multi_task_importances?: number[][];
  
  task_names?: string[];
  
  source_similarities?: Record<string, number>;
}

// ============================================
// ============================================

export interface CompressionEvent {
  
  timestamp: string;
  
  event: EventType;
  
  iteration: number | null;
  
  spaces: SpaceSnapshot;
  
  compression_ratios: {
    
    sample_to_original: number;
    
    surrogate_to_original: number;
  };
  
  pipeline: import('./pipeline').Pipeline;
  
  update_reason?: string;
  
  performance_metrics?: PerformanceMetrics;
}

export interface CompressionHistory {
  
  total_updates: number;
  
  history: CompressionEvent[];
}

// ============================================
// ============================================

export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
}

export interface HealthCheckResponse {
  status: 'healthy' | 'unhealthy';
  service: string;
  data_dir: string;
  timestamp: string;
}

export interface UploadResponse {
  success: boolean;
  message?: string;
  files_received?: string[];
  error?: string;
}
