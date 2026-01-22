import { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { 
  createUploadWizardMachine, 
  WizardState, 
  stateToStep,
  UploadWizardMachine,
} from '../machines/uploadWizardMachine';
import { StepsConfig } from '../components/StepsForm';

export interface UploadResponse {
  success: boolean;
  message?: string;
  error?: string;
  compression?: {
    status: string;
    message: string;
    result?: {
      original_dim: number;
      surrogate_dim: number;
      compression_ratio: number;
    };
  };
}

export interface WizardContext {
  configSpace: Record<string, any> | null;
  stepsConfig: StepsConfig | null;
  historyFiles: File[];
  uploadResult: UploadResponse | null;
}

export interface UseUploadWizardReturn {
  state: WizardState;
  step: number;
  context: WizardContext;
  isUploading: boolean;
  next: (data?: any) => Promise<void>;
  back: () => Promise<void>;
  submit: (files: File[]) => Promise<void>;
  reset: () => void;
  retry: () => Promise<void>;
  setConfigSpace: (config: Record<string, any>) => void;
  setStepsConfig: (config: StepsConfig) => void;
}

export function useUploadWizard(options: {
  onUploadSuccess?: () => void;
} = {}): UseUploadWizardReturn {
  const { onUploadSuccess } = options;
  const machineRef = useRef<UploadWizardMachine | null>(null);
  const [state, setState] = useState<WizardState>('configSpace');
  const [context, setContext] = useState<WizardContext>({
    configSpace: null,
    stepsConfig: null,
    historyFiles: [],
    uploadResult: null,
  });
  const [isUploading, setIsUploading] = useState(false);
  useEffect(() => {
    const machine = createUploadWizardMachine('configSpace');
    machineRef.current = machine;
    const unsubEnter = machine.onEnter(({ current }) => {
      setState(current as WizardState);
    });
    
    return () => {
      unsubEnter();
    };
  }, []);
  const step = useMemo(() => stateToStep[state], [state]);
  const setConfigSpace = useCallback((config: Record<string, any>) => {
    setContext(prev => ({ ...prev, configSpace: config }));
  }, []);
  const setStepsConfig = useCallback((config: StepsConfig) => {
    setContext(prev => ({ ...prev, stepsConfig: config }));
  }, []);
  const next = useCallback(async (data?: any) => {
    const machine = machineRef.current;
    if (!machine) return;
    
    const currentState = machine.getState();
    if (currentState === 'configSpace' && data) {
      setConfigSpace(data);
    } else if (currentState === 'stepsConfig' && data) {
      setStepsConfig(data);
    }
    
    await machine.next();
  }, [setConfigSpace, setStepsConfig]);
  const back = useCallback(async () => {
    const machine = machineRef.current;
    if (!machine) return;
    await machine.back();
  }, []);
  const performUpload = useCallback(async (files: File[]) => {
    const { configSpace, stepsConfig } = context;
    
    if (!configSpace || !stepsConfig) {
      throw new Error('Missing configuration data');
    }
    
    const formData = new FormData();
    const configSpaceBlob = new Blob([JSON.stringify(configSpace, null, 2)], {
      type: 'application/json',
    });
    formData.append('config_space', configSpaceBlob, 'config_space.json');
    const convertStepParams = (params: Record<string, any>): Record<string, any> => {
      const converted = { ...params };
      
      if ('expert_params' in converted) {
        if (typeof converted.expert_params === 'string') {
          converted.expert_params = converted.expert_params
            .split(',')
            .map((s: string) => s.trim())
            .filter((s: string) => s.length > 0);
        } else if (!Array.isArray(converted.expert_params)) {
          converted.expert_params = [];
        }
      }
      
      if ('expert_ranges' in converted && converted.expert_ranges) {
        if (typeof converted.expert_ranges === 'string') {
          try {
            converted.expert_ranges = JSON.parse(converted.expert_ranges);
          } catch {
            converted.expert_ranges = {};
          }
        }
        
        if (typeof converted.expert_ranges === 'object' && !Array.isArray(converted.expert_ranges)) {
          const normalizedRanges: Record<string, [number, number]> = {};
          Object.entries(converted.expert_ranges as Record<string, any>).forEach(([paramName, rangeValue]) => {
            if (!rangeValue) return;
            let minVal: any, maxVal: any;
            if (Array.isArray(rangeValue)) {
              [minVal, maxVal] = rangeValue;
            } else if (typeof rangeValue === 'object') {
              minVal = (rangeValue as any).min ?? rangeValue[0];
              maxVal = (rangeValue as any).max ?? rangeValue[1];
            }
            if (minVal === undefined || maxVal === undefined) return;
            const minNum = typeof minVal === 'number' ? minVal : parseFloat(String(minVal));
            const maxNum = typeof maxVal === 'number' ? maxVal : parseFloat(String(maxVal));
            if (Number.isNaN(minNum) || Number.isNaN(maxNum)) return;
            normalizedRanges[paramName] = [minNum, maxNum];
          });
          converted.expert_ranges = normalizedRanges;
        }
      }
      
      for (const key in converted) {
        if (converted[key] === 'true') converted[key] = true;
        else if (converted[key] === 'false') converted[key] = false;
      }
      
      if ('similarity_method' in converted) {
        if (!converted.importance_calculator && converted.similarity_method) {
          converted.importance_calculator = converted.similarity_method;
        }
        delete converted.similarity_method;
      }
      
      return converted;
    };
    
    const step_params: Record<string, any> = {};
    
    if (stepsConfig.dimension_params && Object.keys(stepsConfig.dimension_params).length > 0) {
      step_params[stepsConfig.dimension_step] = convertStepParams(stepsConfig.dimension_params);
    }
    if (stepsConfig.range_params && Object.keys(stepsConfig.range_params).length > 0) {
      step_params[stepsConfig.range_step] = convertStepParams(stepsConfig.range_params);
    }
    if (stepsConfig.projection_params && Object.keys(stepsConfig.projection_params).length > 0) {
      step_params[stepsConfig.projection_step] = convertStepParams(stepsConfig.projection_params);
    }
    
    const stepsData = {
      dimension_step: stepsConfig.dimension_step,
      range_step: stepsConfig.range_step,
      projection_step: stepsConfig.projection_step,
      step_params,
      ...(stepsConfig.filling_config?.fixed_values &&
        Object.keys(stepsConfig.filling_config.fixed_values).length > 0
        ? { filling_config: stepsConfig.filling_config }
        : {}),
    };
    
    const stepsBlob = new Blob([JSON.stringify(stepsData, null, 2)], {
      type: 'application/json',
    });
    formData.append('steps', stepsBlob, 'steps.json');
    
    files.forEach((file) => {
      formData.append('history', file);
    });
    
    const response = await fetch('/api/upload', {
      method: 'POST',
      body: formData,
    });
    
    return response.json();
  }, [context]);
  const submit = useCallback(async (files: File[]) => {
    const machine = machineRef.current;
    if (!machine) return;
    
    setContext(prev => ({ ...prev, historyFiles: files }));
    await machine.submit();
    setIsUploading(true);
    
    try {
      const result = await performUpload(files);
      setContext(prev => ({ ...prev, uploadResult: result }));
      
      if (result.success) {
        await machine.uploadSuccess();
        setTimeout(() => {
          onUploadSuccess?.();
        }, 2000);
      } else {
        await machine.uploadError();
      }
    } catch (error) {
      setContext(prev => ({
        ...prev,
        uploadResult: {
          success: false,
          error: error instanceof Error ? error.message : 'Upload failed',
        },
      }));
      await machine.uploadError();
    } finally {
      setIsUploading(false);
    }
  }, [performUpload, onUploadSuccess]);
  const reset = useCallback(() => {
    const machine = machineRef.current;
    if (!machine) return;
    
    machine.reset();
    setContext({
      configSpace: null,
      stepsConfig: null,
      historyFiles: [],
      uploadResult: null,
    });
    setIsUploading(false);
  }, []);
  const retry = useCallback(async () => {
    const machine = machineRef.current;
    if (!machine) return;
    
    setContext(prev => ({ ...prev, uploadResult: null }));
    await machine.retry();
  }, []);
  
  return {
    state,
    step,
    context,
    isUploading,
    next,
    back,
    submit,
    reset,
    retry,
    setConfigSpace,
    setStepsConfig,
  };
}

export default useUploadWizard;
