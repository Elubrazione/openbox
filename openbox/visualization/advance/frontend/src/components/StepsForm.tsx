import React, { useState } from 'react';

interface FillingConfig {
  type: string;
  fixed_values?: Record<string, any>;
}

interface ConfigSpaceParameter {
  name: string;
  type: string;
  min?: number;
  max?: number;
  default?: any;
  choices?: any[];
}

type ExpertRangeValue = {
  min?: number | '';
  max?: number | '';
};

export interface StepsConfig {
  dimension_step: string;
  range_step: string;
  projection_step: string;
  dimension_params?: Record<string, any>;
  range_params?: Record<string, any>;
  projection_params?: Record<string, any>;
  filling_config?: FillingConfig;
}

interface StepParameter {
  name: string;
  label: string;
  type: 'number' | 'text' | 'select' | 'multiselect';
  default: any;
  options?: { value: any; label: string }[];
  min?: number;
  max?: number;
  step?: number;
  description?: string;
  shouldDisplay?: (steps: StepsConfig) => boolean;
}

interface StepsFormProps {
  onNext: (steps: StepsConfig) => void;
  onBack: () => void;
  initialData?: StepsConfig;
  configSpace?: Record<string, any>;
}

const StepsForm: React.FC<StepsFormProps> = ({ onNext, onBack, initialData, configSpace }) => {
  // Calculate the number of dimensions in config space
  const maxDimensions = configSpace ? Object.keys(configSpace).length : undefined;

  const getDefaultSteps = (): StepsConfig => ({
    dimension_step: 'd_shap',
    range_step: 'r_boundary',
    projection_step: 'p_none',
    dimension_params: {
      topk: 20,
      exclude_params: [],
      expert_params: [],
      similarity_method: 'shap',
      importance_calculator: 'shap',
    },
    range_params: { top_ratio: 0.8, sigma: 2.0, enable_mixed_sampling: true, initial_prob: 0.9 },
    projection_params: {},
    filling_config: { type: 'default', fixed_values: {} },
  });

  const [steps, setSteps] = useState<StepsConfig>(() => {
    if (!initialData) {
      return getDefaultSteps();
    }
    const defaults = getDefaultSteps();
    const merged: StepsConfig = {
      ...defaults,
      ...initialData,
      dimension_params: { ...defaults.dimension_params, ...initialData.dimension_params },
      range_params: { ...defaults.range_params, ...initialData.range_params },
      projection_params: { ...defaults.projection_params, ...initialData.projection_params },
      filling_config: initialData.filling_config
        ? {
          type: initialData.filling_config.type || 'default',
          fixed_values: initialData.filling_config.fixed_values || {},
        }
        : defaults.filling_config,
    };
    if (merged.dimension_params) {
      const simValue =
        merged.dimension_params.similarity_method ||
        merged.dimension_params.importance_calculator ||
        defaults.dimension_params?.similarity_method;
      merged.dimension_params.similarity_method = simValue;
      merged.dimension_params.importance_calculator = simValue;
    }
    return merged;
  });

  // Step descriptions mapping
  const dimensionSteps = [
    { value: 'd_shap', label: 'SHAP-based Selection', description: 'Select important dimensions using SHAP values' },
    { value: 'd_corr', label: 'Correlation-based Selection', description: 'Select dimensions based on correlation analysis' },
    { value: 'd_expert', label: 'Expert-defined Selection', description: 'Use expert knowledge for dimension selection' },
    { value: 'd_adaptive', label: 'Adaptive Selection', description: 'Dynamic dimension selection with topk adjustment' },
    { value: 'd_none', label: 'No Dimension Selection', description: 'Skip dimension selection step' },
  ];

  const rangeSteps = [
    { value: 'r_boundary', label: 'Boundary-based Range', description: 'Simple boundary analysis for range compression' },
    { value: 'r_shap', label: 'SHAP-weighted Boundary', description: 'Use SHAP values to weight boundary compression' },
    { value: 'r_kde', label: 'KDE-based Range', description: 'Kernel density estimation for range compression' },
    { value: 'r_expert', label: 'Expert-specified Range', description: 'Use expert-defined range compression' },
    { value: 'r_none', label: 'No Range Compression', description: 'Skip range compression step' },
  ];

  const projectionSteps = [
    { value: 'p_quant', label: 'Quantization Projection', description: 'Discretize continuous parameters' },
    { value: 'p_rembo', label: 'REMBO Projection', description: 'Random EMbedding Bayesian Optimization' },
    { value: 'p_hesbo', label: 'HeSBO Projection', description: 'Heterogeneous and Structured BO' },
    { value: 'p_kpca', label: 'Kernel PCA Projection', description: 'Non-linear dimensionality reduction' },
    { value: 'p_none', label: 'No Projection', description: 'Skip projection step' },
  ];

  const paramDefinitions: ConfigSpaceParameter[] = configSpace
    ? Object.entries(configSpace).map(([name, def]: [string, any]) => ({
      name,
      type: def.type,
      min: def.min,
      max: def.max,
      default: def.default,
      choices: def.choices,
    }))
    : [];

  const numericParamDefinitions = paramDefinitions.filter((param) => {
    const paramType = (param.type || '').toLowerCase();
    return paramType === 'integer' || paramType === 'int' || paramType === 'float' || paramType === 'real';
  });

  const parameterNames = paramDefinitions.map((param) => param.name);
  const currentFillingConfig: FillingConfig = steps.filling_config || { type: 'default', fixed_values: {} };

  // Define parameters for each step type
  const stepParameters: Record<string, StepParameter[]> = {
    'd_shap': [
      { name: 'topk', label: 'Top K', type: 'number', default: 20, min: 1, max: maxDimensions, description: maxDimensions ? `Number of top dimensions to select (max: ${maxDimensions})` : 'Number of top dimensions to select' },
    ],
    'd_corr': [
      { name: 'topk', label: 'Top K', type: 'number', default: 20, min: 1, max: maxDimensions, description: maxDimensions ? `Number of top dimensions to select (max: ${maxDimensions})` : 'Number of top dimensions to select' },
      {
        name: 'method', label: 'Method', type: 'select', default: 'spearman', options: [
          { value: 'spearman', label: 'Spearman' },
          { value: 'pearson', label: 'Pearson' },
        ], description: 'Correlation method'
      },
    ],
    'd_expert': [],
    'd_adaptive': [
      { name: 'initial_topk', label: 'Initial Top K', type: 'number', default: 30, min: 1, max: maxDimensions, description: maxDimensions ? `Initial number of dimensions (max: ${maxDimensions})` : 'Initial number of dimensions' },
      { name: 'reduction_ratio', label: 'Reduction Ratio', type: 'number', default: 0.2, min: 0, max: 1, step: 0.1, description: 'Ratio for dimension reduction' },
      { name: 'min_dimensions', label: 'Min Dimensions', type: 'number', default: 5, min: 1, max: maxDimensions, description: maxDimensions ? `Minimum dimensions to keep (max: ${maxDimensions})` : 'Minimum dimensions to keep' },
      { name: 'max_dimensions', label: 'Max Dimensions (optional)', type: 'number', default: maxDimensions ?? 30, min: 1, max: maxDimensions, description: 'Upper bound for adaptive selection' },
      {
        name: 'similarity_method', label: 'Similarity Method (optional)', type: 'select', default: 'shap', options: [
          { value: 'shap', label: 'SHAP (默认)' },
          { value: 'correlation', label: '相关系数（自定义方法）' },
          { value: 'correlation_spearman', label: '相关系数（Spearman）' },
          { value: 'correlation_pearson', label: '相关系数（Pearson）' },
        ], description: '选择自适应维度的相似度/重要性计算方式'
      },
      {
        name: 'correlation_method', label: 'Correlation Method', type: 'select', default: 'spearman', options: [
          { value: 'spearman', label: 'Spearman' },
          { value: 'pearson', label: 'Pearson' },
        ], description: '用于相关性重要性计算的公式', shouldDisplay: (currentSteps) => currentSteps.dimension_params?.similarity_method === 'correlation'
      },
      {
        name: 'update_strategy', label: 'Update Strategy', type: 'select', default: 'periodic', options: [
          { value: 'periodic', label: 'Periodic' },
          { value: 'stagnation', label: 'Stagnation' },
          { value: 'improvement', label: 'Improvement' },
          { value: 'hybrid', label: 'Hybrid' },
          { value: 'composite', label: 'Composite' },
          { value: 'none', label: 'None' },
        ], description: '自适应降维的更新策略'
      },
      {
        name: 'period', label: 'Update Period', type: 'number', default: 5, min: 1, shouldDisplay: (currentSteps) => {
          const strategy = currentSteps.dimension_params?.update_strategy;
          return strategy === 'periodic' || strategy === 'hybrid';
        }, description: '多少轮后尝试更新一次 topk'
      },
      {
        name: 'stagnation_threshold', label: 'Stagnation Threshold', type: 'number', default: 5, min: 1, shouldDisplay: (currentSteps) => {
          const strategy = currentSteps.dimension_params?.update_strategy;
          return strategy === 'stagnation' || strategy === 'hybrid';
        }, description: '停滞阈值（迭代次数）'
      },
      {
        name: 'improvement_threshold', label: 'Improvement Threshold', type: 'number', default: 3, min: 1, shouldDisplay: (currentSteps) => {
          const strategy = currentSteps.dimension_params?.update_strategy;
          return strategy === 'improvement' || strategy === 'hybrid';
        }, description: '需要多少次提升才触发更新'
      },
      {
        name: 'composite_strategies', label: 'Composite Strategies', type: 'multiselect', default: [], options: [
          { value: 'periodic', label: 'Periodic' },
          { value: 'stagnation', label: 'Stagnation' },
          { value: 'improvement', label: 'Improvement' },
          { value: 'hybrid', label: 'Hybrid' },
        ], shouldDisplay: (currentSteps) => currentSteps.dimension_params?.update_strategy === 'composite', description: '选择组合策略中包含的步骤'
      },
    ],
    'r_boundary': [
      { name: 'top_ratio', label: 'Top Ratio', type: 'number', default: 0.8, min: 0, max: 1, step: 0.1, description: 'Ratio of top observations to consider' },
      { name: 'sigma', label: 'Sigma', type: 'number', default: 2.0, min: 0, step: 0.1, description: 'Sigma for boundary calculation' },
      {
        name: 'enable_mixed_sampling', label: 'Enable Mixed Sampling', type: 'select', default: 'true', options: [
          { value: 'true', label: 'True' },
          { value: 'false', label: 'False' },
        ], description: 'Enable mixed sampling strategy'
      },
      { name: 'initial_prob', label: 'Initial Probability', type: 'number', default: 0.9, min: 0, max: 1, step: 0.1, description: 'Initial sampling probability' },
    ],
    'r_shap': [
      { name: 'top_ratio', label: 'Top Ratio', type: 'number', default: 0.8, min: 0, max: 1, step: 0.1, description: 'Ratio of top observations to consider' },
      { name: 'sigma', label: 'Sigma', type: 'number', default: 2.0, min: 0, step: 0.1, description: 'Sigma for boundary calculation' },
      {
        name: 'enable_mixed_sampling', label: 'Enable Mixed Sampling', type: 'select', default: 'true', options: [
          { value: 'true', label: 'True' },
          { value: 'false', label: 'False' },
        ], description: 'Enable mixed sampling strategy'
      },
      { name: 'initial_prob', label: 'Initial Probability', type: 'number', default: 0.9, min: 0, max: 1, step: 0.1, description: 'Initial sampling probability' },
    ],
    'r_kde': [
      { name: 'source_top_ratio', label: 'Source Top Ratio', type: 'number', default: 0.3, min: 0, max: 1, step: 0.1, description: 'Ratio of top source observations' },
      { name: 'kde_coverage', label: 'KDE Coverage', type: 'number', default: 0.6, min: 0, max: 1, step: 0.1, description: 'KDE coverage ratio' },
      {
        name: 'enable_mixed_sampling', label: 'Enable Mixed Sampling', type: 'select', default: 'true', options: [
          { value: 'true', label: 'True' },
          { value: 'false', label: 'False' },
        ], description: 'Enable mixed sampling strategy'
      },
      { name: 'initial_prob', label: 'Initial Probability', type: 'number', default: 0.9, min: 0, max: 1, step: 0.1, description: 'Initial sampling probability' },
    ],
    'r_expert': [
      {
        name: 'enable_mixed_sampling', label: 'Enable Mixed Sampling', type: 'select', default: 'false', options: [
          { value: 'true', label: 'True' },
          { value: 'false', label: 'False' },
        ], description: 'Enable mixed sampling strategy'
      },
      { name: 'initial_prob', label: 'Initial Probability', type: 'number', default: 0.9, min: 0, max: 1, step: 0.1, description: 'Initial sampling probability for mixed strategy' },
    ],
    'p_quant': [
      { name: 'max_num_values', label: 'Max Number of Values', type: 'number', default: 10, min: 2, description: 'Maximum number of discrete values' },
      {
        name: 'adaptive', label: 'Adaptive', type: 'select', default: 'false', options: [
          { value: 'true', label: 'True' },
          { value: 'false', label: 'False' },
        ], description: 'Use adaptive quantization'
      },
    ],
    'p_rembo': [
      { name: 'low_dim', label: 'Low Dimension', type: 'number', default: 10, min: 1, description: 'Target low dimension' },
    ],
    'p_hesbo': [
      { name: 'low_dim', label: 'Low Dimension', type: 'number', default: 10, min: 1, description: 'Target low dimension' },
      { name: 'max_num_values', label: 'Max Number of Values', type: 'number', default: 10, min: 2, description: 'Maximum number of discrete values' },
    ],
    'p_kpca': [
      { name: 'n_components', label: 'Number of Components', type: 'number', default: 10, min: 1, description: 'Number of principal components' },
      {
        name: 'kernel', label: 'Kernel', type: 'select', default: 'rbf', options: [
          { value: 'rbf', label: 'RBF' },
          { value: 'linear', label: 'Linear' },
          { value: 'poly', label: 'Polynomial' },
          { value: 'sigmoid', label: 'Sigmoid' },
        ], description: 'Kernel type'
      },
    ],
  };

  const getParameterDefinition = (paramName: string): any => {
    if (!configSpace) {
      return undefined;
    }
    return configSpace[paramName];
  };

  const getDefaultFixedValue = (paramName: string) => {
    const definition = getParameterDefinition(paramName);
    if (!definition) {
      return '';
    }
    const type = (definition.type || '').toLowerCase();
    if (definition.default !== undefined) {
      return definition.default;
    }
    if (definition.min !== undefined && definition.max !== undefined &&
      (type === 'integer' || type === 'int' || type === 'float' || type === 'real')) {
      const minVal = Number(definition.min);
      const maxVal = Number(definition.max);
      if (!Number.isNaN(minVal) && !Number.isNaN(maxVal)) {
        return (minVal + maxVal) / 2;
      }
    }
    if (type === 'categorical' && Array.isArray(definition.choices) && definition.choices.length > 0) {
      return definition.choices[0];
    }
    return '';
  };

  const updateFillingConfig = (updater: (cfg: FillingConfig) => FillingConfig) => {
    setSteps((prev) => {
      const nextConfig = updater(prev.filling_config || { type: 'default', fixed_values: {} });
      return { ...prev, filling_config: nextConfig };
    });
  };

  const normalizeParamList = (value: any): string[] => {
    if (Array.isArray(value)) {
      return value;
    }
    if (typeof value === 'string') {
      return value
        .split(',')
        .map((item) => item.trim())
        .filter((item) => item.length > 0);
    }
    return [];
  };

  const toggleDimensionParamList = (field: 'exclude_params' | 'expert_params', paramName: string) => {
    if (!parameterNames.includes(paramName)) {
      return;
    }
    const currentList: string[] = normalizeParamList(steps.dimension_params?.[field]);
    const exists = currentList.includes(paramName);
    const nextList = exists ? currentList.filter((name) => name !== paramName) : [...currentList, paramName];
    updateStepParam('dimension_params', field, nextList);
  };

  const handleToggleFixedParam = (paramName: string) => {
    if (!configSpace) {
      return;
    }
    updateFillingConfig((cfg) => {
      const nextValues = { ...(cfg.fixed_values || {}) };
      if (paramName in nextValues) {
        delete nextValues[paramName];
      } else {
        nextValues[paramName] = getDefaultFixedValue(paramName);
      }
      return { ...cfg, fixed_values: nextValues };
    });
  };

  const handleFixedValueChange = (paramName: string, rawValue: string) => {
    const definition = getParameterDefinition(paramName);
    if (!definition) {
      return;
    }
    const type = (definition.type || '').toLowerCase();
    let parsed: any = rawValue;
    if (rawValue === '') {
      parsed = '';
    } else if (type === 'integer' || type === 'int') {
      parsed = parseInt(rawValue, 10);
    } else if (type === 'float' || type === 'real') {
      parsed = parseFloat(rawValue);
    }
    updateFillingConfig((cfg) => ({
      ...cfg,
      fixed_values: {
        ...(cfg.fixed_values || {}),
        [paramName]: parsed,
      },
    }));
  };

  const getExpertRanges = (): Record<string, ExpertRangeValue> => {
    const ranges = steps.range_params?.expert_ranges as Record<string, ExpertRangeValue> | undefined;
    return ranges ? { ...ranges } : {};
  };

  const handleExpertRangeToggle = (paramName: string) => {
    if (!numericParamDefinitions.find((param) => param.name === paramName)) {
      return;
    }
    setSteps((prev) => {
      const currentRanges = { ...(prev.range_params?.expert_ranges || {}) } as Record<string, ExpertRangeValue>;
      if (currentRanges[paramName]) {
        delete currentRanges[paramName];
      } else {
        const definition = getParameterDefinition(paramName);
        const minVal = definition?.min !== undefined ? Number(definition.min) : 0;
        const maxVal = definition?.max !== undefined ? Number(definition.max) : minVal;
        currentRanges[paramName] = {
          min: Number.isNaN(minVal) ? '' : minVal,
          max: Number.isNaN(maxVal) ? '' : maxVal,
        };
      }
      return {
        ...prev,
        range_params: {
          ...prev.range_params,
          expert_ranges: currentRanges,
        },
      };
    });
  };

  const handleExpertRangeChange = (paramName: string, field: 'min' | 'max', rawValue: string) => {
    setSteps((prev) => {
      const currentRanges = { ...(prev.range_params?.expert_ranges || {}) } as Record<string, ExpertRangeValue>;
      const existing = currentRanges[paramName] ? { ...currentRanges[paramName] } : {};
      if (rawValue === '') {
        existing[field] = '';
      } else {
        const parsed = parseFloat(rawValue);
        existing[field] = Number.isNaN(parsed) ? existing[field] : parsed;
      }
      currentRanges[paramName] = existing;
      return {
        ...prev,
        range_params: {
          ...prev.range_params,
          expert_ranges: currentRanges,
        },
      };
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Validate topk values against max dimensions
    if (maxDimensions) {
      const dimensionParams = steps.dimension_params || {};

      if (steps.dimension_step === 'd_shap' || steps.dimension_step === 'd_corr') {
        const topk = dimensionParams.topk;
        if (topk && topk > maxDimensions) {
          alert(`Top K (${topk}) cannot exceed the number of dimensions in config space (${maxDimensions})`);
          return;
        }
      }

      if (steps.dimension_step === 'd_adaptive') {
        const initialTopk = dimensionParams.initial_topk;
        const minDimensions = dimensionParams.min_dimensions;
        const maxDimSetting = dimensionParams.max_dimensions;

        if (initialTopk && initialTopk > maxDimensions) {
          alert(`Initial Top K (${initialTopk}) cannot exceed the number of dimensions in config space (${maxDimensions})`);
          return;
        }

        if (minDimensions && minDimensions > maxDimensions) {
          alert(`Min Dimensions (${minDimensions}) cannot exceed the number of dimensions in config space (${maxDimensions})`);
          return;
        }

        if (maxDimSetting && maxDimSetting > maxDimensions) {
          alert(`Max Dimensions (${maxDimSetting}) cannot exceed the number of dimensions in config space (${maxDimensions})`);
          return;
        }
      }
    }

    if (steps.dimension_step === 'd_adaptive') {
      const dimensionParams = steps.dimension_params || {};
      const minDimensions = dimensionParams.min_dimensions;
      const maxDimSetting = dimensionParams.max_dimensions;
      if (minDimensions && maxDimSetting && maxDimSetting < minDimensions) {
        alert('Max Dimensions cannot be smaller than Min Dimensions');
        return;
      }
    }

    if (steps.range_step === 'r_expert') {
      const ranges = steps.range_params?.expert_ranges as Record<string, ExpertRangeValue> | undefined;
      if (ranges) {
        for (const [paramName, rangeValue] of Object.entries(ranges)) {
          if (!rangeValue) {
            continue;
          }
          const minVal = rangeValue.min;
          const maxVal = rangeValue.max;
          if (minVal === undefined || minVal === '' || maxVal === undefined || maxVal === '') {
            alert(`Please complete both min and max values for expert range "${paramName}"`);
            return;
          }
          const minNum = typeof minVal === 'number' ? minVal : parseFloat(String(minVal));
          const maxNum = typeof maxVal === 'number' ? maxVal : parseFloat(String(maxVal));
          if (Number.isNaN(minNum) || Number.isNaN(maxNum)) {
            alert(`Invalid expert range values for "${paramName}"`);
            return;
          }
          if (minNum >= maxNum) {
            alert(`Expert range for "${paramName}" requires min < max`);
            return;
          }
        }
      }
    }

    const fixedValues = currentFillingConfig.fixed_values || {};
    for (const [paramName, value] of Object.entries(fixedValues)) {
      if (value === undefined || value === null || value === '') {
        alert(`请为固定参数 "${paramName}" 输入有效的取值`);
        return;
      }
    }

    onNext(steps);
  };

  const updateStep = (field: keyof StepsConfig, value: string) => {
    const newSteps = { ...steps, [field]: value };

    // Initialize default parameters for the selected step
    const paramField = field === 'dimension_step' ? 'dimension_params' :
      field === 'range_step' ? 'range_params' : 'projection_params';

    const previousParams = steps[paramField] || {};

    // If the step is '*_none', clear the parameters but keep reusable selections
    if (value.endsWith('_none')) {
      if (field === 'dimension_step') {
        newSteps[paramField] = {
          exclude_params: previousParams.exclude_params || [],
        };
      } else {
        newSteps[paramField] = {};
      }
    } else {
      const params = stepParameters[value] || [];
      const defaultParams: Record<string, any> = {};
      params.forEach((param) => {
        defaultParams[param.name] = param.default;
      });

      if (field === 'dimension_step') {
        if (previousParams.exclude_params) {
          defaultParams.exclude_params = previousParams.exclude_params;
        }
        if (value === 'd_expert') {
          defaultParams.expert_params = normalizeParamList(previousParams.expert_params);
        }
        if (value === 'd_adaptive') {
          const adaptiveKeys = [
            'similarity_method',
            'importance_calculator',
            'correlation_method',
            'update_strategy',
            'period',
            'stagnation_threshold',
            'improvement_threshold',
            'composite_strategies',
            'max_dimensions',
          ];
          adaptiveKeys.forEach((key) => {
            if (previousParams[key] !== undefined) {
              defaultParams[key] = previousParams[key];
            }
          });
        }
      }

      if (field === 'range_step' && value === 'r_expert') {
        defaultParams.expert_ranges = previousParams.expert_ranges || {};
      }

      newSteps[paramField] = defaultParams;
    }

    setSteps(newSteps);
  };

  const updateStepParam = (
    stepType: 'dimension_params' | 'range_params' | 'projection_params',
    paramName: string,
    value: any
  ) => {
    if (stepType === 'dimension_params' && paramName === 'similarity_method') {
      setSteps((prev) => {
        const nextParams = { ...(prev.dimension_params || {}) };
        if (!value) {
          delete nextParams.similarity_method;
          delete nextParams.importance_calculator;
        } else {
          nextParams.similarity_method = value;
          nextParams.importance_calculator = value;
        }
        if (value !== 'correlation') {
          delete nextParams.correlation_method;
        }
        return {
          ...prev,
          dimension_params: nextParams,
        };
      });
      return;
    }

    // Validate topk related parameters against max dimensions
    if (maxDimensions && stepType === 'dimension_params' &&
      (paramName === 'topk' || paramName === 'initial_topk' || paramName === 'min_dimensions' || paramName === 'max_dimensions')) {
      const numValue = typeof value === 'number' ? value : parseFloat(value);
      if (!isNaN(numValue) && numValue > maxDimensions) {
        value = maxDimensions;
      }
    }

    setSteps((prev) => {
      const nextParams = { ...(prev[stepType] || {}) };

      if (value === '' || value === undefined || (Array.isArray(value) && value.length === 0)) {
        delete nextParams[paramName];
      } else {
        nextParams[paramName] = value;
      }

      if (stepType === 'dimension_params' && paramName === 'update_strategy' && value !== 'composite') {
        delete nextParams.composite_strategies;
      }

      return {
        ...prev,
        [stepType]: nextParams,
      };
    });
  };

  const getStepLabel = (value: string, stepList: { value: string; label: string }[]) => {
    return stepList.find(s => s.value === value)?.label || value;
  };

  const renderStepParameters = (
    stepValue: string,
    paramType: 'dimension_params' | 'range_params' | 'projection_params'
  ) => {
    const params = stepParameters[stepValue];
    if (!params || params.length === 0 || stepValue.endsWith('_none')) {
      return null;
    }

    return (
      <div className="step-parameters">
        <h4>Parameters</h4>
        {params.filter((param) => !param.shouldDisplay || param.shouldDisplay(steps)).map((param) => {
          const currentValue = steps[paramType]?.[param.name];
          return (
            <div key={param.name} className="param-field">
              <label>{param.label}</label>
              {param.type === 'number' && (
                <input
                  type="number"
                  value={currentValue === undefined ? '' : currentValue}
                  onChange={(e) => {
                    const raw = e.target.value;
                    if (raw === '') {
                      updateStepParam(paramType, param.name, '');
                      return;
                    }
                    const parsed = parseFloat(raw);
                    updateStepParam(paramType, param.name, Number.isNaN(parsed) ? '' : parsed);
                  }}
                  min={param.min}
                  max={param.max}
                  step={param.step || 1}
                />
              )}
              {param.type === 'text' && (
                <input
                  type="text"
                  value={currentValue ?? param.default ?? ''}
                  onChange={(e) => updateStepParam(paramType, param.name, e.target.value)}
                  placeholder={param.description}
                />
              )}
              {param.type === 'select' && param.options && (
                <select
                  value={
                    ((currentValue === undefined ? param.default : currentValue) === true)
                      ? 'true'
                      : ((currentValue === undefined ? param.default : currentValue) === false)
                        ? 'false'
                        : (currentValue === undefined ? param.default : currentValue)
                  }
                  onChange={(e) => {
                    const val = e.target.value === 'true'
                      ? true
                      : e.target.value === 'false'
                        ? false
                        : e.target.value;
                    updateStepParam(paramType, param.name, val);
                  }}
                >
                  {param.options.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              )}
              {param.type === 'multiselect' && param.options && (
                <select
                  multiple
                  value={currentValue ?? param.default ?? []}
                  onChange={(e) => {
                    const selected = Array.from(e.target.selectedOptions).map((opt) => opt.value);
                    updateStepParam(paramType, param.name, selected);
                  }}
                >
                  {param.options.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              )}
              {param.description && (
                <p className="param-hint">{param.description}</p>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  const renderExcludeSection = () => {
    if (!configSpace || parameterNames.length === 0 || steps.dimension_step === 'd_none' || steps.dimension_step === 'd_expert') {
      return null;
    }
    const selected = steps.dimension_params?.exclude_params || [];
    return (
      <div className="chip-section">
        <h4>手动剔除参数（可选）</h4>
        <p className="param-hint">从维度选择中排除特定参数</p>
        <div className="chip-grid">
          {parameterNames.map((name) => {
            const isSelected = selected.includes(name);
            return (
              <button
                type="button"
                key={name}
                className={`chip ${isSelected ? 'selected' : ''}`}
                onClick={() => toggleDimensionParamList('exclude_params', name)}
              >
                {name}
              </button>
            );
          })}
        </div>
      </div>
    );
  };

  const renderExpertParamsSection = () => {
    if (!configSpace || steps.dimension_step !== 'd_expert' || parameterNames.length === 0) {
      return null;
    }
    const selected = normalizeParamList(steps.dimension_params?.expert_params);
    return (
      <div className="chip-section">
        <h4>Expert 维度选择</h4>
        <p className="param-hint">选择需要保留的参数，系统将只保留这些维度</p>
        <div className="chip-grid">
          {parameterNames.map((name) => {
            const isSelected = selected.includes(name);
            return (
              <button
                type="button"
                key={name}
                className={`chip ${isSelected ? 'selected' : ''}`}
                onClick={() => toggleDimensionParamList('expert_params', name)}
              >
                {name}
              </button>
            );
          })}
        </div>
      </div>
    );
  };

  const renderExpertRangeSection = () => {
    if (!configSpace || steps.range_step !== 'r_expert') {
      return null;
    }
    if (numericParamDefinitions.length === 0) {
      return (
        <div className="range-section">
          <h4>Expert 范围设置</h4>
          <p className="param-hint">当前搜索空间没有可设置范围的数值型参数</p>
        </div>
      );
    }
    const expertRanges = steps.range_params?.expert_ranges as Record<string, ExpertRangeValue> | undefined;
    return (
      <div className="range-section">
        <h4>Expert 范围设置</h4>
        <p className="param-hint">勾选需要收缩范围的参数并填写新的上下界</p>
        <div className="range-card-grid">
          {numericParamDefinitions.map((param) => {
            const isActive = !!expertRanges?.[param.name];
            const rangeValue = expertRanges?.[param.name] || {};
            return (
              <div key={param.name} className={`range-card ${isActive ? 'active' : ''}`}>
                <div className="range-card-header">
                  <label>
                    <input
                      type="checkbox"
                      checked={isActive}
                      onChange={() => handleExpertRangeToggle(param.name)}
                    />
                    <span>{param.name}</span>
                  </label>
                  <span className="range-tag">{param.type}</span>
                </div>
                {isActive && (
                  <div className="range-inputs">
                    <div className="form-field">
                      <label>最小值</label>
                      <input
                        type="number"
                        value={rangeValue.min ?? ''}
                        onChange={(e) => handleExpertRangeChange(param.name, 'min', e.target.value)}
                        min={param.min}
                        max={param.max}
                      />
                    </div>
                    <div className="form-field">
                      <label>最大值</label>
                      <input
                        type="number"
                        value={rangeValue.max ?? ''}
                        onChange={(e) => handleExpertRangeChange(param.name, 'max', e.target.value)}
                        min={param.min}
                        max={param.max}
                      />
                    </div>
                    <p className="param-hint">
                      原始范围: [{param.min}, {param.max}]
                    </p>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const renderFixedValueCards = () => {
    const fixedValues = currentFillingConfig.fixed_values || {};
    if (!fixedValues || Object.keys(fixedValues).length === 0) {
      return <p className="param-hint">尚未选择固定参数</p>;
    }
    return (
      <div className="fixed-value-list">
        {Object.entries(fixedValues).map(([paramName, value]) => {
          const definition = getParameterDefinition(paramName);
          if (!definition) {
            return null;
          }
          const type = (definition.type || '').toLowerCase();
          const displayValue = value ?? '';
          return (
            <div key={paramName} className="fixed-value-card">
              <div className="fixed-value-card-header">
                <strong>{paramName}</strong>
                <button type="button" onClick={() => handleToggleFixedParam(paramName)} aria-label="移除固定值">
                  ×
                </button>
              </div>
              {type === 'categorical' && Array.isArray(definition.choices) ? (
                <select
                  value={displayValue}
                  onChange={(e) => handleFixedValueChange(paramName, e.target.value)}
                >
                  {definition.choices.map((choice: any) => (
                    <option key={choice} value={choice}>
                      {choice}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="number"
                  value={displayValue}
                  onChange={(e) => handleFixedValueChange(paramName, e.target.value)}
                  min={definition.min}
                  max={definition.max}
                  step={type === 'float' || type === 'real' ? '0.01' : '1'}
                />
              )}
              {definition.min !== undefined && definition.max !== undefined && (
                <p className="param-hint">
                  原始范围: [{definition.min}, {definition.max}]
                </p>
              )}
              {definition.choices && (
                <p className="param-hint">
                  可选项: {definition.choices.join(', ')}
                </p>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  const renderFixedValueSection = () => {
    if (!configSpace) {
      return (
        <div className="chip-section">
          <h4>固定采样参数</h4>
          <p className="param-hint">请先完成参数空间配置后再使用该功能</p>
        </div>
      );
    }
    return (
      <div className="chip-section">
        <h4>固定采样参数（可选）</h4>
        <p className="param-hint">当与 BO 集成时，可将部分参数固定为指定值</p>
        <div className="chip-grid">
          {parameterNames.map((name) => {
            const isSelected = !!currentFillingConfig.fixed_values?.[name];
            return (
              <button
                type="button"
                key={name}
                className={`chip ${isSelected ? 'selected' : ''}`}
                onClick={() => handleToggleFixedParam(name)}
              >
                {name}
              </button>
            );
          })}
        </div>
        {renderFixedValueCards()}
      </div>
    );
  };

  return (
    <form onSubmit={handleSubmit} className="steps-form">
      <div className="form-header">
        <h3>Step 2: Configure Compression Steps</h3>
        <p className="form-description">Select the compression pipeline configuration</p>
      </div>

      <div className="form-content">
        <div className="form-section">
          <div className="form-field">
            <label>Dimension Selection Step</label>
            <select
              value={steps.dimension_step}
              onChange={(e) => updateStep('dimension_step', e.target.value)}
              required
            >
              {dimensionSteps.map((step) => (
                <option key={step.value} value={step.value}>
                  {step.label}
                </option>
              ))}
            </select>
            <p className="field-hint">
              {dimensionSteps.find(s => s.value === steps.dimension_step)?.description}
            </p>
            {renderStepParameters(steps.dimension_step, 'dimension_params')}
            {renderExcludeSection()}
            {renderExpertParamsSection()}
          </div>

          <div className="form-field">
            <label>Range Compression Step</label>
            <select
              value={steps.range_step}
              onChange={(e) => updateStep('range_step', e.target.value)}
              required
            >
              {rangeSteps.map((step) => (
                <option key={step.value} value={step.value}>
                  {step.label}
                </option>
              ))}
            </select>
            <p className="field-hint">
              {rangeSteps.find(s => s.value === steps.range_step)?.description}
            </p>
            {renderStepParameters(steps.range_step, 'range_params')}
            {renderExpertRangeSection()}
          </div>

          <div className="form-field">
            <label>Projection Step</label>
            <select
              value={steps.projection_step}
              onChange={(e) => updateStep('projection_step', e.target.value)}
              required
            >
              {projectionSteps.map((step) => (
                <option key={step.value} value={step.value}>
                  {step.label}
                </option>
              ))}
            </select>
            <p className="field-hint">
              {projectionSteps.find(s => s.value === steps.projection_step)?.description}
            </p>
            {renderStepParameters(steps.projection_step, 'projection_params')}
          </div>
        </div>

        <div className="info-box">
          <h4>Configuration Summary</h4>
          <ul>
            <li>
              <strong>Dimension Step:</strong> {getStepLabel(steps.dimension_step, dimensionSteps)}
            </li>
            <li>
              <strong>Range Step:</strong> {getStepLabel(steps.range_step, rangeSteps)}
            </li>
            <li>
              <strong>Projection Step:</strong> {getStepLabel(steps.projection_step, projectionSteps)}
            </li>
          </ul>
        </div>

        {/* <div className="form-section">
          <div className="form-field">
            <label>Filling Strategy & Fixed Parameters</label>
            {renderFixedValueSection()}
          </div>
        </div> */}
      </div>

      <div className="form-actions">
        <button type="button" className="btn btn-secondary" onClick={onBack}>
          Back
        </button>
        <button type="submit" className="btn btn-primary">
          Next: Upload History
        </button>
      </div>
    </form>
  );
};

export default StepsForm;
