import { useMemo, useCallback } from 'react';
import {
  CompressionHistory,
  PipelineStep,
  CompressionStats,
  UseCompressionPipelineReturn,
} from '../types';

const filterActiveSteps = (
  steps: PipelineStep[],
  originalParams: number
): PipelineStep[] => {
  return steps.filter((step, index) => {
    const inputDim = index === 0
      ? originalParams
      : steps[index - 1].output_space_params;
    const outputDim = step.output_space_params;
    const isNoneStep = step.name.toLowerCase().includes('none') ||
                       step.name.toLowerCase() === 'nonecompressionstep';
    if (isNoneStep) return false;
    const hasCompressionInfo = step.compression_info &&
      (step.compression_info.compressed_params?.length > 0 ||
       step.compression_info.avg_compression_ratio !== undefined);
    const isProjectionStep = step.name.toLowerCase().includes('projection') ||
                             step.name.toLowerCase().includes('transformative') ||
                             step.type.toLowerCase().includes('projection');

    const isUselessProjection = isProjectionStep && inputDim === outputDim && !hasCompressionInfo;
    if (isUselessProjection) return false;
    return outputDim > 0;
  });
};

export const useCompressionPipeline = (
  data: CompressionHistory | null
): UseCompressionPipelineReturn => {
  const event = useMemo(() => {
    if (!data?.history?.length) return null;
    return data.history[0];
  }, [data]);
  const pipeline = useMemo(() => event?.pipeline ?? null, [event]);
  const activeSteps = useMemo(() => {
    if (!pipeline?.steps || !event) return [];
    return filterActiveSteps(pipeline.steps, event.spaces.original.n_parameters);
  }, [pipeline, event]);
  const hasStepType = useCallback((typePrefix: string): boolean => {
    return activeSteps.some(step =>
      step.type.toLowerCase().includes(typePrefix.toLowerCase())
    );
  }, [activeSteps]);
  const hasSHAP = useMemo(() => {
    return hasStepType('SHAP') &&
           activeSteps.some(step => step.calculator?.includes('SHAP'));
  }, [hasStepType, activeSteps]);

  const hasCorrelation = useMemo(() => hasStepType('Correlation'), [hasStepType]);
  const hasAdaptive = useMemo(() => hasStepType('Adaptive'), [hasStepType]);

  const hasImportanceBasedDimension = useMemo(() => {
    return hasSHAP || hasCorrelation || hasAdaptive;
  }, [hasSHAP, hasCorrelation, hasAdaptive]);
  const dimensionStepWithCalculator = useMemo(() => {
    return activeSteps.find(
      step => step.name === 'dimension_selection' && step.calculator
    );
  }, [activeSteps]);
  const getCompressionStats = useCallback((): CompressionStats | null => {
    if (!event) return null;

    const originalDim = event.spaces.original.n_parameters;
    const finalDim = event.spaces.surrogate.n_parameters;
    const dimensionFlow = [
      originalDim,
      ...activeSteps.map(s => s.output_space_params),
    ];

    return {
      originalDim,
      finalDim,
      ratio: event.compression_ratios.surrogate_to_original,
      stepCount: activeSteps.length,
      dimensionFlow,
    };
  }, [event, activeSteps]);

  return {
    event,
    pipeline,
    activeSteps,
    hasStepType,
    getCompressionStats,
    hasSHAP,
    hasCorrelation,
    hasAdaptive,
    hasImportanceBasedDimension,
    dimensionStepWithCalculator,
  };
};

export default useCompressionPipeline;
