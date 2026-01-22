import { useMemo } from 'react';
import {
  CompressionHistory,
  ChartMockData,
  UseChartVisibilityReturn,
  RangeCompressionStepInfo,
} from '../types';
import { useCompressionPipeline } from './useCompressionPipeline';

export const useChartVisibility = (
  data: CompressionHistory | null
): UseChartVisibilityReturn => {
  const {
    event,
    activeSteps,
    hasImportanceBasedDimension,
    hasAdaptive,
    dimensionStepWithCalculator,
  } = useCompressionPipeline(data);
  const hasAdaptiveUpdateHistory = useMemo(() => {
    if (!data?.history) return false;
    return data.history.length > 1 &&
           data.history.some(e => e.event === 'adaptive_update');
  }, [data]);
  const hasMultiTaskData = useMemo(() => {
    return event?.performance_metrics?.multi_task_importances !== undefined;
  }, [event]);
  const hasTransferLearningData = useMemo(() => {
    return event?.performance_metrics?.source_similarities !== undefined;
  }, [event]);
  const rangeCompressionSteps = useMemo((): RangeCompressionStepInfo[] => {
    return activeSteps
      .map((step, index) => ({ step, index }))
      .filter(({ step }) =>
        step.compression_info &&
        step.compression_info.compressed_params &&
        step.compression_info.compressed_params.length > 0
      );
  }, [activeSteps]);
  const visibility = useMemo(() => ({
    showParameterImportance: hasImportanceBasedDimension && !!dimensionStepWithCalculator,
    showDimensionEvolution: hasAdaptive && hasAdaptiveUpdateHistory,
    showMultiTaskHeatmap: hasMultiTaskData,
    showSourceSimilarities: hasTransferLearningData,
    showRangeCompression: rangeCompressionSteps.length > 0,
  }), [
    hasImportanceBasedDimension,
    dimensionStepWithCalculator,
    hasAdaptive,
    hasAdaptiveUpdateHistory,
    hasMultiTaskData,
    hasTransferLearningData,
    rangeCompressionSteps.length,
  ]);
  const chartData = useMemo((): ChartMockData => {
    if (!event) {
      return {
        paramImportances: null,
        iterations: null,
        dimensions: null,
        multiTaskImportances: null,
        taskNames: null,
        sourceSimilarities: null,
      };
    }
    const paramImportances = visibility.showParameterImportance
      ? event.spaces.original.parameters.map((_, idx) => 0.1 + Math.random() * 0.9)
      : null;
    const iterations = visibility.showDimensionEvolution
      ? [0, 10, 20, 30, 40, 50]
      : null;
    const dimensions = visibility.showDimensionEvolution
      ? [12, 12, 10, 8, 8, 6]
      : null;
    const multiTaskImportances = hasMultiTaskData
      ? event.performance_metrics!.multi_task_importances!
      : null;

    const taskNames = event.performance_metrics?.task_names ?? null;

    const sourceSimilarities = hasTransferLearningData
      ? event.performance_metrics!.source_similarities!
      : null;

    return {
      paramImportances,
      iterations,
      dimensions,
      multiTaskImportances,
      taskNames,
      sourceSimilarities,
    };
  }, [event, visibility, hasMultiTaskData, hasTransferLearningData]);

  return {
    ...visibility,
    chartData,
    rangeCompressionSteps,
  };
};

export default useChartVisibility;
