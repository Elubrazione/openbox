import { useMemo, useRef } from 'react';
import type { EChartsOption } from 'echarts';
import type { 
  ChartData, 
  ChartConfigContext,
  UseChartConfigParams,
  UseChartConfigReturn,
  ChartDimensions,
  BarChartData,
  LineChartData,
  HeatmapChartData,
} from '../charts/types';
import { 
  getChartFactory, 
  createChartConfig,
  createCompressionRatioStrategy,
  createDimensionEvolutionStrategy,
  createMultiTaskHeatmapStrategy,
} from '../charts/factory';
import { chartHeights } from '../charts/config';

// ============================================
// ============================================

export function useChartConfig<T extends ChartData>({
  type,
  data,
  options = {},
  deps = [],
}: UseChartConfigParams<T>): UseChartConfigReturn {
  const renderCount = useRef(0);
  renderCount.current++;
  
  const isValid = useMemo(() => {
    if (!data) return false;
    const strategy = getChartFactory().getStrategy(type);
    return strategy?.validateData(data) ?? false;
  }, [type, data]);
  
  const option = useMemo<EChartsOption>(() => {
    if (!isValid) return {};
    return createChartConfig(type, data, options as Partial<ChartConfigContext>);
  }, [type, data, isValid, ...deps, JSON.stringify(options)]);
  
  const dimensions = useMemo<ChartDimensions>(() => ({
    height: options.dimensions?.height ?? chartHeights.medium,
    width: options.dimensions?.width,
  }), [options.dimensions?.height, options.dimensions?.width]);
  
  const error = useMemo(() => {
    if (!data) return 'No data provided';
    if (!isValid) return `Invalid data for chart type: ${type}`;
    return null;
  }, [data, isValid, type]);
  
  return { option, dimensions, isValid, error };
}

// ============================================
// ============================================

export function useBarChartConfig(
  data: BarChartData | null,
  options?: Partial<ChartConfigContext>
): UseChartConfigReturn {
  return useChartConfig({
    type: 'bar',
    data: data as BarChartData,
    options,
    deps: [data?.categories?.length, data?.values?.length],
  });
}

export function useHorizontalBarConfig(
  data: BarChartData | null,
  options?: Partial<ChartConfigContext>
): UseChartConfigReturn {
  return useChartConfig({
    type: 'horizontalBar',
    data: data as BarChartData,
    options,
    deps: [data?.categories?.length, data?.values?.length],
  });
}

export function useLineChartConfig(
  data: LineChartData | null,
  options?: Partial<ChartConfigContext>
): UseChartConfigReturn {
  return useChartConfig({
    type: 'line',
    data: data as LineChartData,
    options,
    deps: [data?.xData?.length, data?.yData?.length],
  });
}

export function useHeatmapConfig(
  data: HeatmapChartData | null,
  options?: Partial<ChartConfigContext>
): UseChartConfigReturn {
  return useChartConfig({
    type: 'heatmap',
    data: data as HeatmapChartData,
    options,
    deps: [data?.xCategories?.length, data?.yCategories?.length],
  });
}

// ============================================
// ============================================

export function useParameterImportanceConfig(
  paramNames: string[] | null,
  importances: number[] | null,
  topK = 20
): UseChartConfigReturn {
  const processedData = useMemo<BarChartData | null>(() => {
    if (!paramNames || !importances) return null;
    
    const absImportances = importances.map(Math.abs);
    const indices = absImportances
      .map((_, idx) => idx)
      .sort((a, b) => absImportances[b] - absImportances[a])
      .slice(0, topK);
    
    return {
      categories: indices.map(i => paramNames[i].split('.').pop() || paramNames[i]),
      values: indices.map(i => absImportances[i]),
      labels: true,
    };
  }, [paramNames, importances, topK]);
  
  return useHorizontalBarConfig(processedData, {
    title: { text: `Top-${topK} Parameter Importance` },
    tooltip: {
      formatter: (params: any) => {
        const data = params[0];
        return `${data.name}<br/>Importance: ${data.value.toFixed(4)}`;
      },
    },
    dimensions: { height: chartHeights.xlarge },
  });
}

export function useDimensionReductionConfig(
  stepNames: string[] | null,
  dimensions: number[] | null
): UseChartConfigReturn {
  const data = useMemo<BarChartData | null>(() => {
    if (!stepNames || !dimensions) return null;
    return { categories: stepNames, values: dimensions, labels: true };
  }, [stepNames, dimensions]);
  
  return useBarChartConfig(data, {
    title: { text: 'Dimension Reduction Across Steps' },
  });
}

export function useCompressionRatioConfig(
  stepNames: string[] | null,
  ratios: number[] | null
): UseChartConfigReturn {
  const strategy = useMemo(() => createCompressionRatioStrategy(), []);
  
  const option = useMemo<EChartsOption>(() => {
    if (!stepNames || !ratios) return {};
    const data: BarChartData = { categories: stepNames, values: ratios };
    return strategy.createOption(data, { type: 'bar', theme: 'light', title: { text: 'Compression Ratio by Step' } });
  }, [stepNames, ratios, strategy]);
  
  return {
    option,
    dimensions: { height: chartHeights.medium },
    isValid: !!(stepNames && ratios),
    error: (!stepNames || !ratios) ? 'Missing data' : null,
  };
}

export function useDimensionEvolutionConfig(
  iterations: number[] | null,
  dimensions: number[] | null
): UseChartConfigReturn {
  const strategy = useMemo(() => createDimensionEvolutionStrategy(), []);
  
  const option = useMemo<EChartsOption>(() => {
    if (!iterations || !dimensions) return {};
    const data: LineChartData = { xData: iterations, yData: dimensions, smooth: true, areaStyle: true };
    return strategy.createOption(data, { type: 'line', theme: 'light' });
  }, [iterations, dimensions, strategy]);
  
  return {
    option,
    dimensions: { height: chartHeights.medium },
    isValid: !!(iterations && dimensions),
    error: (!iterations || !dimensions) ? 'Missing data' : null,
  };
}

export function useMultiTaskHeatmapConfig(
  paramNames: string[] | null,
  taskNames: string[] | null,
  importances: number[][] | null
): UseChartConfigReturn {
  const strategy = useMemo(() => createMultiTaskHeatmapStrategy(), []);
  
  const option = useMemo<EChartsOption>(() => {
    if (!paramNames || !taskNames || !importances) return {};
    const data: HeatmapChartData = { xCategories: paramNames, yCategories: taskNames, values: importances };
    return strategy.createOption(data, { type: 'heatmap', theme: 'light' });
  }, [paramNames, taskNames, importances, strategy]);
  
  const height = useMemo(() => chartHeights.auto(taskNames?.length ?? 0, 40, 300), [taskNames?.length]);
  
  return {
    option,
    dimensions: { height },
    isValid: !!(paramNames && taskNames && importances),
    error: (!paramNames || !taskNames || !importances) ? 'Missing data' : null,
  };
}
