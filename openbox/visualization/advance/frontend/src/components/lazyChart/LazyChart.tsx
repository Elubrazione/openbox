import React, { useMemo, useCallback } from 'react';
import ReactECharts from 'echarts-for-react';
import type { EChartsOption } from 'echarts';
import { useLazyChart } from '../../hooks/useLazyChart';
import { useChunkedData } from '../../hooks/useChunkedData';
import { ChartPlaceholder } from './ChartPlaceholder';
import { LoadingBar } from './LoadingBar';
import type { LazyChartProps } from '../../types';
const defaultGetOption = (data: any[], type: string): EChartsOption => {
  const baseOption: EChartsOption = {
    tooltip: {
      trigger: type === 'pie' ? 'item' : 'axis',
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true,
    },
  };
  switch (type) {
    case 'line':
      return {
        ...baseOption,
        xAxis: {
          type: 'category',
          data: data.map((_, i) => i),
          boundaryGap: false,
        },
        yAxis: { type: 'value' },
        series: [{
          type: 'line',
          data: data,
          smooth: true,
          sampling: 'lttb',
          areaStyle: {
            opacity: 0.3,
          },
        }],
      };

    case 'bar':
      return {
        ...baseOption,
        xAxis: {
          type: 'category',
          data: data.map((_, i) => i),
        },
        yAxis: { type: 'value' },
        series: [{
          type: 'bar',
          data: data,
          large: true,
          largeThreshold: 500,
        }],
      };

    case 'scatter':
      return {
        ...baseOption,
        xAxis: { type: 'value' },
        yAxis: { type: 'value' },
        series: [{
          type: 'scatter',
          data: data,
          large: true,
          largeThreshold: 2000,
          symbolSize: 4,
        }],
      };

    default:
      return {
        ...baseOption,
        xAxis: { type: 'category', data: data.map((_, i) => i) },
        yAxis: { type: 'value' },
        series: [{ type: type as 'line', data: data }],
      };
  }
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    width: '100%',
    background: '#fff',
    borderRadius: '8px',
    padding: '20px',
    marginBottom: '20px',
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.06)',
    position: 'relative',
  },
  title: {
    fontSize: '16px',
    fontWeight: 600,
    color: '#2c3e50',
    marginBottom: '16px',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  badge: {
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    fontSize: '10px',
    padding: '2px 8px',
    borderRadius: '10px',
    fontWeight: 500,
  },
  chartWrapper: {
    position: 'relative',
  },
  stats: {
    position: 'absolute',
    top: '8px',
    right: '8px',
    background: 'rgba(255, 255, 255, 0.9)',
    padding: '4px 10px',
    borderRadius: '4px',
    fontSize: '11px',
    color: '#666',
    boxShadow: '0 1px 4px rgba(0,0,0,0.1)',
  },
};

export const LazyChart: React.FC<LazyChartProps> = ({
  data,
  type,
  getOption,
  option: directOption,
  height = 400,
  width = '100%',
  title,
  chunkSize = 500,
  chunkThreshold = 1000,
  lazyThreshold = 0.1,
  lazyRootMargin = '100px',
  disableLazy = false,
  disableChunking = false,
  onLoadComplete,
  placeholder,
  className = '',
  style = {},
  showProgress = true,
  notMerge = true,
  lazyUpdate = true,
}) => {
  const { ref, isVisible } = useLazyChart({
    threshold: lazyThreshold,
    rootMargin: lazyRootMargin,
    disabled: disableLazy,
  });
  const { 
    displayData, 
    progress, 
    isComplete, 
    isLoading 
  } = useChunkedData(data, {
    chunkSize,
    threshold: chunkThreshold,
    enabled: !disableChunking && isVisible,
    onComplete: onLoadComplete,
  });
  const chartOption = useMemo<EChartsOption>(() => {
    if (directOption) {
      return directOption;
    }
    if (getOption) {
      return getOption(displayData);
    }
    return defaultGetOption(displayData, type);
  }, [directOption, getOption, displayData, type]);
  const chartHeight = typeof height === 'number' ? height : parseInt(height, 10) || 400;
  const shouldShowProgress = showProgress && isVisible && !isComplete && data.length > chunkThreshold;
  const isLargeDataset = data.length > chunkThreshold;

  return (
    <div 
      ref={ref}
      className={`lazy-chart ${className}`}
      style={{
        ...styles.container,
        ...style,
      }}
    >
{title && (
        <div style={styles.title}>
          {title}
          {isLargeDataset && (
            <span style={styles.badge}>
              {data.length.toLocaleString()} points
            </span>
          )}
        </div>
      )}
<div style={styles.chartWrapper}>
        {!isVisible ? (
          placeholder || (
            <ChartPlaceholder 
              height={chartHeight} 
              title={title || 'Chart'}
              animate={true}
            />
          )
        ) : (
          <>
{shouldShowProgress && (
              <LoadingBar 
                progress={progress}
                showText={true}
                dataInfo={{
                  loaded: displayData.length,
                  total: data.length,
                }}
              />
            )}
<ReactECharts
              option={chartOption}
              style={{ 
                height: chartHeight,
                width: typeof width === 'number' ? `${width}px` : width,
              }}
              notMerge={notMerge}
              lazyUpdate={lazyUpdate}
              opts={{
                renderer: data.length > 5000 ? 'canvas' : 'svg',
              }}
            />
{isLargeDataset && isComplete && (
              <div style={styles.stats}>
                ✓ {displayData.length.toLocaleString()} points rendered
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default LazyChart;
