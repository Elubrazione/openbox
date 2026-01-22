import React from 'react';
import type { LoadingBarProps } from '../../types';

const styles: Record<string, React.CSSProperties> = {
  container: {
    width: '100%',
    padding: '12px 16px',
    background: 'rgba(255, 255, 255, 0.95)',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
    marginBottom: '12px',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '8px',
  },
  label: {
    fontSize: '13px',
    fontWeight: 600,
    color: '#2c3e50',
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
  },
  percentage: {
    fontSize: '13px',
    fontWeight: 700,
    color: '#667eea',
  },
  barContainer: {
    height: '8px',
    background: '#e9ecef',
    borderRadius: '4px',
    overflow: 'hidden',
    position: 'relative',
  },
  bar: {
    height: '100%',
    borderRadius: '4px',
    transition: 'width 0.3s ease-out',
    position: 'relative',
    overflow: 'hidden',
  },
  shimmer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.3) 50%, transparent 100%)',
    animation: 'progress-shimmer 1.5s infinite',
  },
  dataInfo: {
    marginTop: '6px',
    fontSize: '11px',
    color: '#999',
    textAlign: 'right' as const,
  },
};
const keyframes = `
  @keyframes progress-shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
  }
`;

export const LoadingBar: React.FC<LoadingBarProps> = ({
  progress,
  height = 8,
  showText = true,
  color = 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
  backgroundColor = '#e9ecef',
  className = '',
  style = {},
  dataInfo,
}) => {
  const normalizedProgress = Math.min(100, Math.max(0, progress));
  const isComplete = normalizedProgress >= 100;

  return (
    <>
      <style>{keyframes}</style>
      
      <div 
        className={`loading-bar ${className}`}
        style={{
          ...styles.container,
          ...style,
        }}
      >
<div style={styles.header}>
          <span style={styles.label}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#667eea" strokeWidth="2.5">
              <polyline points="22,12 18,12 15,21 9,3 6,12 2,12" />
            </svg>
            {isComplete ? 'Data Loaded' : 'Loading Data...'}
          </span>
          
          {showText && (
            <span style={styles.percentage}>
              {normalizedProgress.toFixed(0)}%
            </span>
          )}
        </div>
<div 
          style={{
            ...styles.barContainer,
            height: `${height}px`,
            background: backgroundColor,
          }}
        >
          <div 
            style={{
              ...styles.bar,
              width: `${normalizedProgress}%`,
              background: color,
            }}
          >
            {!isComplete && <div style={styles.shimmer} />}
          </div>
        </div>
{dataInfo && (
          <div style={styles.dataInfo}>
            {dataInfo.loaded.toLocaleString()} / {dataInfo.total.toLocaleString()} data points
          </div>
        )}
      </div>
    </>
  );
};

export default LoadingBar;
