import React from 'react';
import type { ChartPlaceholderProps } from '../../types';

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'linear-gradient(135deg, #f5f7fa 0%, #e4e8ed 100%)',
    borderRadius: '8px',
    position: 'relative',
    overflow: 'hidden',
  },
  content: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '16px',
    zIndex: 1,
  },
  icon: {
    width: '64px',
    height: '64px',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    borderRadius: '12px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: 'white',
    fontSize: '28px',
    boxShadow: '0 4px 12px rgba(102, 126, 234, 0.3)',
  },
  title: {
    fontSize: '14px',
    color: '#666',
    fontWeight: 500,
  },
  subtitle: {
    fontSize: '12px',
    color: '#999',
    marginTop: '-8px',
  },
  shimmer: {
    position: 'absolute',
    top: 0,
    left: '-100%',
    width: '100%',
    height: '100%',
    background: 'linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.4) 50%, transparent 100%)',
    animation: 'shimmer 2s infinite',
  },
};
const shimmerKeyframes = `
  @keyframes shimmer {
    0% { left: -100%; }
    100% { left: 100%; }
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }
`;

export const ChartPlaceholder: React.FC<ChartPlaceholderProps> = ({
  height = 400,
  width = '100%',
  title = 'Waiting to load...',
  animate = true,
  className = '',
  style = {},
}) => {
  return (
    <>
<style>{shimmerKeyframes}</style>
      
      <div 
        className={`chart-placeholder ${className}`}
        style={{
          ...styles.container,
          height: typeof height === 'number' ? `${height}px` : height,
          width: typeof width === 'number' ? `${width}px` : width,
          ...style,
        }}
      >
{animate && <div style={styles.shimmer} />}
        
        <div style={{
          ...styles.content,
          animation: animate ? 'pulse 2s ease-in-out infinite' : 'none',
        }}>
<div style={styles.icon}>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
              <line x1="3" y1="9" x2="21" y2="9"/>
              <line x1="9" y1="21" x2="9" y2="9"/>
              <path d="M13 13h4v4h-4z"/>
            </svg>
          </div>
<span style={styles.title}>{title}</span>
<span style={styles.subtitle}>Scroll down to load chart</span>
        </div>
      </div>
    </>
  );
};

export default ChartPlaceholder;
