import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import type { UseChunkedDataOptions, UseChunkedDataReturn } from '../types';

// requestIdleCallback polyfill
const requestIdleCallbackPolyfill = 
  typeof window !== 'undefined' && 'requestIdleCallback' in window
    ? window.requestIdleCallback
    : (callback: IdleRequestCallback, options?: IdleRequestOptions): number => {
        const start = Date.now();
        return window.setTimeout(() => {
          callback({
            didTimeout: false,
            timeRemaining: () => Math.max(0, 50 - (Date.now() - start)),
          });
        }, options?.timeout ?? 1) as unknown as number;
      };

const cancelIdleCallbackPolyfill = 
  typeof window !== 'undefined' && 'cancelIdleCallback' in window
    ? window.cancelIdleCallback
    : (id: number) => window.clearTimeout(id);

export function useChunkedData<T>(
  data: T[],
  options: UseChunkedDataOptions = {}
): UseChunkedDataReturn<T> {
  const {
    chunkSize = 500,
    enabled = true,
    threshold = 1000,
    onComplete,
    idleTimeout = 50,
  } = options;
  const shouldChunk = useMemo(() => 
    enabled && data.length > threshold, 
    [enabled, data.length, threshold]
  );

  const [displayData, setDisplayData] = useState<T[]>(() => 
    shouldChunk ? [] : data
  );
  const [progress, setProgress] = useState(() => shouldChunk ? 0 : 100);
  const [isLoading, setIsLoading] = useState(shouldChunk);
  
  const idleCallbackRef = useRef<number | null>(null);
  const currentIndexRef = useRef(0);
  const onCompleteRef = useRef(onComplete);
  useEffect(() => {
    onCompleteRef.current = onComplete;
  }, [onComplete]);
  const reset = useCallback(() => {
    if (idleCallbackRef.current !== null) {
      cancelIdleCallbackPolyfill(idleCallbackRef.current);
      idleCallbackRef.current = null;
    }
    currentIndexRef.current = 0;
    
    if (shouldChunk) {
      setDisplayData([]);
      setProgress(0);
      setIsLoading(true);
    } else {
      setDisplayData(data);
      setProgress(100);
      setIsLoading(false);
    }
  }, [shouldChunk, data]);
  useEffect(() => {
    if (!data.length) {
      setDisplayData([]);
      setProgress(100);
      setIsLoading(false);
      return;
    }
    if (!shouldChunk) {
      setDisplayData(data);
      setProgress(100);
      setIsLoading(false);
      return;
    }
    if (idleCallbackRef.current !== null) {
      cancelIdleCallbackPolyfill(idleCallbackRef.current);
    }
    currentIndexRef.current = 0;
    setDisplayData([]);
    setProgress(0);
    setIsLoading(true);
    const renderChunk = () => {
      const startIndex = currentIndexRef.current;
      const endIndex = Math.min(startIndex + chunkSize, data.length);
      const chunk = data.slice(startIndex, endIndex);
      
      setDisplayData(prev => [...prev, ...chunk]);
      currentIndexRef.current = endIndex;
      
      const newProgress = Math.min(100, (endIndex / data.length) * 100);
      setProgress(newProgress);

      if (endIndex < data.length) {
        idleCallbackRef.current = requestIdleCallbackPolyfill(renderChunk, { 
          timeout: idleTimeout 
        });
      } else {
        setIsLoading(false);
        idleCallbackRef.current = null;
        onCompleteRef.current?.();
      }
    };
    idleCallbackRef.current = requestIdleCallbackPolyfill(renderChunk, { 
      timeout: idleTimeout 
    });
    return () => {
      if (idleCallbackRef.current !== null) {
        cancelIdleCallbackPolyfill(idleCallbackRef.current);
        idleCallbackRef.current = null;
      }
    };
  }, [data, chunkSize, shouldChunk, idleTimeout]);

  const isComplete = progress >= 100;

  return { 
    displayData, 
    progress, 
    isComplete, 
    isLoading,
    reset,
  };
}

export default useChunkedData;
