import { useRef, useState, useEffect } from 'react';
import type { UseLazyChartOptions, UseLazyChartReturn } from '../types';

export const useLazyChart = (options: UseLazyChartOptions = {}): UseLazyChartReturn => {
  const {
    threshold = 0.1,
    rootMargin = '100px',
    onVisible,
    disabled = false,
  } = options;

  const ref = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(disabled);
  const [hasLoaded, setHasLoaded] = useState(disabled);
  const onVisibleRef = useRef(onVisible);
  const observerRef = useRef<IntersectionObserver | null>(null);
  useEffect(() => {
    onVisibleRef.current = onVisible;
  }, [onVisible]);

  useEffect(() => {
    if (disabled) {
      setIsVisible(true);
      setHasLoaded(true);
      return;
    }
    if (hasLoaded) return;

    const element = ref.current;
    if (!element) return;
    if (observerRef.current) {
      observerRef.current.disconnect();
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const [entry] = entries;
        if (entry.isIntersecting && !hasLoaded) {
          setIsVisible(true);
          setHasLoaded(true);
          onVisibleRef.current?.();
          observer.disconnect();
          observerRef.current = null;
        }
      },
      {
        threshold,
        rootMargin,
      }
    );

    observerRef.current = observer;
    observer.observe(element);

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
        observerRef.current = null;
      }
    };
  }, [threshold, rootMargin, hasLoaded, disabled]);

  return { ref, isVisible, hasLoaded };
};

export default useLazyChart;
