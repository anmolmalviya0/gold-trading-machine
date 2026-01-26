'use client';

import React, { useEffect, useRef } from 'react';

interface TradingViewChartProps {
  symbol: string;
  interval?: string;
  theme?: 'light' | 'dark';
}

declare global {
  interface Window {
    TradingView: any;
  }
}

export default function TradingViewChart({ symbol, interval = '15m', theme = 'dark' }: TradingViewChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Map UI timeframes to TradingView interval codes
    const intervalMap: Record<string, string> = {
        '5m': '5',
        '15m': '15',
        '30m': '30',
        '1h': '60',
        '1d': 'D'
    };
    
    const tvInterval = intervalMap[interval] || '15';

    const script = document.createElement('script');
    script.src = 'https://s3.tradingview.com/tv.js';
    script.async = true;
    script.onload = () => {
      if (containerRef.current && window.TradingView) {
        // Clear container before re-rendering
        containerRef.current.innerHTML = '';
        
        new window.TradingView.widget({
          autosize: true,
          symbol: `BINANCE:${symbol}USDT`,
          interval: tvInterval,
          timezone: 'Etc/UTC',
          theme: theme,
          // ... rest of config
          style: '1',
          locale: 'en',
          toolbar_bg: '#f1f3f6',
          enable_publishing: false,
          hide_side_toolbar: false, // SHOW drawing tools
          allow_symbol_change: true,
          container_id: containerRef.current.id,
          studies: [
            'RSI@tv-basicstudies',
            'MASimple@tv-basicstudies',
            'ATR@tv-basicstudies'
          ],
          disabled_features: ['header_screenshot', 'header_compare'],
          enabled_features: ['study_templates'],
          overrides: {
            "mainSeriesProperties.candleStyle.upColor": "#10b981",
            "mainSeriesProperties.candleStyle.downColor": "#ef4444",
            "mainSeriesProperties.candleStyle.borderUpColor": "#10b981",
            "mainSeriesProperties.candleStyle.borderDownColor": "#ef4444",
            "mainSeriesProperties.candleStyle.wickUpColor": "#10b981",
            "mainSeriesProperties.candleStyle.wickDownColor": "#ef4444",
            "paneProperties.background": "#050505",
            "paneProperties.vertGridProperties.color": "rgba(16, 185, 129, 0.05)",
            "paneProperties.horzGridProperties.color": "rgba(16, 185, 129, 0.05)",
          }
        });
      }
    };
    document.head.appendChild(script);

    return () => {
      // Clean up script if needed
      if (script.parentNode) {
        script.parentNode.removeChild(script);
      }
    };
  }, [symbol, interval, theme]);

  return (
    <div 
      id={`tv_chart_${symbol}`} 
      ref={containerRef} 
      className="w-full h-full border border-emerald-900/20 rounded-lg overflow-hidden" 
    />
  );
}
