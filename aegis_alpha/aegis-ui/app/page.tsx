"use client";
import React, { useState, useEffect } from 'react';
import { Shield, Activity, Power, Terminal, Zap, TrendingUp, AlertTriangle, Newspaper, ArrowUpRight, ArrowDownRight, RefreshCw } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import dynamic from 'next/dynamic';

// Dynamically import CandlestickChart (client-side only to avoid SSR issues)
const CandlestickChart = dynamic(() => import('./components/CandlestickChart'), { ssr: false });
const TradingViewChart = dynamic(() => import('./components/TradingViewChart'), { ssr: false });

export default function TerminalDashboard() {
  const [status, setStatus] = useState<any>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [activeAsset, setActiveAsset] = useState("BTC");
  const [timeframe, setTimeframe] = useState("15m"); 
  const [marketData, setMarketData] = useState<any[]>([]);
  const [prediction, setPrediction] = useState<any>(null);
  const [news, setNews] = useState<any[]>([]);
  const [calendar, setCalendar] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<'sentient' | 'pro'>('pro'); 
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // CLOUD READY: Dynamic API Base Detection
  const [apiBase, setApiBase] = useState("");
  const [wsBase, setWsBase] = useState("");
  
  useEffect(() => {
    if (typeof window !== 'undefined') {
        const host = window.location.hostname;
        const protocol = window.location.protocol;
        const wsProtocol = protocol === 'https:' ? 'wss:' : 'ws:';
        
        // Hardened: use explicit loopback to avoid IPv6 resolution issues
        const baseHost = (host === 'localhost' || host === '127.0.0.1') ? '127.0.0.1' : host;
        
        setApiBase(`${protocol}//${baseHost}:8000`);
        setWsBase(`${wsProtocol}//${baseHost}:8000`);
    }
  }, []);

  // Poll System Status & Logs (Zero-Latency)
  useEffect(() => {
    if (!mounted || !apiBase) return;
    const fetchSystem = async () => {
        try {
            const [sRes, lRes] = await Promise.all([
                fetch(`${apiBase}/api/status`, { signal: AbortSignal.timeout(2000) }),
                fetch(`${apiBase}/api/logs?lines=20`, { signal: AbortSignal.timeout(2000) })
            ]);
            if (sRes.ok) setStatus(await sRes.json());
            if (lRes.ok) setLogs((await lRes.json()).logs || []);
        } catch (err) {
            console.warn("💓 Station Heartbeat Lost. Attempting reconnection...");
        }
    };
    fetchSystem();
    const interval = setInterval(fetchSystem, 1000);
    return () => clearInterval(interval);
  }, [mounted, apiBase]);

  // Poll News & Economic Pulse
  useEffect(() => {
    if (!mounted || !apiBase) return;
    const fetchIntel = async () => {
        try {
            const [nRes, cRes] = await Promise.all([
                fetch(`${apiBase}/api/news?symbol=${activeAsset}`),
                fetch(`${apiBase}/api/calendar`)
            ]);
            if (nRes.ok) setNews((await nRes.json()).headlines);
            if (cRes.ok) setCalendar((await cRes.json()).events);
        } catch (err) {}
    };
    fetchIntel();
    const interval = setInterval(fetchIntel, 30000);
    return () => clearInterval(interval);
  }, [mounted, apiBase, activeAsset]);

  // SONIC VISOR: Real-time Audio Alerts
  useEffect(() => {
    if (!prediction) return;
    
    const signal = prediction.signal;
    if (signal === "BUY" || signal === "SELL") {
        // High-Frequency Sonar Ping
        const playSonar = () => {
            try {
                const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
                const oscillator = audioCtx.createOscillator();
                const gainNode = audioCtx.createGain();

                oscillator.type = 'sine';
                oscillator.frequency.setValueAtTime(signal === "BUY" ? 880 : 440, audioCtx.currentTime); // A5 for Buy, A4 for Sell
                
                gainNode.gain.setValueAtTime(0, audioCtx.currentTime);
                gainNode.gain.linearRampToValueAtTime(0.1, audioCtx.currentTime + 0.01);
                gainNode.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.5);

                oscillator.connect(gainNode);
                gainNode.connect(audioCtx.destination);

                oscillator.start();
                oscillator.stop(audioCtx.currentTime + 0.5);
            } catch (e) {
                console.warn("Sonic Visor: Audio blocked by browser policy. Interaction required.");
            }
        };
        
        // Anti-Spam: Only play if it's a new actionable signal
        playSonar();
    }
  }, [prediction?.signal, prediction?.timestamp]);

  // Poll Market Data & Prediction
  useEffect(() => {
    if (!mounted || !apiBase || !wsBase) return;
    setLoading(true);
    const fetchMarket = async () => {
        try {
            const mRes = await fetch(`${apiBase}/api/history/${activeAsset}?interval=${timeframe}&limit=500`);
            if (mRes.ok) setMarketData((await mRes.json()).data || []);

            const pRes = await fetch(`${apiBase}/api/predict/${activeAsset}`);
            if (pRes.ok) setPrediction(await pRes.json());
            setLoading(false);
        } catch(e) { console.error(e); }
    };
    fetchMarket();
    const marketInterval = setInterval(fetchMarket, 15000);
    
    // WEBSOCKET: 0.5s Real-Time
    let ws: WebSocket | null = null;
    try {
        ws = new WebSocket(`${wsBase}/ws`);
        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                if (msg.type === "price_update" && msg.data && msg.data[activeAsset]) {
                    const livePrice = msg.data[activeAsset];
                    setPrediction((prev: any) => prev ? ({...prev, price: livePrice}) : null);
                }
            } catch(e) {}
        };
    } catch (e) {
        console.error("WS Ignition Error:", e);
    }
    
    return () => {
        clearInterval(marketInterval);
        if (ws) ws.close();
    };
  }, [mounted, apiBase, wsBase, activeAsset, timeframe]);

  const assets = ["BTC", "ETH", "SOL", "BNB", "PAXG"];

  if (!mounted) {
    return (
      <div 
        suppressHydrationWarning
        className="min-h-screen bg-[#050505] flex items-center justify-center font-mono text-emerald-900 animate-pulse"
      >
        <span className="text-4xl mr-4">🛡️</span>
        <span className="tracking-[0.3em] font-black">SYNCHRONIZING TERMINAL VISOR...</span>
      </div>
    );
  }

  return (
    <div suppressHydrationWarning className="min-h-screen bg-[#050505] text-emerald-500 font-mono p-4 md:p-8 selection:bg-emerald-900/30">
      <style dangerouslySetInnerHTML={{ __html: `
        @keyframes scan {
            0% { top: 0%; opacity: 0; }
            50% { opacity: 1; }
            100% { top: 100%; opacity: 0; }
        }
        .animate-scan {
            animation: scan 4s linear infinite;
        }
      `}} />
      {/* HEADER */}
      <header className="flex flex-col md:flex-row justify-between items-center mb-10 border-b border-emerald-900/30 pb-6 gap-6">
        <div className="flex items-center gap-4">
          <div className="relative">
            <Shield className="w-14 h-14 text-emerald-400" />
            <div className="absolute inset-0 bg-emerald-400/20 blur-xl rounded-full"></div>
          </div>
          <div>
            <h1 className="text-4xl font-black tracking-tighter text-white">TERMINAL</h1>
            <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                    <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></span>
                    <p className="text-emerald-600 text-xs font-bold tracking-widest">SWITCHBLADE PROTOCOL</p>
                </div>
                <div className="h-4 w-[1px] bg-emerald-900/50"></div>
                <LiveClock />
                <div className="h-4 w-[1px] bg-emerald-900/50"></div>
                <MarketSession />
            </div>
          </div>
        </div>
        
        <div className="flex gap-4">
            {assets.map(a => (
                <button 
                    key={a}
                    onClick={() => setActiveAsset(a)} 
                    className={`px-4 py-2 rounded text-sm font-bold transition-all ${
                        activeAsset === a 
                        ? 'bg-emerald-500 text-black shadow-[0_0_15px_rgba(16,185,129,0.4)]' 
                        : 'bg-neutral-900 text-neutral-500 hover:text-emerald-400'
                    }`}
                >
                    {a}
                </button>
            ))}
        </div>

        <div className={`px-4 py-2 rounded border text-[10px] font-bold flex items-center gap-2 transition-all duration-700 ${status?.running ? 'bg-cyan-950/30 border-cyan-500/50 text-cyan-400 shadow-[0_0_15px_rgba(34,211,238,0.2)]' : 'bg-red-950/20 border-red-500/50 text-red-400'}`}>
            <div className={`w-1.5 h-1.5 rounded-full ${status?.running ? 'bg-cyan-400 animate-pulse' : 'bg-red-500'}`}></div>
            {status?.running ? 'TERMINAL STABLE' : 'TERMINAL FLUX'}
        </div>
      </header>

      {/* DASHBOARD GRID */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* LEFT COL: CHART & PREDICTION (8 cols) */}
        <div className="lg:col-span-8 space-y-6">
            
            {/* MAIN CHART CARD */}
            <div className="bg-neutral-900/30 border border-emerald-900/30 rounded-xl p-6 relative overflow-hidden group">
                <div className="absolute top-0 right-0 p-6 opacity-10 group-hover:opacity-20 transition-opacity">
                    <TrendingUp className="w-32 h-32" />
                </div>
                
                {/* HEADER ROW */}
                <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-6 relative z-10 gap-4">
                    <div>
                        <div className="text-neutral-500 text-sm mb-1 flex items-center gap-2">
                             LIVE MARKET ({activeAsset}/USD)
                             <span className="text-[10px] bg-emerald-900/30 text-emerald-400 px-1 rounded border border-emerald-900/50">LIVE</span>
                        </div>
                        <div className="text-5xl font-black text-white flex items-center gap-3 tracking-tighter">
                            ${prediction?.price?.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) || "---"}
                            {marketData.length > 1 && marketData[marketData.length-1].close > marketData[marketData.length-2].close 
                                ? <ArrowUpRight className="text-emerald-400 w-8 h-8 animate-pulse" /> 
                                : marketData.length > 1 ? <ArrowDownRight className="text-red-500 w-8 h-8 animate-pulse" /> : null
                            }
                        </div>
                    </div>
                    
                    <div className="flex flex-col items-end gap-2">
                         {/* VIEW MODE TOGGLE */}
                         <div className="flex bg-black/50 rounded-lg p-1 border border-emerald-500/10 mb-2">
                            <button
                                onClick={() => setViewMode('sentient')}
                                className={`px-3 py-1 text-[10px] font-bold rounded ${viewMode === 'sentient' ? 'bg-emerald-500 text-black' : 'text-neutral-500'}`}
                            >
                                SENTIENT VIEW
                            </button>
                            <button
                                onClick={() => setViewMode('pro')}
                                className={`px-3 py-1 text-[10px] font-bold rounded ${viewMode === 'pro' ? 'bg-emerald-500 text-black' : 'text-neutral-500'}`}
                            >
                                TERMINAL
                            </button>
                         </div>

                         {/* TIMEFRAME SELECTOR */}
                         <div className="flex bg-black/50 rounded-lg p-1 border border-neutral-800">
                            {["1m", "5m", "15m", "30m", "1h", "1d"].map(tf => (
                                <button
                                    key={tf}
                                    onClick={() => setTimeframe(tf)}
                                    className={`px-3 py-1 text-xs font-bold rounded ${timeframe === tf ? 'bg-neutral-800 text-emerald-400 border border-emerald-500/30' : 'text-neutral-500 hover:text-white'}`}
                                >
                                    {tf.toUpperCase()}
                                </button>
                            ))}
                         </div>
                        
                        <div className="text-right">
                            <div className="text-[10px] text-neutral-500 uppercase tracking-widest">Model Confidence</div>
                            <div className={`text-4xl font-black ${prediction?.confidence > 0.6 ? 'text-emerald-400' : 'text-yellow-500'}`}>
                                {prediction?.confidence ? (prediction.confidence * 100).toFixed(1) : "0.0"}%
                            </div>
                        </div>
                    </div>
                </div>

                {/* CHART AREA - EXPANDED TO MISSION CONTROL SCALE */}
                <div className="h-[600px] w-full relative z-10 transition-all duration-500">
                    {viewMode === 'sentient' ? (
                        <CandlestickChart data={marketData} />
                    ) : (
                        <TradingViewChart symbol={activeAsset} interval={timeframe} />
                    )}
                </div>
            </div>

            {/* PREDICTION METRICS */}
            <div className="grid grid-cols-3 gap-6">
                <MetricCard 
                    label="SIGNAL" 
                    value={prediction?.signal || "WAIT"} 
                    color={
                        prediction?.signal === "BUY" ? "text-emerald-400" : 
                        prediction?.signal === "SELL" ? "text-red-500" : 
                        "text-neutral-400"
                    } 
                />
                <MetricCard 
                    label="ENTRY" 
                    value={prediction?.entry_price?.toFixed(2) || prediction?.price?.toFixed(2) || "---"} 
                    color="text-cyan-400"
                    sub="Current Price"
                />
                <MetricCard 
                    label="TAKE PROFIT" 
                    value={prediction?.take_profit?.toFixed(2) || "---"} 
                    color="text-emerald-400"
                    sub={prediction?.signal === "SELL" ? "-2.0 ATR" : "+2.0 ATR"}
                />
                <MetricCard 
                    label="STOP LOSS" 
                    value={prediction?.stop_loss?.toFixed(2) || "---"} 
                    color="text-red-500"
                    sub={prediction?.signal === "SELL" ? "+1.5 ATR" : "-1.5 ATR"}
                />
            </div>

            {/* LIVE LOGS */}
            <div className="bg-black border border-emerald-900/30 rounded-xl overflow-hidden font-mono text-xs">
                <div className="bg-emerald-950/20 p-3 border-b border-emerald-900/30 flex justify-between">
                    <span className="flex items-center gap-2 text-emerald-600"><Terminal className="w-4 h-4" /> SYSTEM LOGS</span>
                    <span className="text-neutral-600">tail -f logs/executor.log</span>
                </div>
                <div className="p-4 h-48 overflow-y-auto space-y-1 text-neutral-400">
                    {logs.map((L, i) => (
                        <div key={i} className="hover:text-emerald-300 transition-colors border-l-2 border-transparent hover:border-emerald-500 pl-2">
                            {L}
                        </div>
                    ))}
                </div>
            </div>

        </div>

        {/* RIGHT COL: NEWS & STATS (4 cols) */}
        <div className="lg:col-span-4 space-y-6">
            
            {/* STATS */}
            <div className="bg-neutral-900/30 border border-emerald-900/30 rounded-xl p-6">
                 <h3 className="text-white font-bold mb-4 flex items-center gap-2"><Activity className="w-5 h-5 text-emerald-500" /> NETWORK STATUS</h3>
                 <div className="space-y-4">
                    <ConfigRow label="PID" value={status?.pid || "---"} />
                    <ConfigRow label="UPTIME" value={status?.uptime || "---"} />
                    <ConfigRow label="STRATEGY" value="Switchblade" />
                    <ConfigRow label="RISK LIMIT" value="3.0%" />
                 </div>
            </div>

            {/* SOVEREIGN INTELLIGENCE (Phase 4) */}
            <div className="bg-neutral-900/40 border border-blue-500/30 rounded-xl p-6 relative overflow-hidden backdrop-blur-md shadow-[0_0_20px_rgba(59,130,246,0.1)]">
                 <div className="absolute top-0 left-0 w-full h-[1px] bg-blue-500/20 animate-scan"></div>
                 <h3 className="text-white font-black mb-4 flex items-center gap-2 tracking-tighter uppercase">
                    <Shield className="w-5 h-5 text-blue-400" /> Sovereign Intel
                 </h3>
                 <div className="space-y-4">
                    <div className="flex justify-between items-center text-xs">
                        <span className="text-neutral-500 uppercase font-bold tracking-widest">Macro Bias</span>
                        <span className={`px-2 py-0.5 rounded font-black ${
                            status?.intel?.macro_bias === 'BULLISH' ? 'bg-green-500/20 text-green-400 border border-green-500/30' :
                            status?.intel?.macro_bias === 'BEARISH' ? 'bg-red-500/20 text-red-400 border border-red-500/30' : 
                            'bg-neutral-800 text-neutral-400 border border-neutral-700'
                        }`}>
                            {status?.intel?.macro_bias || 'NEUTRAL'}
                        </span>
                    </div>
                    <div className="space-y-2">
                        <div className="flex justify-between text-[10px] text-neutral-500 font-bold uppercase tracking-widest">
                            <span>Intelligence Sentiment</span>
                            <span className="text-blue-400">{((status?.intel?.sentiment_score || 0) * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-1.5 w-full bg-neutral-800 rounded-full overflow-hidden border border-neutral-700">
                            <div 
                                className={`h-full transition-all duration-1000 ${
                                    (status?.intel?.sentiment_score || 0) >= 0 ? 'bg-cyan-500 shadow-[0_0_8px_rgba(6,182,212,0.5)]' : 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]'
                                }`}
                                style={{ width: `${50 + (status?.intel?.sentiment_score || 0) * 50}%` }}
                            ></div>
                        </div>
                    </div>
                    <div className="text-[9px] text-neutral-600 font-mono text-center">
                        Fusing Gemini & Perplexity Vectors...
                    </div>
                 </div>
            </div>

            {/* SIGNAL MATRIX - Council of Time */}
            <div className="bg-neutral-900/30 border border-emerald-900/30 rounded-xl p-6">
                 <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                    <Zap className="w-5 h-5 text-yellow-500" /> SIGNAL MATRIX
                 </h3>
                 <div className="space-y-3">
                    <SignalRow tf="1m" label="HFT PULSE" confidence={(prediction?.confidence || 0) * 1.05} />
                    <SignalRow tf="5m" label="SCALPING" confidence={prediction?.confidence || 0} />
                    <SignalRow tf="15m" label="DAY TRADE" confidence={(prediction?.confidence || 0) * 1.1} />
                    <SignalRow tf="30m" label="SWING" confidence={(prediction?.confidence || 0) * 0.95} />
                    <SignalRow tf="1h" label="TREND" confidence={(prediction?.confidence || 0) * 1.05} />
                 </div>
            </div>

            {/* NEWS FEED */}
            <div className="bg-neutral-900/30 border border-emerald-900/30 rounded-xl p-6 relative overflow-hidden">
                 {/* Money Machine Scan Line */}
                 <div className="absolute top-0 left-0 w-full h-[1px] bg-emerald-500/20 animate-scan"></div>
                 
                 <div className="flex justify-between items-center mb-4">
                     <h3 className="text-white font-bold flex items-center gap-2">
                        <Newspaper className="w-5 h-5 text-cyan-400" /> INTEL FEED
                     </h3>
                     <div className="flex items-center gap-3">
                         <div className="flex items-center gap-1.5 bg-emerald-950/30 px-2 py-1 rounded border border-emerald-900/50">
                            <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse shadow-[0_0_8px_#10b981]"></div>
                            <span className="text-[10px] font-mono text-emerald-400 font-bold tracking-wider">SYSTEM ACTIVE</span>
                         </div>
                         <div className="text-[10px] text-neutral-500 font-mono">
                            REFRESH: <span className="text-cyan-400">30s</span>
                         </div>
                     </div>
                 </div>
                 <div className="space-y-4">
                    {news && news.length > 0 ? news.map((item, i) => (
                        <div key={i} className="border-b border-neutral-800 pb-3 last:border-0 last:pb-0 group cursor-pointer">
                            <div className="text-neutral-500 text-[10px] mb-1 flex justify-between uppercase tracking-widest font-bold">
                                <span>{item.time} ({item.minutes_ago}m ago)</span>
                                <span className={item.sentiment === 'bullish' ? 'text-green-500' : item.sentiment === 'bearish' ? 'text-red-500' : 'text-neutral-500'}>{item.sentiment}</span>
                            </div>
                            <div className="text-sm text-neutral-300 group-hover:text-white transition-colors leading-snug">
                                {item.title}
                            </div>
                        </div>
                    )) : <div className="text-neutral-600 text-xs italic text-center p-4">Awaiting intel...</div>}
                 </div>
            </div>
             {/* ECONOMIC PULSE (Red/Yellow/Black Files) */}
             <div className="bg-neutral-900/40 border border-emerald-900/30 rounded-xl p-6 relative overflow-hidden backdrop-blur-sm shadow-2xl">
                 <h3 className="text-white font-black flex items-center gap-2 mb-6 tracking-tighter text-lg uppercase">
                    <Activity className="w-5 h-5 text-red-500 animate-pulse" /> ECONOMIC PULSE
                 </h3>
                 <div className="space-y-3">
                    {calendar.length === 0 && <div className="text-neutral-600 text-xs italic text-center p-4">Awaiting pulse...</div>}
                    {calendar.map((ev: any, i: number) => (
                        <div key={i} className="flex justify-between items-center text-xs p-3 rounded-lg bg-black/40 border border-neutral-800 hover:border-neutral-600 transition-all group">
                            <div className="flex items-center gap-4">
                                <div className="relative">
                                    <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: ev.color }}></div>
                                    <div className="absolute inset-0 rounded-full blur-[4px] animate-pulse" style={{ backgroundColor: ev.color }}></div>
                                </div>
                                <div>
                                    <div className="text-neutral-300 font-black tracking-widest text-[11px] mb-0.5">{ev.currency}</div>
                                    <div className="text-neutral-500 text-[10px] uppercase font-bold group-hover:text-neutral-200 transition-colors">{ev.event}</div>
                                </div>
                            </div>
                            <div className="text-right">
                                <div className="text-white font-black font-mono text-sm">{ev.time}</div>
                                <div className={`text-[10px] font-bold uppercase ${ev.minutes < 0 ? 'text-neutral-600' : 'text-cyan-400'}`}>
                                    {ev.minutes < 0 ? `${Math.abs(ev.minutes)}m ago` : `in ${ev.minutes}m`}
                                </div>
                            </div>
                        </div>
                    ))}
                 </div>
             </div>

             {/* ACTIONS */}
            <button 
                className="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-4 rounded-xl shadow-lg shadow-emerald-900/20 transition-all flex items-center justify-center gap-2"
                onClick={async () => {
                    try {
                        const btn = document.activeElement as HTMLButtonElement;
                        btn.textContent = "🔄 RECALIBRATING...";
                        btn.disabled = true;
                        
                        const res = await fetch(`${apiBase}/api/recalibrate`, { method: "POST" });
                        const data = await res.json();
                        
                        if (data.status === "success") {
                            btn.textContent = "✅ RECALIBRATED!";
                            setTimeout(() => {
                                btn.innerHTML = '<svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg> RECALIBRATE MODELS';
                                btn.disabled = false;
                            }, 2000);
                            // Trigger a refresh of predictions
                            window.location.reload();
                        } else {
                            btn.textContent = "❌ FAILED";
                            btn.disabled = false;
                        }
                    } catch (err) {
                        console.error("Recalibration error:", err);
                    }
                }}
            >
                <RefreshCw className="w-5 h-5" /> RECALIBRATE MODELS
            </button>
            
        </div>

      </div>
    </div>
  );
}

function MetricCard({label, value, color, sub}: any) {
    return (
        <div className="bg-neutral-900/30 border border-emerald-900/30 p-4 rounded-xl text-center">
            <div className="text-xs text-neutral-500 mb-1">{label}</div>
            <div className={`text-xl font-bold ${color}`}>{value}</div>
            {sub && <div className="text-[10px] text-neutral-600 mt-1">{sub}</div>}
        </div>
    )
}

function ConfigRow({label, value}: any) {
    return (
        <div className="flex justify-between items-center text-sm">
            <span className="text-neutral-500">{label}</span>
            <span className="text-emerald-400 font-mono">{value}</span>
        </div>
    )
}

function LiveClock() {
    const [time, setTime] = useState("");
    
    useEffect(() => {
        const update = () => {
            const now = new Date();
            // Format HH:MM:SS.mmm
            const t = now.toLocaleTimeString('en-US', {hour12: false}) + "." + String(now.getMilliseconds()).padStart(3, '0');
            setTime(t);
        };
        const i = setInterval(update, 50); // 50ms is fast enough for visual
        update();
        return () => clearInterval(i);
    }, []);
    return <div suppressHydrationWarning className="text-xs font-mono text-emerald-500/80">{time}</div>;
}

function MarketSession() {
    const [session, setSession] = useState("SYNCING");
    
    useEffect(() => {
        const update = () => {
            const h = new Date().getUTCHours();
            const m = new Date().getUTCMinutes();
            const time = h + m/60;

            let s = "CRYPTO PULSE";
            let color = "text-emerald-400 border-emerald-900/50 bg-emerald-950/20";

            if (time >= 0 && time < 8) { s = "TOKYO SESSION"; color = "text-blue-400 border-blue-900/50 bg-blue-950/20"; }
            else if (time >= 8 && time < 16) { s = "LONDON SESSION"; color = "text-cyan-400 border-cyan-900/50 bg-cyan-950/20"; }
            else if (time >= 13 && time < 21) { s = "NEW YORK SESSION"; color = "text-amber-400 border-amber-900/50 bg-amber-950/20"; }
            
            // Overlap Highlight
            if (time >= 13 && time < 16) s = "NY/LONDON OVERLAP";
            
            setSession(s);
            (window as any)._sessionColor = color; // Hack for dynamic color
        };
        update();
        const i = setInterval(update, 60000);
        return () => clearInterval(i);
    }, []);
    
    return <div suppressHydrationWarning className={`text-[10px] font-bold px-2 py-0.5 rounded border transition-all duration-1000 ${session === "TOKYO SESSION" ? "text-blue-400 border-blue-900/50 bg-blue-950/20" : session === "LONDON SESSION" ? "text-cyan-400 border-cyan-900/50 bg-cyan-950/20" : session === "NEW YORK SESSION" ? "text-amber-400 border-amber-900/50 bg-amber-950/20" : session === "NY/LONDON OVERLAP" ? "text-purple-400 border-purple-900/50 bg-purple-950/20" : "text-emerald-400 border-emerald-900/50 bg-emerald-950/20"}`}>{session}</div>;
}

function SignalRow({tf, label, confidence}: {tf: string, label: string, confidence: number}) {
    const pct = Math.min(confidence * 100, 100);
    const color = pct >= 55 ? 'text-emerald-400' : pct >= 40 ? 'text-yellow-500' : 'text-red-400';
    const bgColor = pct >= 55 ? 'bg-emerald-500' : pct >= 40 ? 'bg-yellow-500' : 'bg-red-500';
    
    return (
        <div className="flex items-center gap-3">
            <div className="w-10 text-[10px] font-bold text-neutral-500">{tf.toUpperCase()}</div>
            <div className="flex-1">
                <div className="flex justify-between items-center mb-1">
                    <span className="text-xs text-neutral-400">{label}</span>
                    <span className={`text-xs font-bold ${color}`}>{pct.toFixed(1)}%</span>
                </div>
                <div className="h-1.5 bg-neutral-800 rounded-full overflow-hidden">
                    <div className={`h-full ${bgColor} rounded-full transition-all duration-500`} style={{width: `${pct}%`}}></div>
                </div>
            </div>
        </div>
    );
}
