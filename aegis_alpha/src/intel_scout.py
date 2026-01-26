import os
import json
import asyncio
import httpx
from datetime import datetime
from typing import Dict, List
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class IntelScout:
    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        self.api_base = "http://127.0.0.1:8000"
        
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        self.intel_buffer = {
            "sentiment_score": 0.0,
            "macro_bias": "NEUTRAL",
            "high_impact_events": [],
            "last_update": None
        }

    async def poll_perplexity(self, asset: str = "Bitcoin"):
        """Interrogate Perplexity Pro for live macro shifts and regulatory events"""
        if not self.perplexity_key:
            return
        
        print(f"📡 {asset}: Interrogating Perplexity Pro for Live Context...")
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.perplexity_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a quantitative crypto macro analyst. Analyze the last 1 hour of news for the specified asset. Output a JSON object with: 'bias' (BULLISH/BEARISH/NEUTRAL), 'confidence' (0-1), and 'reason' (short string)."
                },
                {
                    "role": "user",
                    "content": f"Analyze the current market sentiment and any major breaking news for {asset} for immediate trading decision."
                }
            ],
            "response_format": {"type": "json_object"}
        }

        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(url, headers=headers, json=payload, timeout=15.0)
                if res.status_code == 200:
                    data = res.json()
                    content = json.loads(data['choices'][0]['message']['content'])
                    self.intel_buffer["macro_bias"] = content.get("bias", "NEUTRAL")
                    self.intel_buffer["sentiment_score"] = content.get("confidence", 0.5) * (1 if content.get("bias") == "BULLISH" else -1)
                    print(f"✅ Perplexity Insight: {self.intel_buffer['macro_bias']} ({content.get('reason')})")
        except Exception as e:
            print(f"⚠️ Perplexity Error: {e}")

    async def poll_gemini(self, symbol: str = "BTC"):
        """Analyze local news headlines via Gemini for institutional sentiment traps"""
        if not self.gemini_key:
            return
        
        try:
            # 1. Fetch current news from API
            async with httpx.AsyncClient() as client:
                res = await client.get(f"{self.api_base}/api/news?symbol={symbol}")
                if res.status_code != 200: return
                headlines = [h['title'] for h in res.json().get('headlines', [])]
            
            if not headlines: return
            
            print(f"🧠 {symbol}: Fusing Gemini Reasoning with {len(headlines)} headlines...")
            
            prompt = f"""
            Analyze these {symbol} headlines for 15-minute timeframe trading sentiment. 
            Identify if this is retail noise or institutional accumulation.
            Headlines: {json.dumps(headlines)}
            Output JSON: {{"sentiment": -1.0 to 1.0, "assessment": "short string"}}
            """
            
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            # Simple extractor for markdown wrapped JSON
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            
            analysis = json.loads(text)
            # Smooth the score: Average Perplexity and Gemini
            p_score = self.intel_buffer["sentiment_score"]
            g_score = analysis.get("sentiment", 0.0)
            self.intel_buffer["sentiment_score"] = (p_score + g_score) / 2
            print(f"✅ Gemini Assessment: {analysis.get('assessment')} | Final Score: {self.intel_buffer['sentiment_score']:.2f}")

        except Exception as e:
            print(f"⚠️ Gemini Error: {e}")

    async def push_to_bridge(self):
        """Broadcast the Sovereign Intelligence to the Master Bridge"""
        try:
            async with httpx.AsyncClient() as client:
                await client.post(f"{self.api_base}/api/intel/update", json=self.intel_buffer)
        except:
            pass

    async def logic_loop(self):
        """The Eternal Reasoning Loop (30s Cadence)"""
        print("🚀 SOVEREIGN MIND: Intelligence Engine IGNITED.")
        while True:
            await self.poll_perplexity("Bitcoin")
            await self.poll_gemini("BTC")
            self.intel_buffer["last_update"] = datetime.now().isoformat()
            await self.push_to_bridge()
            await asyncio.sleep(60)

scout = IntelScout()
if __name__ == "__main__":
    asyncio.run(scout.logic_loop())
