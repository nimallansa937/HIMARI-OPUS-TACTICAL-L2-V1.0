"""
HIMARI Layer 2 - LLM Signal Integration
Subsystem J: LLM Artifact Injector (Methods J1-J5)

Purpose:
    Extract trading signals from financial news and charts using open-source LLMs.
    Provides sentiment, event classification, and reasoning embeddings.

Methods:
    J1: Open-Source Financial LLMs - FinLLaVA, FinGPT, Qwen2.5-7B
    J2: Structured Signal Extraction - JSON sentiment output
    J3: Event Classification - News categorization (FED, CRYPTO, TECH, etc.)
    J4: RAG Knowledge Base - Hallucination prevention via grounded retrieval
    J5: Asynchronous Processing - Batch every 5 minutes to amortize latency

Expected Performance:
    - Sentiment accuracy: 70-75% (vs human labels)
    - Latency: 2-5 seconds per batch (amortized over 5 minutes)
    - False positive rate: <15%
    - Event classification accuracy: 80-85%
"""

from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import numpy as np
import asyncio
import json
from datetime import datetime
from loguru import logger

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available - LLM integration will be disabled")


@dataclass
class LLMConfig:
    """Configuration for LLM artifact injector"""
    # Method J1: Model selection
    model_name: str = "TheFinAI/FinLLaVA"  # Primary financial LLM
    fallback_model: str = "Qwen/Qwen2.5-7B-Instruct"  # Fallback if FinLLaVA unavailable

    # Method J2: Structured extraction
    max_tokens: int = 512
    temperature: float = 0.1  # Low temperature for deterministic output

    # Method J3: Event classification
    event_categories: List[str] = None

    # Method J4: RAG
    use_rag: bool = True
    knowledge_base_path: Optional[str] = None

    # Method J5: Async processing
    batch_interval_seconds: int = 300  # 5 minutes
    confidence_threshold: float = 0.85  # Only use high-confidence signals

    # Resource limits
    max_vram_gb: float = 16.0
    device: str = "cuda"

    def __post_init__(self):
        if self.event_categories is None:
            self.event_categories = [
                "FED_POLICY",
                "CRYPTO_REGULATION",
                "TECH_EARNINGS",
                "MACRO_DATA",
                "EXCHANGE_NEWS",
                "WHALE_ACTIVITY",
                "PROTOCOL_UPDATE",
                "SECURITY_BREACH"
            ]


@dataclass
class LLMSignal:
    """Structured LLM output (Method J2)"""
    sentiment_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    event_category: str  # From LLMConfig.event_categories
    reasoning: str  # Text explanation
    reasoning_embedding: Optional[np.ndarray] = None  # 768-dim embedding
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class LLMArtifactInjector:
    """
    LLM-based signal extraction for trading.

    Example:
        >>> injector = LLMArtifactInjector()
        >>> signal = await injector.process_news("Fed raises rates by 25bps")
        >>> print(signal.sentiment_score, signal.event_category)
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.batch_queue: List[str] = []
        self.is_initialized = False

        # Method J4: RAG knowledge base
        self.knowledge_base = self._init_knowledge_base()
        self._faiss_index = None
        self._kb_embeddings = None

        logger.info(f"LLM Injector initialized with model: {self.config.model_name}")
    
    def _init_knowledge_base(self) -> Dict[str, List[str]]:
        """Initialize RAG knowledge base with financial domain knowledge."""
        return {
            "FED_POLICY": [
                "Federal Reserve rate decisions impact all asset classes. Rate hikes typically strengthen USD and pressure crypto.",
                "FOMC meetings occur 8 times per year. Forward guidance matters as much as the decision.",
                "Quantitative tightening reduces liquidity, historically bearish for risk assets."
            ],
            "CRYPTO_REGULATION": [
                "SEC enforcement actions create short-term volatility but may signal long-term clarity.",
                "Spot ETF approvals have historically led to significant price appreciation.",
                "Country-level bans have limited long-term impact unless representing major markets."
            ],
            "TECH_EARNINGS": [
                "Crypto-exposed tech companies (MSTR, COIN) can move BTC prices after hours.",
                "Semiconductor supply chain issues can impact mining profitability.",
                "FAANG earnings influence overall risk sentiment in markets."
            ],
            "MACRO_DATA": [
                "CPI above expectations typically bearish for risk assets in short term.",
                "NFP misses can signal recession fears, initially bearish then bullish on rate cut hopes.",
                "GDP surprises influence Fed policy expectations."
            ],
            "WHALE_ACTIVITY": [
                "Large exchange inflows often precede selling pressure.",
                "Wallet dormancy breaks can signal long-term holder distribution.",
                "Miner outflows to exchanges historically precede corrections."
            ],
            "SECURITY_BREACH": [
                "Exchange hacks create immediate selling pressure on affected tokens.",
                "DeFi exploits can cascade to related protocols and liquidity pools.",
                "Bridge hacks have systemic implications across L2s."
            ],
            "PROTOCOL_UPDATE": [
                "Major upgrades (forks, EIPs) can create volatility around implementation.",
                "Successful upgrades often lead to appreciation within 2-4 weeks.",
                "Failed or delayed upgrades significantly damage token sentiment."
            ]
        }
    
    def _retrieve_context(self, query: str, top_k: int = 2) -> str:
        """
        Retrieve relevant context from knowledge base (Method J4).
        
        Uses keyword matching with optional FAISS vector search fallback.
        """
        # Keyword-based category detection
        query_lower = query.lower()
        matched_contexts = []
        
        category_keywords = {
            "FED_POLICY": ["fed", "fomc", "rate", "powell", "monetary", "interest"],
            "CRYPTO_REGULATION": ["sec", "regulation", "lawsuit", "etf", "approval", "ban"],
            "TECH_EARNINGS": ["earnings", "revenue", "guidance", "nvidia", "apple", "google"],
            "MACRO_DATA": ["cpi", "inflation", "unemployment", "gdp", "jobs", "nfp"],
            "WHALE_ACTIVITY": ["whale", "exchange", "transfer", "wallet", "moved"],
            "SECURITY_BREACH": ["hack", "exploit", "breach", "stolen", "drained"],
            "PROTOCOL_UPDATE": ["upgrade", "fork", "eip", "update", "launch", "mainnet"]
        }
        
        # Find matching categories
        for category, keywords in category_keywords.items():
            if any(kw in query_lower for kw in keywords):
                if category in self.knowledge_base:
                    matched_contexts.extend(self.knowledge_base[category][:top_k])
        
        # If no keyword match, try FAISS if available
        if not matched_contexts:
            matched_contexts = self._faiss_search(query, top_k)
        
        # Format context
        if matched_contexts:
            context = "\nRelevant context:\n" + "\n".join(f"- {c}" for c in matched_contexts[:top_k]) + "\n"
            return context
        
        return ""
    
    def _faiss_search(self, query: str, top_k: int = 2) -> List[str]:
        """Search knowledge base using FAISS vector similarity."""
        try:
            import faiss
            
            # Build index if not exists
            if self._faiss_index is None:
                self._build_faiss_index()
            
            if self._faiss_index is None:
                return []
            
            # Get query embedding
            query_emb = self._get_text_embedding(query)
            if query_emb is None:
                return []
            
            # Search
            query_emb = np.array([query_emb]).astype('float32')
            distances, indices = self._faiss_index.search(query_emb, top_k)
            
            # Get results
            all_docs = []
            for docs in self.knowledge_base.values():
                all_docs.extend(docs)
            
            return [all_docs[i] for i in indices[0] if i < len(all_docs)]
            
        except ImportError:
            return []
        except Exception as e:
            logger.debug(f"FAISS search failed: {e}")
            return []
    
    def _build_faiss_index(self):
        """Build FAISS index from knowledge base."""
        try:
            import faiss
            
            all_docs = []
            for docs in self.knowledge_base.values():
                all_docs.extend(docs)
            
            if not all_docs:
                return
            
            # Get embeddings for all docs
            embeddings = []
            for doc in all_docs:
                emb = self._get_text_embedding(doc)
                if emb is not None:
                    embeddings.append(emb)
            
            if not embeddings:
                return
            
            self._kb_embeddings = np.array(embeddings).astype('float32')
            self._faiss_index = faiss.IndexFlatL2(self._kb_embeddings.shape[1])
            self._faiss_index.add(self._kb_embeddings)
            logger.info(f"Built FAISS index with {len(embeddings)} documents")
            
        except Exception as e:
            logger.debug(f"Failed to build FAISS index: {e}")
    
    def _get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text using sentence-transformers or fallback."""
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, '_embed_model'):
                self._embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            return self._embed_model.encode(text)
        except ImportError:
            # Fallback: simple hash-based embedding
            words = text.lower().split()
            emb = np.zeros(64)
            for word in words:
                emb[hash(word) % 64] += 1
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb
        except Exception:
            return None

    def initialize(self):
        """Load LLM model (Method J1)"""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers not installed - LLM features disabled")
            return False

        try:
            # Try primary model
            logger.info(f"Loading {self.config.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            self.is_initialized = True
            logger.info(f"✓ Loaded {self.config.model_name}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load {self.config.model_name}: {e}")

            # Try fallback model
            try:
                logger.info(f"Falling back to {self.config.fallback_model}...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.fallback_model)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.fallback_model,
                    torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                    device_map="auto"
                )
                self.is_initialized = True
                logger.info(f"✓ Loaded fallback model {self.config.fallback_model}")
                return True

            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                self.is_initialized = False
                return False

    def _build_prompt(self, news_text: str, use_rag: bool = True) -> str:
        """Build structured prompt for sentiment extraction (Method J2)"""
        # Method J4: RAG context injection
        rag_context = ""
        if use_rag and self.config.use_rag:
            rag_context = self._retrieve_context(news_text)

        prompt = f"""You are a financial sentiment analyzer for cryptocurrency trading.

{rag_context}
Analyze the following news and provide a structured response in JSON format.

News: {news_text}

Provide your analysis in this exact JSON format:
{{
    "sentiment_score": <float between -1.0 (very bearish) and 1.0 (very bullish)>,
    "confidence": <float between 0.0 and 1.0>,
    "event_category": "<one of: {', '.join(self.config.event_categories)}>",
    "reasoning": "<brief explanation>"
}}

JSON response:"""

        return prompt

    def _parse_llm_output(self, output_text: str) -> Optional[Dict[str, Any]]:
        """Parse LLM JSON output (Method J2)"""
        try:
            # Extract JSON from response
            # Handle cases where LLM adds extra text before/after JSON
            start_idx = output_text.find('{')
            end_idx = output_text.rfind('}')

            if start_idx == -1 or end_idx == -1:
                logger.warning("No JSON found in LLM output")
                return None

            json_str = output_text[start_idx:end_idx+1]
            parsed = json.loads(json_str)

            # Validate required fields
            required = ["sentiment_score", "confidence", "event_category", "reasoning"]
            if not all(k in parsed for k in required):
                logger.warning(f"Missing required fields in LLM output: {parsed}")
                return None

            # Validate ranges
            if not (-1.0 <= parsed["sentiment_score"] <= 1.0):
                logger.warning(f"Sentiment score out of range: {parsed['sentiment_score']}")
                return None

            if not (0.0 <= parsed["confidence"] <= 1.0):
                logger.warning(f"Confidence out of range: {parsed['confidence']}")
                return None

            # Validate category (Method J3)
            if parsed["event_category"] not in self.config.event_categories:
                logger.warning(f"Unknown event category: {parsed['event_category']}")
                # Default to closest match or generic
                parsed["event_category"] = "MACRO_DATA"

            return parsed

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing LLM output: {e}")
            return None

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from last hidden state"""
        if not self.is_initialized:
            return np.zeros(768)

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            if self.config.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Use mean of last hidden state as embedding
                embedding = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()[0]

            return embedding

        except Exception as e:
            logger.warning(f"Failed to get embedding: {e}")
            return np.zeros(768)

    async def process_news(self, news_text: str) -> Optional[LLMSignal]:
        """
        Process single news item to extract trading signal.

        Args:
            news_text: News headline or article snippet

        Returns:
            LLMSignal with sentiment, confidence, category, reasoning
        """
        if not self.is_initialized:
            logger.warning("LLM not initialized - call initialize() first")
            return None

        if not news_text or len(news_text.strip()) < 10:
            logger.warning("News text too short")
            return None

        try:
            # Build prompt
            prompt = self._build_prompt(news_text)

            # Generate
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            if self.config.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the generated part (remove prompt)
            response = response[len(prompt):].strip()

            # Parse structured output (Method J2)
            parsed = self._parse_llm_output(response)
            if parsed is None:
                return None

            # Get reasoning embedding
            embedding = self._get_embedding(parsed["reasoning"])

            # Create signal
            signal = LLMSignal(
                sentiment_score=float(parsed["sentiment_score"]),
                confidence=float(parsed["confidence"]),
                event_category=parsed["event_category"],
                reasoning=parsed["reasoning"],
                reasoning_embedding=embedding
            )

            # Filter by confidence threshold
            if signal.confidence < self.config.confidence_threshold:
                logger.debug(f"Low confidence signal filtered: {signal.confidence:.2f}")
                return None

            logger.info(f"LLM signal: {signal.event_category} sentiment={signal.sentiment_score:.2f} conf={signal.confidence:.2f}")
            return signal

        except Exception as e:
            logger.error(f"Error processing news with LLM: {e}")
            return None

    async def process_batch(self, news_items: List[str]) -> List[LLMSignal]:
        """
        Process batch of news items (Method J5: Asynchronous processing).

        Args:
            news_items: List of news texts

        Returns:
            List of LLMSignal objects
        """
        signals = []

        for news in news_items:
            signal = await self.process_news(news)
            if signal is not None:
                signals.append(signal)

        logger.info(f"Processed batch: {len(signals)}/{len(news_items)} valid signals")
        return signals

    def add_to_batch_queue(self, news_text: str):
        """Add news to batch queue for async processing (Method J5)"""
        self.batch_queue.append(news_text)

    async def process_queued_batch(self) -> List[LLMSignal]:
        """Process all queued items and clear queue (Method J5)"""
        if not self.batch_queue:
            return []

        signals = await self.process_batch(self.batch_queue)
        self.batch_queue.clear()
        return signals

    def get_aggregated_sentiment(self, signals: List[LLMSignal]) -> Dict[str, float]:
        """
        Aggregate multiple signals into single sentiment score.

        Returns:
            Dict with 'sentiment', 'confidence', 'event_counts'
        """
        if not signals:
            return {"sentiment": 0.0, "confidence": 0.0, "event_counts": {}}

        # Confidence-weighted average sentiment
        total_weight = sum(s.confidence for s in signals)
        if total_weight == 0:
            weighted_sentiment = 0.0
        else:
            weighted_sentiment = sum(s.sentiment_score * s.confidence for s in signals) / total_weight

        # Average confidence
        avg_confidence = np.mean([s.confidence for s in signals])

        # Event category counts (Method J3)
        event_counts = {}
        for s in signals:
            event_counts[s.event_category] = event_counts.get(s.event_category, 0) + 1

        return {
            "sentiment": float(weighted_sentiment),
            "confidence": float(avg_confidence),
            "event_counts": event_counts
        }


# Quick export
__all__ = ['LLMArtifactInjector', 'LLMConfig', 'LLMSignal']
