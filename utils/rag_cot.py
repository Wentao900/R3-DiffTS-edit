from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.trend_prior import build_trend_fields

try:
    from transformers import pipeline
except ImportError:  # pragma: no cover
    pipeline = None


@dataclass
class RAGCoTConfig:
    use_rag_cot: bool = False
    use_retrieval: bool = True
    generate_cot: bool = True
    rag_topk: int = 3
    use_two_stage_rag: bool = False
    rag_stage1_topk: int = 6
    rag_stage2_topk: int = 3
    two_stage_gate: bool = True
    trend_slope_eps: float = 1.0e-3
    cot_model: Optional[str] = None
    cot_max_new_tokens: int = 96
    cot_temperature: float = 0.7
    cot_cache_size: int = 512
    cot_device: Optional[str] = None


class RAGCoTPipeline:
    def __init__(
        self,
        domain_desc: str,
        search_df: Optional[pd.DataFrame],
        lookback_len: int,
        pred_len: int,
        config: Optional[RAGCoTConfig] = None,
    ) -> None:
        self.domain_desc = domain_desc
        self.lookback_len = lookback_len
        self.pred_len = pred_len
        self.config = config or RAGCoTConfig()
        self.cache: OrderedDict[str, Dict[str, object]] = OrderedDict()
        self.search_df = self._prepare_search_df(search_df)
        self.vectorizer = None
        self.matrix = None
        if self.config.use_retrieval and not self.search_df.empty:
            self.vectorizer = TfidfVectorizer(stop_words="english", max_features=4096, ngram_range=(1, 2))
            self.matrix = self.vectorizer.fit_transform(self.search_df["fact"].tolist())
        self.generator = self._build_generator()

    def _prepare_search_df(self, search_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if search_df is None:
            return pd.DataFrame(columns=["fact", "start_date", "end_date"])
        prepared = search_df.copy()
        prepared["fact"] = prepared["fact"].fillna("").astype(str)
        return prepared.loc[prepared["fact"].str.len() > 0].reset_index(drop=True)

    def _build_generator(self):
        if pipeline is None or not self.config.cot_model:
            return None
        device = -1
        if self.config.cot_device and self.config.cot_device.startswith("cuda"):
            device = int(self.config.cot_device.split(":")[1]) if ":" in self.config.cot_device else 0
        return pipeline("text-generation", model=self.config.cot_model, tokenizer=self.config.cot_model, device=device)

    def _numeric_summary(self, values: Sequence[float]) -> str:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            return "no numeric history"
        slope = float(arr[-1] - arr[0]) / max(arr.size - 1, 1)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        return f"mean={mean:.4f}; std={std:.4f}; slope={slope:.4f}; last={arr[-1]:.4f}"

    def _retrieve(self, query: str, topk: int, allowed_indices: Optional[Sequence[int]] = None) -> List[str]:
        if self.vectorizer is None or self.matrix is None or topk <= 0:
            return []
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix).ravel()
        if sims.size == 0:
            return []
        if allowed_indices is not None:
            mask = np.full_like(sims, fill_value=-np.inf, dtype=float)
            allowed_indices = np.asarray(allowed_indices, dtype=int)
            if allowed_indices.size == 0:
                return []
            mask[allowed_indices] = sims[allowed_indices]
            sims = mask
        top_indices = sims.argsort()[::-1][:topk]
        return [
            self.search_df.iloc[idx]["fact"]
            for idx in top_indices
            if np.isfinite(sims[idx]) and sims[idx] > 0 and str(self.search_df.iloc[idx]["fact"]).strip()
        ]

    def _fallback_cot(self, numeric_summary: str, retrieved: Sequence[str]) -> str:
        evidence = " ".join(retrieved[:2]) if retrieved else "No strong evidence retrieved."
        return (
            "Trend reasoning: "
            f"numeric window suggests {numeric_summary}. "
            f"Supporting evidence: {evidence}. "
            "Infer the likely short-term direction, strength, and volatility before forecasting."
        )

    def _generate_cot(self, numeric_summary: str, retrieved: Sequence[str]) -> str:
        if not self.config.generate_cot:
            return ""
        if self.generator is None:
            return self._fallback_cot(numeric_summary, retrieved)
        prompt = (
            f"{self.domain_desc}\n"
            f"Numeric summary: {numeric_summary}\n"
            f"Retrieved evidence: {' '.join(retrieved) if retrieved else 'None'}\n"
            "Return a short reasoning paragraph, then a compact JSON with keys direction, strength, volatility."
        )
        outputs = self.generator(
            prompt,
            max_new_tokens=self.config.cot_max_new_tokens,
            do_sample=self.config.cot_temperature > 0,
            temperature=max(self.config.cot_temperature, 1.0e-5),
            num_return_sequences=1,
        )
        generated = outputs[0]["generated_text"]
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()
        return generated or self._fallback_cot(numeric_summary, retrieved)

    def _compose_text(
        self,
        raw_text: str,
        numeric_summary: str,
        retrieved: Sequence[str],
        cot_text: str,
    ) -> str:
        blocks = [self.domain_desc]
        if raw_text and raw_text != "NA":
            blocks.append(f"[RAW TEXT] {raw_text}")
        blocks.append(f"[NUMERIC SUMMARY] {numeric_summary}")
        if retrieved:
            blocks.append("[RETRIEVED EVIDENCE] " + " ".join(retrieved))
        if cot_text:
            blocks.append("[COT] " + cot_text)
        return " ".join(blocks)

    def _trend_hypothesis_query(self, cot_text: str, numeric_history: Sequence[float]) -> str:
        fields = build_trend_fields(cot_text, numeric_history, slope_eps=self.config.trend_slope_eps)
        return (
            f"future direction {fields['direction']} "
            f"trend strength {fields['strength']} "
            f"volatility {fields['volatility']}"
        )

    def _cache_get(self, key: str) -> Optional[Dict[str, object]]:
        cached = self.cache.get(key)
        if cached is not None:
            self.cache.move_to_end(key)
        return cached

    def _cache_put(self, key: str, value: Dict[str, object]) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        while len(self.cache) > self.config.cot_cache_size:
            self.cache.popitem(last=False)

    def build(
        self,
        raw_text: str,
        numeric_history: Sequence[float],
        allowed_indices: Optional[Sequence[int]] = None,
    ) -> Dict[str, object]:
        cache_key = f"{raw_text}|||{np.asarray(numeric_history, dtype=float).round(6).tolist()}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        numeric_summary = self._numeric_summary(numeric_history)
        base_query = f"{raw_text} {numeric_summary}".strip()
        if self.config.use_rag_cot and self.config.use_retrieval:
            if self.config.use_two_stage_rag:
                stage1 = self._retrieve(base_query, self.config.rag_stage1_topk, allowed_indices=allowed_indices)
                hypothesis_seed = self._generate_cot(numeric_summary, stage1)
                stage2_query = self._trend_hypothesis_query(hypothesis_seed, numeric_history)
                should_gate = self.config.two_stage_gate and raw_text == "NA" and abs(float(np.asarray(numeric_history)[-1] - np.asarray(numeric_history)[0])) < self.config.trend_slope_eps
                stage2 = [] if should_gate else self._retrieve(stage2_query, self.config.rag_stage2_topk, allowed_indices=allowed_indices)
                retrieved = list(dict.fromkeys(stage1 + stage2))[: max(self.config.rag_topk, self.config.rag_stage2_topk)]
                cot_text = hypothesis_seed
            else:
                retrieved = self._retrieve(base_query, self.config.rag_topk, allowed_indices=allowed_indices)
                cot_text = self._generate_cot(numeric_summary, retrieved)
        elif self.config.use_rag_cot:
            retrieved = []
            cot_text = self._generate_cot(numeric_summary, retrieved)
        else:
            retrieved = []
            cot_text = ""

        text = self._compose_text(raw_text, numeric_summary, retrieved, cot_text) if self.config.use_rag_cot else raw_text
        result = {
            "text": text,
            "cot_text": cot_text,
            "numeric_summary": numeric_summary,
            "retrieved": retrieved,
        }
        self._cache_put(cache_key, result)
        return result
