# -*- coding: utf-8 -*-
"""搜索意图两级分类器（82 主题 Qwen 修订版）的运行时封装工具。

将「分类器/」目录下的模型产物封装为可复用的单例服务，供分类节点对资讯做
「模型主分类」：给定资讯标题（作为代表搜索词）与正文（作为关联高点击标题上下文），
用 MiniLM 384 维向量与 82 个归一化主题中心做余弦相似度分配，返回模型预测的一级/二级
主题、相似度、建议人工复核（噪声）判定以及该二级主题的代表搜索词，供下游 LLM 仅做
结果校验。

模型文件（来源于 `分类器/` 目录，非重新训练，复用聚类标签分配新词）：
- prototypes_82.npz：82 个主题中心向量、主题 ID 与训练样本数
- topic_catalog.json：一级/二级主题名称及代表搜索词
- model_metadata.json：聚类方法、模型版本、低置信度复核阈值与训练来源

依赖：
- sentence-transformers / numpy / torch（需安装）
- embedding 模型 paraphrase-multilingual-MiniLM-L12-v2
  （本地部署于 /space/models/minilm，也可经 SEARCH_INTENT_EMBEDDING_MODEL 指定）
"""
import importlib as _importlib
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# language server 静态类型检查不追溯处理、也不真正 import 重型库(sentence-transformers/torch)。
# 这里改为"运行时按需惰性 import"，避免 LSP 诊断在分析 src 时因追溯 torch 等重型库而过慢/超时；
# 真正加载发生在首次调用 get_classifier()（懒加载单例）之时。
_SENTENCE_TRANSFORMER_MODULE: Optional[Any] = None


def _get_sentence_transformer():
    """惰性加载 sentence_transformers 的 SentenceTransformer 类（避免静态分析追溯 torch）。"""
    global _SENTENCE_TRANSFORMER_MODULE
    if _SENTENCE_TRANSFORMER_MODULE is None:
        _SENTENCE_TRANSFORMER_MODULE = _importlib.import_module("sentence_transformers")
    return _SENTENCE_TRANSFORMER_MODULE.SentenceTransformer

# 官方 huggingface.co 沙箱不通，走镜像源（需在 sentence-transformers/huggingface 加载前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

logger = logging.getLogger(__name__)


class SearchIntentClassifier:
    """82 主题最近质心分类器（L2 归一化 MiniLM 向量 + 余弦相似度分配）。

    复用已持久化模型产物的预测口径：
    - 推理文本格式：`搜索词：{title}。相关高点击资讯标题：{content}`（无上下文时仅搜索词）
    - 取与 82 个归一化主题中心中余弦相似度最高者作为预测
    - 建议人工复核：最高相似度低于阈值，或 top2 差距过小，或该主题训练样本过少
    """

    def __init__(self, model_dir: Path, device: str = "cpu") -> None:
        self.model_dir = Path(model_dir)
        metadata_path = self.model_dir / "model_metadata.json"
        catalog_path = self.model_dir / "topic_catalog.json"
        prototype_path = self.model_dir / "prototypes_82.npz"
        for p in (metadata_path, catalog_path, prototype_path):
            if not p.exists():
                raise FileNotFoundError(f"缺少分类器模型文件: {p}")

        self.metadata: Dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        catalog: Dict[str, Any] = json.loads(catalog_path.read_text(encoding="utf-8"))
        self.parent_topics: Dict[str, Any] = catalog.get("parent_topics", {}) or {}
        self.leaf_topics: Dict[str, Any] = catalog.get("leaf_topics", {}) or {}

        with np.load(prototype_path) as saved:
            self.topic_ids = np.asarray(saved["topic_ids"], dtype=np.int32)
            self.centroids = np.asarray(saved["normalized_centroids"], dtype=np.float32)
            self.training_counts = np.asarray(saved["training_counts"], dtype=np.int32)

        project_local_model = Path(__file__).resolve().parents[2] / ".models" / "minilm"
        model_path = os.environ.get("SEARCH_INTENT_EMBEDDING_MODEL")
        if not model_path and project_local_model.exists():
            model_path = str(project_local_model)
        model_path = model_path or self.metadata.get("embedding_model", "")
        if not model_path or str(model_path).startswith("/space/"):
            model_path = DEFAULT_MODEL_NAME
        if model_path:
            source_path = Path(model_path)
            if source_path.exists():
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                model_path = str(source_path)
        self.encoder = _get_sentence_transformer()(model_path, device=device)

        self.low_similarity_cutoff = float(self.metadata.get("low_similarity_review_cutoff", 0.0))
        self.min_support = int(self.metadata.get("minimum_support_for_auto_review_clearance", 3))
        self.min_margin = float(self.metadata.get("minimum_top_two_margin_for_auto_review_clearance", 0.02))
        self.top_k = int(self.metadata.get("nearby_top_k", 5) or 5)

    def classify(self, titles, contexts=None) -> list:
        titles = [str(t) for t in titles]
        if contexts is None:
            contexts = [""] * len(titles)
        documents = [
            f"搜索词：{term}。相关高点击资讯标题：{context}" if context else f"搜索词：{term}"
            for term, context in zip(titles, contexts)
        ]
        vectors = np.asarray(self.encoder.encode(
            documents, batch_size=64, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True,
        ), dtype=np.float32)
        scores = vectors @ self.centroids.T
        order = np.argsort(-scores, axis=1)
        results = []
        for row in range(len(documents)):
            best_col, second_col = int(order[row, 0]), int(order[row, 1])
            topic_id = int(self.topic_ids[best_col])
            second_id = int(self.topic_ids[second_col])
            topic = self.leaf_topics.get(str(topic_id), {}) or {}
            second_topic = self.leaf_topics.get(str(second_id), {}) or {}
            similarity = float(scores[row, best_col])
            margin = similarity - float(scores[row, second_col])
            support = int(self.training_counts[best_col])
            review = (
                similarity < self.low_similarity_cutoff
                or margin < self.min_margin
                or support < self.min_support
            )
            review_reasons = []
            if similarity < self.low_similarity_cutoff:
                review_reasons.append("low_similarity")
            if margin < self.min_margin:
                review_reasons.append("close_candidates")
            if support < self.min_support:
                review_reasons.append("low_support")
            top_k = max(1, int(self.top_k))
            top_k = min(top_k, int(self.centroids.shape[0]))
            nearby = []
            for j in range(top_k):
                col = int(order[row, j])
                tid = int(self.topic_ids[col])
                tp = self.leaf_topics.get(str(tid), {}) or {}
                nearby.append({
                    "leaf_topic_id": tid,
                    "leaf_topic_name": str(tp.get("leaf_topic_name", "")),
                    "parent_topic_name": str(tp.get("parent_topic_name", "")),
                    "centroid_cosine_similarity": float(scores[row, col]),
                    "representative_searches": list(tp.get("representative_searches", []) or []),
                })
            results.append({
                "leaf_topic_id": topic_id,
                "leaf_topic_name": str(topic.get("leaf_topic_name", "")),
                "parent_topic_id": int(topic.get("parent_topic_id", -1)),
                "parent_topic_name": str(topic.get("parent_topic_name", "")),
                "second_topic_id": second_id,
                "second_topic_name": str(second_topic.get("leaf_topic_name", "")),
                "centroid_cosine_similarity": similarity,
                "cosine_margin_to_second_best": margin,
                "training_examples_in_topic": support,
                "manual_review_recommended": bool(review),
                "review_reasons": review_reasons,
                "similarity_cutoff": self.low_similarity_cutoff,
                "nearby_topics": nearby,
            })
        return results


@lru_cache(maxsize=1)
def get_classifier() -> SearchIntentClassifier:
    """获取懒加载的单例分类器（首次调用才加载 embedding 模型）。"""
    root = Path(__file__).resolve().parent.parent.parent
    model_dir = root / "分类器"
    if not model_dir.exists():
        raise FileNotFoundError(f"未找到分类器模型目录: {model_dir}")
    classifier = SearchIntentClassifier(model_dir, device="cpu")
    logger.info("SearchIntentClassifier(82 topics) loaded (device=cpu)")
    return classifier


@lru_cache(maxsize=512)
def _classify_cached(title: str, content: str) -> dict:
    """按 (标题, 正文) 缓存模型分类结果，避免同一/相似资讯重复向量化与计算。

    说明：中心向量本来已持久化于 prototypes_82.npz（加载毫秒级，非耗时来源）；
    真正耗时的是 embedding 模型(MiniLM)一次性冷加载与词向量化，本缓存消除重复向量化。
    """
    classifier = get_classifier()
    result = classifier.classify([str(title)], [str(content or "")])[0]
    leaf = classifier.leaf_topics.get(str(result["leaf_topic_id"]), {}) or {}
    representative_searches = list(leaf.get("representative_searches", []) or [])
    nearby_topics = []
    for nb in result.get("nearby_topics") or []:
        near = classifier.leaf_topics.get(str(nb.get("leaf_topic_id", "")), {}) or {}
        nearby_topics.append({
            "leaf_id": int(nb.get("leaf_topic_id", -1)),
            "leaf_name": str(nb.get("leaf_topic_name", "")),
            "parent_id": int(near.get("parent_topic_id", -1)),
            "parent_name": str(nb.get("parent_topic_name", "")),
            "similarity": float(nb.get("centroid_cosine_similarity", 0.0)),
            "representative_searches": list(nb.get("representative_searches", []) or []),
        })
    return {
        "leaf_id": int(result["leaf_topic_id"]),
        "leaf_name": str(result["leaf_topic_name"]),
        "parent_id": int(result["parent_topic_id"]),
        "parent_name": str(result["parent_topic_name"]),
        "nearest_leaf_id": int(result["leaf_topic_id"]),
        "nearest_leaf_name": str(result["leaf_topic_name"]),
        "similarity": float(result["centroid_cosine_similarity"]),
        "second_best_similarity": float(result["centroid_cosine_similarity"] - result["cosine_margin_to_second_best"]),
        "cosine_gap": float(result["cosine_margin_to_second_best"]),
        "training_support": int(result["training_examples_in_topic"]),
        "similarity_cutoff": float(result["similarity_cutoff"]),
        "review_reasons": list(result["review_reasons"]),
        "is_noise": bool(result["manual_review_recommended"]),
        "top_terms": representative_searches,
        "representative_searches": representative_searches,
        "nearby_topics": nearby_topics,
    }


def classify_article(title: str, content: str = "") -> dict:
    """用 82 主题模型对单条资讯做主分类（结果按 (标题,正文) 做 LRU 缓存）。

    资讯标题作为该资讯的代表搜索词（term），正文作为关联资讯标题（context）。

    返回字段说明：
    - leaf_id / leaf_name：模型最终预测的二级主题
    - parent_id / parent_name：所属一级主题
    - similarity：与所属质心的 cosine 相似度
    - second_best_similarity / cosine_gap：次优相似度及差距
    - is_noise：是否建议人工复核（低相似度/低置信度/小样本）
    - top_terms / representative_searches：该二级主题的代表词与代表搜索词
    - nearby_topics：向量最相近的 top-k 相邻主题（供 B/D 象限拣选高曝光参考）
    """
    cached = _classify_cached(str(title), str(content or ""))
    if isinstance(cached, dict):
        import copy as _copy
        return _copy.deepcopy(cached)
    return cached


def warmup(encode: bool = True) -> bool:
    """服务启动预热：加载并常驻 embedding 模型（并可选做一次最小编码）。

    应在长驻服务启动后就绪、正式接流量前调用一次。模型加载完成后以单例常驻，
    此后每次对新资讯分类仅需 ~0.1s 的向量化编码，避免首个请求冷启动卡顿。

    encode=True 时额外跑一次最小输入编码，提前完成 torch 后端/线程池初始化。
    """
    get_classifier()
    if encode:
        try:
            get_classifier().encoder.encode(
                ["搜索词：预热。相关高点击资讯标题：预热"],
                batch_size=1,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        except Exception as exc:  # 预热编码失败不影响模型已加载
            logger.warning("warmup encode 失败（模型已加载，跳过预热编码）: %s", exc)
    logger.info("SearchIntentClassifier warmup done（embedding 模型已常驻）")
    return True
