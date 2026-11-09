"""模型分类节点：仅用搜索意图两级分类器对资讯做主分类，产出模型预测结果供 LLM 校验。"""
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from tools.runtime import Context

from graphs.state import ModelClassifyInput, ModelClassifyOutput, NearbyTopic
from tools.search_classifier import classify_article

logger = logging.getLogger(__name__)


def category_model_classify_node(
    state: ModelClassifyInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> ModelClassifyOutput:
    """
    title: 模型分类
    desc: 先用搜索意图两级分类器对资讯做主分类，得到模型预测的一级/二级主题、相似度与噪声判定，供下游 LLM 仅做结果校验；模型不可用时标记为不可用，由 LLM 校验节点自主分类兜底
    """
    ctx = runtime.context

    try:
        model_result: Any = classify_article(state.title, state.content)
    except Exception as exc:  # 模型不可用时标记不可用，由下游 LLM 节点自主分类兜底
        logger.info("模型分类不可用，交由LLM自主分类: %s", exc)
        return ModelClassifyOutput(
            model_available=False,
            model_is_noise=False,
            model_leaf_id=-1,
            model_leaf_name="",
            model_parent_name="",
            model_similarity=0.0,
            model_top_terms=[],
            model_rep_searches=[],
            model_pred_text="（此处无需模型预测，由LLM自主分类）",
            nearby_topics=[],
        )

    model_leaf_id = int(model_result.get("leaf_id", -1))
    model_pred_text = (
        f"模型预测二级主题：{model_result.get('leaf_name', '')}（leaf_id={model_leaf_id}，"
        f"所属一级主题：{model_result.get('parent_name', '')}，cosine相似度："
        f"{model_result.get('similarity', 0):.4f}，噪声判定：{model_result.get('is_noise', False)}）"
        f"\n代表关键词：{'、'.join(model_result.get('top_terms', []) or [])}"
        f"\n代表搜索词：{'、'.join(model_result.get('representative_searches', []) or [])}"
    )

    nearby_topics: list = []
    for nb in model_result.get("nearby_topics", []) or []:
        nearby_topics.append(
            NearbyTopic(
                leaf_id=int(nb.get("leaf_id", -1)),
                leaf_name=str(nb.get("leaf_name", "")),
                parent_id=int(nb.get("parent_id", -1)),
                parent_name=str(nb.get("parent_name", "")),
                similarity=float(nb.get("similarity", 0) or 0),
                representative_searches=list(nb.get("representative_searches", []) or []),
            )
        )

    return ModelClassifyOutput(
        model_available=True,
        model_is_noise=bool(model_result.get("is_noise", False)),
        model_leaf_id=model_leaf_id,
        model_leaf_name=str(model_result.get("leaf_name", "")),
        model_parent_name=str(model_result.get("parent_name", "")),
        model_similarity=float(model_result.get("similarity", 0) or 0),
        model_second_similarity=float(model_result.get("second_best_similarity", 0) or 0),
        model_cosine_gap=float(model_result.get("cosine_gap", 0) or 0),
        model_training_support=int(model_result.get("training_support", 0) or 0),
        model_similarity_cutoff=float(model_result.get("similarity_cutoff", 0.6542854905128479)),
        model_review_reasons=list(model_result.get("review_reasons", [])),
        model_top_terms=list(model_result.get("top_terms", []) or []),
        model_rep_searches=list(model_result.get("representative_searches", []) or []),
        model_pred_text=model_pred_text,
        nearby_topics=nearby_topics,
    )
