"""资讯推荐位出词工作流主图（单条资讯完整处理链路，平铺展示，画布可见全部节点）。

topic_recall → topic_fit_judge →(命中专题先出专题词)→ hot_theme_recall → hot_theme_fit_judge →(适合则 hot_theme_word_generate 出热点题材词)→ category_model_classify → category_llm_verify →(按四象限)→ quadrant_a~d
其中：专题流程与东方财富「热点题材」流程均为前置必走链路，产出各自的额外搜索词（topic_hit_result / hot_theme_hit_result）后统一汇入模型分类→象限生词；
B/D 低曝光象限经 hot_topic_select 拣选相邻高曝光参照后再生词；无法归类也进入热主题拣选，经 route_after_hot（region 非 B/D）导向 AI 生词兜底。
"""
from langgraph.graph import StateGraph, END

from graphs.state import (
    GlobalState,
    GraphInput,
    GraphOutput,
    TopicFitRoute,
    RegionRoute,
    HasHotThemeFit,
)
from graphs.nodes.topic_recall_node import topic_recall_node
from graphs.nodes.topic_fit_judge_node import topic_fit_judge_node
from graphs.nodes.topic_block_judge_node import topic_block_judge_node
from graphs.nodes.topic_word_generate_node import topic_word_generate_node
from graphs.nodes.hot_theme_recall_node import hot_theme_recall_node
from graphs.nodes.hot_theme_fit_judge_node import hot_theme_fit_judge_node
from graphs.nodes.hot_theme_block_judge_node import hot_theme_block_judge_node
from graphs.nodes.hot_theme_word_generate_node import hot_theme_word_generate_node
from graphs.nodes.category_model_classify_node import category_model_classify_node
from graphs.nodes.category_llm_verify_node import category_llm_verify_node
from graphs.nodes.category_llm_fallback_node import category_llm_fallback_node, route_category_confidence
from graphs.nodes.category_candidate_generate_node import category_candidate_generate_node
from graphs.nodes.quadrant_a_node import quadrant_a_node
from graphs.nodes.quadrant_b_node import quadrant_b_node
from graphs.nodes.quadrant_c_node import quadrant_c_node
from graphs.nodes.quadrant_d_node import quadrant_d_node
from graphs.nodes.hot_topic_select_node import hot_topic_select_node
from graphs.nodes.ai_generate_node import ai_generate_node
from graphs.nodes.priority_judge_node import priority_judge_node
from graphs.nodes.final_summary_node import final_summary_node


def route_topic_fit(state: TopicFitRoute) -> str:
    """根据是否存在适配专题决定后续走向。"""
    if state.has_fit_topic:
        return "采用专题"
    return "生成推荐词"


def route_hot_theme_fit(state: HasHotThemeFit) -> str:
    """根据是否存在适合跳转的热点题材决定是否生成热点题材相关搜索词，随后统一进入模型分类。"""
    if state.has_fit_hot_theme:
        return "生成热点词"
    return "进模型分类"


def route_region(state: RegionRoute) -> str:
    """根据能否归类及类目所属四象限区域，路由到对应生词策略。

    低曝光象限（B/D）先经「相邻高曝光主题拣选」获取参照后再生词；
    无法归类/异常区域的资讯同样进入热主题拣选，经 route_after_hot（region 非 B/D）导向 AI 生词兜底。
    """
    if not state.can_classify:
        return "AI生词"
    if state.region == "A":
        return "象限A"
    if state.region == "C":
        return "象限C"
    return "热主题拣选"


def route_after_hot(state: RegionRoute) -> str:
    """热主题拣选完成后，按原区域回到对应低曝光象限生词；异常区域走 AI 生词兜底。"""
    if state.region == "B":
        return "象限B"
    if state.region == "D":
        return "象限D"
    return "AI生词"


def _build_graph() -> StateGraph:
    builder = StateGraph(GlobalState, input_schema=GraphInput, output_schema=GraphOutput)

    # Agent 节点：通过 metadata 注入各自模型配置文件
    builder.add_node("topic_recall", topic_recall_node)
    builder.add_node("topic_fit_judge", topic_fit_judge_node, metadata={"type": "agent", "llm_cfg": "config/topic_fit_judge_llm_cfg.json"})
    builder.add_node("topic_block", topic_block_judge_node)
    builder.add_node("topic_word_generate", topic_word_generate_node, metadata={"type": "agent", "llm_cfg": "config/topic_word_generate_llm_cfg.json"})
    builder.add_node("hot_theme_recall", hot_theme_recall_node)
    builder.add_node("hot_theme_fit_judge", hot_theme_fit_judge_node, metadata={"type": "agent", "llm_cfg": "config/hot_theme_fit_judge_llm_cfg.json"})
    builder.add_node("hot_theme_block", hot_theme_block_judge_node)
    builder.add_node("hot_theme_word_generate", hot_theme_word_generate_node, metadata={"type": "agent", "llm_cfg": "config/hot_theme_word_generate_llm_cfg.json"})
    builder.add_node("category_model_classify", category_model_classify_node)
    builder.add_node("category_llm_fallback", category_llm_fallback_node, metadata={"type": "agent", "llm_cfg": "config/category_fallback_llm_cfg.json"})
    builder.add_node("category_llm_verify", category_llm_verify_node, metadata={"type": "agent", "llm_cfg": "config/category_classify_llm_cfg.json"})
    builder.add_node("category_candidate_generate", category_candidate_generate_node)
    builder.add_node("quadrant_a", quadrant_a_node, metadata={"type": "agent", "llm_cfg": "config/quadrant_a_llm_cfg.json"})
    builder.add_node("quadrant_b", quadrant_b_node, metadata={"type": "agent", "llm_cfg": "config/quadrant_b_llm_cfg.json"})
    builder.add_node("quadrant_c", quadrant_c_node, metadata={"type": "agent", "llm_cfg": "config/quadrant_c_llm_cfg.json"})
    builder.add_node("quadrant_d", quadrant_d_node, metadata={"type": "agent", "llm_cfg": "config/quadrant_d_llm_cfg.json"})
    builder.add_node("hot_topic_select", hot_topic_select_node, metadata={"type": "agent", "llm_cfg": "config/hot_topic_select_llm_cfg.json"})
    builder.add_node("ai_generate", ai_generate_node, metadata={"type": "agent", "llm_cfg": "config/ai_generate_llm_cfg.json"})
    builder.add_node("priority_judge", priority_judge_node, metadata={"type": "agent", "llm_cfg": "config/priority_llm_cfg.json"})
    builder.add_node("final_summary", final_summary_node)

    builder.set_entry_point("topic_recall")
    builder.add_edge("topic_recall", "topic_fit_judge")
    builder.add_edge("topic_fit_judge", "topic_block")

    builder.add_conditional_edges(
        source="topic_block",
        path=route_topic_fit,
        path_map={
            "采用专题": "topic_word_generate",
            "生成推荐词": "hot_theme_recall",
        },
    )

    # 专题适配时：先输出专题相关搜索词，随后仍进入热点题材必走流程
    builder.add_edge("topic_word_generate", "hot_theme_recall")

    # 热点题材必走流程：召回→判断→(规则屏蔽嵌套跳转)→(适合则生成热点题材搜索词)→优先出词判断→模型分类
    builder.add_edge("hot_theme_recall", "hot_theme_fit_judge")
    builder.add_edge("hot_theme_fit_judge", "hot_theme_block")
    builder.add_conditional_edges(
        source="hot_theme_block",
        path=route_hot_theme_fit,
        path_map={
            "生成热点词": "hot_theme_word_generate",
            "进模型分类": "priority_judge",
        },
    )
    builder.add_edge("hot_theme_word_generate", "priority_judge")

    # 优先出词判断（专题 vs 热点）完成后，进入模型分类→按象限生词
    builder.add_edge("priority_judge", "category_model_classify")

    builder.add_conditional_edges("category_model_classify", route_category_confidence, {
        "fallback": "category_llm_fallback", "verify": "category_llm_verify",
    })
    builder.add_edge("category_llm_fallback", "category_llm_verify")

    builder.add_conditional_edges(
        source="category_llm_verify",
        path=route_region,
        path_map={
            "象限A": "quadrant_a",
            "热主题拣选": "hot_topic_select",
            "象限C": "quadrant_c",
            "AI生词": "ai_generate",
        },
    )

    builder.add_conditional_edges(
        source="hot_topic_select",
        path=route_after_hot,
        path_map={
            "象限B": "quadrant_b",
            "象限D": "quadrant_d",
            "AI生词": "ai_generate",
        },
    )

    builder.add_edge("quadrant_a", "category_candidate_generate")
    builder.add_edge("quadrant_b", "category_candidate_generate")
    builder.add_edge("quadrant_c", "category_candidate_generate")
    builder.add_edge("quadrant_d", "category_candidate_generate")
    builder.add_edge("category_candidate_generate", "final_summary")
    builder.add_edge("ai_generate", "final_summary")
    builder.add_edge("final_summary", END)

    return builder


main_graph = _build_graph().compile()
