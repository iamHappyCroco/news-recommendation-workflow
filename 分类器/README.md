# 完整工作流的分类模型产物

本目录保留跑批使用的原生 82 主题模型产物：

- `prototypes_82.npz`：主题中心向量、主题 ID 和训练样本数。
- `topic_catalog.json`：一级／二级主题及代表搜索词，供模型分类、LLM 校验和兜底使用。
- `model_metadata.json`：模型版本、相似度门槛及复核参数。

预测实现为 `src/tools/search_classifier.py`，使用 `.models/minilm` 编码资讯，再与归一化主题中心计算余弦相似度。相似度是排序信号，不是概率。

原训练与单独搜索词分类脚本已清理。完整资讯跑批使用 `scripts/batch_run.py` 或 `scripts/batch_final_summary.py`，会继续经过 LLM 校验和后续生词。
