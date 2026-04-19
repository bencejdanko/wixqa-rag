# wixqa-rag

### Baseline RAG Implementation

Table: Baseline RAG System Configuration
|  | Configuration |
| --- | --- |
| Embedding model | all-MiniLM-L6-v2 |
| Vector database | PgVector | 
| Retriever type | HNSW |
| Generator model | gpt-oss-120b |
| Prompting Strategy | Zero-shot |

: Baseline system configuration for our baseline RAG implementation.

Table: WixQA-Synthetic Baseline results
| Chunk Size | Overlap | k | Context Recall | F1 | ROUGE-1 | Observation |
| --- | --- | --- | --- | --- | --- | --- |
| 300 | 30 | 1 | 0.4064 | 0.34138 | 0.42498 | |
| 300 | 30 | 3 | 0.5613 | 0.34728 | 0.43660 | |
| 300 | 30 | 5 | 0.6521 | 0.34405 | 0.43763 | |
| 300 | 30 | 10 | 0.7444 | 0.33869 | 0.43627 | |
| 300 | 60 | 1 | 0.4083 | 0.34304 | 0.42647 | |
| 300 | 60 | 3 | 0.5687 | 0.35029 | 0.44016 | |
| 300 | 60 | 5 | 0.6554 | 0.34648 | 0.44046 | |
| 300 | 60 | 10 | 0.7553 | 0.34026 | 0.43758 | |
| 500 | 30 | 1 | 0.4773 | 0.35659 | 0.44119 | Less overlap does not appear to significantly impact performance. |
| 500 | 30 | 3 | 0.6512 | 0.35566 | 0.44981 | Consistent performance gains with 60 overlap strategy |
| 500 | 30 | 5 | 0.7324 | 0.35313 | 0.45057 | |
| 500 | 30 | 10 | 0.8132 | 0.34613 | 0.44557 | |
| 500 | 60 | 1 | 0.4790 | 0.35658 | 0.44194 | |
| 500 | 60 | 3 | 0.6560 | 0.3574 | 0.4512 | |
| 500 | 60 | 5 | 0.7348 | 0.35291 | 0.45046 | |
| 500 | 60 | 10 | **0.8158** | 0.34672 | 0.44699 | Higher k consistently raises percieved judge recall score. |

: Full results across all attempted chunking and overlap strategies. At higher k={5,10}, no more than ~0.5% of API calls were dropped due to API limits.

- **Context Recall**: Measures whether the retrieved context contains the necessary
information to answer the question. We use LLM as a judge with temperature 0. We provide the 
judge the query, the ground-truth answer, and the retrieved context to the
LLM judge to determine whether the retrieved context contains sufficient information to
answer the query correctly by returning binary answer: Yes or NO.
- **F1 Score**: Measures token overlap between the generated answer and the ground-truth
answer.
- **ROUGE-1**: Measures unigram overlaps and content coverage.
