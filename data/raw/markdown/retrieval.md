# Retrieval

## Ranking

Dense and sparse results are fused before structural filtering.

```python
def rank(results):
    return sorted(results, key=lambda item: item.score, reverse=True)
```
