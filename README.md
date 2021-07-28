# MBTI Personality Classification based on users' twitter posts
------

## Dependencies:
```shell
pip install torch pandas
```
---

### Deliverables:
---
- [x] (unit tested) utils.py : dataframe parser using reduce & singledispatch
- [ ] (currently working on...) agent.py : NLPAgent(kwargs) | Just build constructor & tokenizer based on user kwargs

### Preprocessing strategy:
- [x] 50 posts per user, split by "|||" delimiter.
- [ ] So 1st use TweetTokenizer.tokenize (to handle links) -> then join strings back and then use BertweetTokenizer : normalization=True, fast=False (Which will normalize tweets (add URL tokens) and use then bpe) | Padding according to max words after split.
