# MBTI Personality Classification based on users' twitter posts
------

## Dependencies:
```shell
pip install torch pandas transformers tokenizers datasets numpy
```
---

### Deliverables:
---
- [x] (unit tested) utils.py : dataframe parser using reduce & singledispatch
- [ ] (currently working on...) agent.py : NLPAgent(kwargs) | Just build constructor & tokenizer based on user kwargs

### Preprocessing strategy:
---
- [x] 50 posts per user, split by "|||" delimiter.
- [ ] BertweetTokenizer : normalization=True, fast=False (Which will normalize tweets (add URL tokens) and use then bpe) | Padding according to max words after split.



### Agent Description:
---

- [ ] 
- [ ] I'll need amd driver installed & ROCM(?)4.2, to run tests on local



# Notes:
---

- Doing this repo in tf2 (never used before)







