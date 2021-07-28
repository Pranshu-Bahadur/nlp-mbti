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
- [ ] utils.py : df parser (Instead of removing links keep only site name.)
- [ ] This might be a multilabel classification problem. Need to investigate further.
- [ ] 50 posts per user, split by "|||" delimiter. (Exploding might be correct. However, maybe explode every 4 posts? ~avg=35; 140==35x4 > 128)
- [ ] I don't think links should be removed. They should be tokenized instead.

## Logic Modules:
- [ ] controller (model-based RL?)
- [ ] constructor (init func)
- [ ] action space, state space (search space of hyperparams, models)
- [ ] back tracking (worst case reset? maybe...)
- [ ] state tracking (s, a, r)
- [ ] optimization based on, validation acc
- [ ] policy based on prev states
- [ ] agent
- [ ] constructor - (splitter)
- [ ] state tracking/loading
- [ ] (Run agent for 5e?)
- [ ] preprocess/ agent env gen strat
- [ ] Tweet to token format scraping
- [ ] exploding
- [ ] tokenizer config
- [ ] Multi-label/Single label set up
- [ ] action/behavior (train, eval)
- [ ] Handle Imbalance
- [ ] subset clustering
- [ ] weighted sampling
- [ ] weighted loss
- [ ] Use of text generation model? ..
- [ ] main/run (same args stuff; web hosted gui?)
