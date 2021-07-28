# MBTI Personality Classification based on users' twitter posts
------

[Kaggle Dataset]: https://www.kaggle.com/datasnaek/mbti-type

## Repository structure:

## Design: (Tentative)
- [ ] test (for-each module, for-each function) [Unit testing]
- [ ] controller (model-based RL?)
      - [ ] constructor (init func)
      - [ ] action space, state space (search space of hyperparams, models)
      - [ ] back tracking (worst case reset? maybe...)
      - [ ] state tracking (s, a, r)
      - [ ] optimization based on, validation acc
      - [ ] policy based on prev states
      - [ ] (Run agent for 5e?)
      - [ ] preprocess/ agent env gen strat
           - [ ] Tweet to token format scraping
           - [ ] exploding
           - [ ] tokenizer config
           - [ ] Multi-label/Single label set up
- [ ] agent
      - [ ] constructor - (splitter)
      - [ ] state tracking/loading
      - [ ] action/behavior (train, eval)
            - [ ] Handle Imbalance
                 - [ ] subset clustering
                 - [ ] weighted sampling
                 - [ ] weighted loss
                 - [ ] Use of text generation model? ..
- [ ] main/run (same args stuff; web hosted gui?)
