# MBTI Personality Classification based on users' twitter posts
------


## Repository structure:

## Design: (Tentative)
- [ ] test (for-each module, for-each function) [Unit testing]

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
