# Questions

## What make CODA better than Dualprompt?

1. orthogonal loss
2. much larger prompt quantity
    - CODA has 800 prompts, while Dualprompt only has 106 prompts
    - more prompts means more learnable parameters

### Experiment Design

1. without orthogonal loss
    - proved to be useless
        - ablate orthogonal loss on A
        - ablate orthogonal loss on AK
        - ablate orthogonal loss on AKP
2. 100 x 2 x emb_d prompts
