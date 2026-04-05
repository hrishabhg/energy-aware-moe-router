## Paper 1

## Paper 2 - Switch
- Number of selected experts = 1 as against > 1 in paper-1
- Capacity factor explicitly mentioned in addition to load balancing penality (as per claude, it was there in paper-1 also and GShard paper. Termilogy evolved)
- Capacity factor is a hard engineering constraint as memory needs to be allocated physically.
- What does skip FFN for a token mean? - A dropped token does not get feature refinement.
    - Why is this not catastrophic? 
        - A few reasons. First, the network is deep — missing one FFN out of, say, 12 or 24 layers is a partial degradation, not a total failure. 
        - Second, the attention layers in subsequent blocks can partially compensate — they can attend to other tokens that did get their FFN processing, effectively borrowing refined representations. 
        - Third, the load-balancing loss keeps the drop rate low enough (typically a few percent of tokens) that it's a statistical rarity for any given token to be dropped at multiple layers.
- Mixed Precision (Choosing b32 matrix for routing [single device] and casting to b16 once expert is selected) was able to manage the speed of training as equal to b16 based routing matrix.
- Smaller param initialisation was used to avoid high peaky variance to avoid specific expert assignment.
- Sparse models have a structural disadvantage at fine-tuning compared to dense models of equivalent quality. You got the pre-training efficiency gain from sparsity, but you pay for it during fine-tuning because your parameter-to-data ratio is unfavorable. 
- A favorable exchange rate: you trade cheap parameters (total model size) for expensive compute (training steps).
- A large dense model with equitable params as switch transforms performs takes longer to complete a step (2.5x). From training steps (constant FLOPS), perplexity reduction is same per step but time to take a step is 2.5x lesser than dense model due to parallism.
- A dense model trained using distillation outperforms dense model trained from scratch. It keeps fraction of gains achieved by sparse models for pre-training as well as fine-tuning.