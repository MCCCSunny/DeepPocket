from torch.distributions import Categorical,Normal
import torch

probs = torch.tensor([0.0541, 0.0397, 0.0335, 0.0359, 0.0315, 0.0327, 0.0336, 0.0331, 0.0318,
        0.0305, 0.0324, 0.0339, 0.0356, 0.0343, 0.0319, 0.0392, 0.0299, 0.0299,
        0.0362, 0.0338, 0.0322, 0.0330, 0.0404, 0.0309, 0.0355, 0.0312, 0.0331,
        0.0393, 0.0308])

cat = Categorical(probs = probs)
print(cat.sample())