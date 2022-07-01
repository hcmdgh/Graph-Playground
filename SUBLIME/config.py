from typing import Literal

dataset: Literal['cora'] = 'cora'

mode: Literal['structure_refinement', 'structure_inference'] = 'structure_refinement'

downstream_task: Literal['classification', 'clustering'] = 'classification'

learner_type: Literal['fgp'] = 'fgp'

lr = 0.01
weight_decay = 0.0
