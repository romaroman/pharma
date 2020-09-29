import logging
from typing import Union, Tuple

import confuse
import utils


logger = logging.getLogger('config')


timestamp = utils.get_str_timestamp()
confuse = confuse.Configuration('_', 'simclr')
confuse.set_file('config.yaml')


confuse_s = confuse['SimCLR']

batch_size: int = confuse_s['BatchSize'].as_number()
epochs: int = confuse_s['Epochs'].as_number()

log_every_n_steps: int = confuse_s['LogEveryNSteps'].as_number()
eval_every_n_epochs: int = confuse_s['EvalEveryNEpochs'].as_number()

fine_tune_from: Union[None, int] = confuse_s['FineTuneFrom'].get()
weight_decay: float = float(confuse_s['WeightDecay'].get())
fp16_precision: bool = confuse_s['FP16Precision'].get()

model_out_dim: int = confuse_s['Model']['OutDim'].as_number()
model_base_model: str = confuse_s['Model']['BaseModel'].as_str()

dataset_s: int = confuse_s['Dataset']['S'].as_number()
dataset_input_shape: Tuple[int, int, int] = eval(confuse_s['Dataset']['InputShape'].get())
dataset_num_workers: int = confuse_s['Dataset']['NumWorkers'].as_number()
dataset_valid_size: float = confuse_s['Dataset']['ValidSize'].as_number()

loss_temperature: float = confuse_s['Loss']['Temperature'].as_number()
loss_use_cosine_similarity: bool = confuse_s['Loss']['UseCosineSimilarity'].get()
