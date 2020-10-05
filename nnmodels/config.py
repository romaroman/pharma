import confuse
import logging
from typing import Union, Tuple

from nnmodels.enums import Model

import utils


logger = logging.getLogger('config')


timestamp = utils.get_str_timestamp()
confuse = confuse.Configuration('_', 'simclr')
confuse.set_file('config.yaml')


confuse_nn = confuse['NN']
confuse_cc = confuse_nn['Common']

used_model: Model = Model[confuse_nn['UsedModel'].as_str()]

batch_size: int = confuse_cc['BatchSize'].as_number()
epochs: int = confuse_cc['Epochs'].as_number()

log_every_n_steps: int = confuse_cc['LogEveryNSteps'].as_number()
eval_every_n_epochs: int = confuse_cc['EvalEveryNEpochs'].as_number()

fine_tune_from: Union[None, int] = confuse_cc['FineTuneFrom'].get()
weight_decay: float = float(confuse_cc['WeightDecay'].get())
fp16_precision: bool = confuse_cc['FP16Precision'].get()

dataset_s: int = confuse_cc['Dataset']['S'].as_number()
dataset_input_shape: Tuple[int, int, int] = eval(confuse_cc['Dataset']['InputShape'].get())
dataset_num_workers: int = confuse_cc['Dataset']['NumWorkers'].as_number()
dataset_valid_size: float = confuse_cc['Dataset']['ValidSize'].as_number()

confuse_simclr = confuse_nn['SimCLR']

simclr_model_out_dim: int = confuse_simclr['OutDim'].as_number()
simclr_model_base_model: str = confuse_simclr['BaseModel'].as_str()
simclr_loss_temperature: float = confuse_simclr['Loss']['Temperature'].as_number()
simclr_loss_use_cosine_similarity: bool = confuse_simclr['Loss']['UseCosineSimilarity'].get()
