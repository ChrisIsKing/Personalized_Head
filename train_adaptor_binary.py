from datasets import load_dataset
from snip_dataset import get_snip_dataset 
from clinc_dataset import get_clinc_dataset, intent
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.trainer_utils import speed_metrics
import sys
import torch
# from transformers import RobertaTokenizer, RobertaConfig, RobertaModelWithHeads
from transformers import BertTokenizer, BertConfig, BertModelWithHeads, BertModel, BertForSequenceClassification
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction, TrainerCallback, Trainer
import numpy as np
# from torchsummary import summary
# reference: https://github.com/Adapter-Hub/adapter-transformers/blob/master/notebooks/01_Adapter_Training.ipynb
# python 3.8 cuda 11.2 install using: pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

adapters_to_freeze = [int(i) for i in sys.argv[1:]]

model_name = "bert-base-uncased"
# dataset_name = "snips_built_in_intents"
dataset_name = "clinc_oos"
# dataset_label_num = 150
tokenizer = BertTokenizer.from_pretrained(model_name)
if dataset_name == "clinc_oos":
  label = "intent"
  dataset_label_num = 150
  train, test, dev = get_clinc_dataset(tokenizer = tokenizer)
else:
  label = "label"
  dataset_label_num = 7
  train, test = get_snip_dataset(tokenizer = tokenizer)
# dataset = load_dataset(dataset_name)


def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")

# Encode the input data
# dataset = load_dataset(dataset_name, "plus")
# dataset = dataset.map(encode_batch, batched=True)
# dataset = dataset.rename_column(label, "labels")
# dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
# train = dataset["train"]
# test = dataset["test"]

# if dataset_name == "snips_built_in_intents":
#   dataset = dataset["train"].train_test_split(test_size=0.2)

# config = BertConfig.from_pretrained(
#     model_name,
# )
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    # config=config,
)

# model.add_adapter(dataset_name)
# model.train_adapter(dataset_name)
model.load_adapter("training_output/checkpoint-46500/clinc_oos")
model.set_active_adapters("clinc_oos")


# model.add_classification_head(
#     dataset_name,
#     num_labels=dataset_label_num,
#     layers = 2
#   )

# for name, param in model.named_parameters():
#   print(name)
#   if 'classifier' not in name: # classifier layer
#     param.requires_grad = False

print("after adaptor")


print("adapter to freeze", adapters_to_freeze)

model.set_active_adapters(dataset_name, skip_layers=adapters_to_freeze)

# for name, param in model.named_parameters():
#   if 'adapters' in name:
#     param.requires_grad = False

print("#################BEFORE TRAINIG##################")
# for name, param in model.named_parameters():
#   print(name, param.data)
#   if param.requires_grad:
#     print(name, param.size())


def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    if type(output) == torch.Tensor:
        print("norm:", output.data.norm())

# for module in model.modules():
#     if not isinstance(module, nn.Sequential):
#         module.register_forward_hook(printnorm)

  
training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=50,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=100,
    output_dir="./training_output",
    overwrite_output_dir=True,
    remove_unused_columns=False,
)


def compute_accuracy(p: EvalPrediction):
  preds = np.argmax(p.predictions, axis=1)
  return {"acc": (preds == p.label_ids).mean()}

class AdapterDropTrainerCallback(TrainerCallback):
  def on_step_begin(self, args, state, control, **kwargs):
    skip_layers = list(range(0,12))
    kwargs['model'].set_active_adapters(dataset_name, skip_layers=skip_layers)

  def on_evaluate(self, args, state, control, **kwargs):
    skip_layers = list(range(0,12))
    kwargs['model'].set_active_adapters(dataset_name, skip_layers=skip_layers)

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=test,
    compute_metrics=compute_accuracy,
)

# trainer.add_callback(AdapterDropTrainerCallback())

# trainer.add_callback(AdapterDropTrainerCallback())
# trainer.train()
print("#################AFTER TRAINIG##################")
# for name, param in model.named_parameters():
#     print(name, param.data)
metrics, output =  trainer.evaluate(eval_dataset=test)


preds = []
gold = []
count = 0

for i in range(0, len(test), 150):
  batch = output.predictions[i:i+150]
  pred = intent[batch[:,1].argmax()]
  preds.append(pred)
  gold_label = test[i]['name']
  gold.append(gold_label)
  if pred == gold_label:
    count+=1

print(preds)
print(gold)
print(count, len(preds))
print('Accuracy = {}'.format(count/len(preds)))


