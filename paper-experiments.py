from simpletransformers.classification import ClassificationModel
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, precision_score, recall_score
import datetime

train_df = pd.read_csv('datasets/train.tsv', sep='\t')
train_df.columns = ['text_a', 'text_b', 'labels']

eval_df = pd.read_csv('datasets/test.tsv', sep='\t')
eval_df.columns = ['text_a', 'text_b', 'labels']
eval_target = eval_df['labels'].tolist()

bank_df = pd.read_csv('datasets/Bank_m_relations.tsv', sep='\t')
bank_df.columns = ['text_a', 'text_b', 'labels']
bank_target = bank_df['labels'].tolist()

empire_df = pd.read_csv('datasets/Empire_m_relations.tsv', sep='\t')
empire_df.columns = ['text_a', 'text_b', 'labels']
empire_target = empire_df['labels'].tolist()

mm2012_df = pd.read_csv('datasets/MM2012_m_relations.tsv', sep='\t')
mm2012_df.columns = ['text_a', 'text_b', 'labels']
mm2012_target = mm2012_df['labels'].tolist()

money_df = pd.read_csv('datasets/Money_m_relations.tsv', sep='\t')
money_df.columns = ['text_a', 'text_b', 'labels']
money_target = money_df['labels'].tolist()

problem_df = pd.read_csv('datasets/Problem_m_relations.tsv', sep='\t')
problem_df.columns = ['text_a', 'text_b', 'labels']
problem_target = problem_df['labels'].tolist()

welfare_df = pd.read_csv('datasets/Welfare_m_relations.tsv', sep='\t')
welfare_df.columns = ['text_a', 'text_b', 'labels']
welfare_target = welfare_df['labels'].tolist()

train_args={
  "output_dir": "outputs/",
  "cache_dir": "cache/",

  "fp16": True,
  "fp16_opt_level": "O1",
  "max_seq_length": 256,
  "train_batch_size": 32,
  "eval_batch_size": 32,
  "gradient_accumulation_steps": 1,
  "num_train_epochs": 50,
  "weight_decay": 0,
  "learning_rate": 4e-5,
  "adam_epsilon": 1e-8,
  "warmup_ratio": 0.06,
  "warmup_steps": 0,
  "max_grad_norm": 1.0,
  "do_lower_case": True,

  "logging_steps": 50,
  "evaluate_during_training": False,
  "evaluate_during_training_steps": 2000,
  "evaluate_during_training_verbose": False,
  "use_cached_eval_features": False,
  "save_eval_checkpoints": False,
  "save_steps": 2000,
  "no_cache": False,
  "save_model_every_epoch": True,
  "tensorboard_dir": None,

  "overwrite_output_dir": True,
  "reprocess_input_data": True,
  
  "n_gpu": 2,
  "silent": False,
  "use_multiprocessing": True,

  "wandb_project": None,
  "wandb_kwargs": {},

  "use_early_stopping": True,
  "early_stopping_patience": 3,
  "early_stopping_delta": 0,

}

# Create a ClassificationModel
model = ClassificationModel('roberta', 'roberta-base', num_labels=4, use_cuda=True, cuda_device=0, args=train_args)
#print(train_df.head())

print(datetime.datetime.now())
# Train the model
model.train_model(train_df, eval_df=eval_df)
print(datetime.datetime.now())
# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)
print(result)
preds = []
for pred in model_outputs:
    preds.append(pred.tolist().index(max(pred.tolist())))

print('Test F1-macro score:',f1_score(eval_target, preds,average='macro'))
print('Test precision score:',precision_score(eval_target, preds,average='macro'))
print('Test recall score:',recall_score(eval_target, preds,average='macro'))

result, model_outputs, wrong_predictions = model.eval_model(mm2012_df, acc=sklearn.metrics.accuracy_score)
print(result)
preds = []
for pred in model_outputs:
    preds.append(pred.tolist().index(max(pred.tolist())))

print('MM2012 F1-macro score:',f1_score(mm2012_target, preds,average='macro'))
print('MM2012 precision score:',precision_score(mm2012_target, preds,average='macro'))
print('MM2012 recall score:',recall_score(mm2012_target, preds,average='macro')

result, model_outputs, wrong_predictions = model.eval_model(bank_df, acc=sklearn.metrics.accuracy_score)      
print(result)
preds = []
for pred in model_outputs:
    preds.append(pred.tolist().index(max(pred.tolist())))

print('Bank F1-macro score:',f1_score(bank_target, preds,average='macro'))
print('Bank precision score:',precision_score(bank_target, preds,average='macro'))
print('Bank recall score:',recall_score(bank_target, preds,average='macro'))


result, model_outputs, wrong_predictions = model.eval_model(empire_df, acc=sklearn.metrics.accuracy_score)
print(result)
preds = []
for pred in model_outputs:
    preds.append(pred.tolist().index(max(pred.tolist())))

print('Empire F1-macro score:',f1_score(empire_target, preds,average='macro'))
print('Empire precision score:',precision_score(empire_target, preds,average='macro'))
print('Empire recall score:',recall_score(empire_target, preds,average='macro'))


result, model_outputs, wrong_predictions = model.eval_model(money_df, acc=sklearn.metrics.accuracy_score)
print(result)
preds = []
for pred in model_outputs:
    preds.append(pred.tolist().index(max(pred.tolist())))

print('Money F1-macro score:',f1_score(money_target, preds,average='macro'))
print('Money precision score:',precision_score(money_target, preds,average='macro'))
print('Money recall score:',recall_score(money_target, preds,average='macro'))

result, model_outputs, wrong_predictions = model.eval_model(problem_df, acc=sklearn.metrics.accuracy_score)
print(result)
preds = []
for pred in model_outputs:
    preds.append(pred.tolist().index(max(pred.tolist())))

print('Problem F1-macro score:',f1_score(problem_target, preds,average='macro'))
print('Problem precision score:',precision_score(problem_target, preds,average='macro'))
print('Problem recall score:',recall_score(problem_target, preds,average='macro'))

result, model_outputs, wrong_predictions = model.eval_model(welfare_df, acc=sklearn.metrics.accuracy_score)
print(result)
preds = []
for pred in model_outputs:
    preds.append(pred.tolist().index(max(pred.tolist())))

print('Welfare F1-macro score:',f1_score(welfare_target, preds,average='macro'))
print('Welfare precision score:',precision_score(welfare_target, preds,average='macro'))
print('Welfare recall score:',recall_score(welfare_target, preds,average='macro'))

#predictions, raw_outputs = model.predict([["that position was non-judgmentalism", "the State has taken a moral position"]])
#print(predictions)
#print(raw_outputs)
