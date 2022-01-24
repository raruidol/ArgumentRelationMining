from simpletransformers.classification import ClassificationModel
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, precision_score, recall_score

model = ClassificationModel('albert', 'trained_models/albertxxlv2-model-trained', cuda_device=0, args={'silent':True})


nb1 = '..'

while nb1 is not '':

    nb1 = input('First sentence: ')
    if nb1 == '':
        continue
    nb2 = input('Second sentence: ')

    predictions, raw_outputs = model.predict([[nb1, nb2]])
    #print(nb1, nb2)
    if predictions[0] == 0:
        print('Inference')
    elif predictions[0] == 1:
        print('Conflict')
    elif predictions[0] == 2:
        print('Rephrase')
    elif predictions[0] == 3:
        print('No Relation')


    print('--------------------------------------------------------')

'''
predictions, raw_outputs = model.predict([["I think we shouldn't ignore this problem.", "This problem can't be ignored."]])
print("I think we shouldn't ignore this problem. // This problem can't be ignored.")
if predictions[0] == 0:
    print('Inference')
elif predictions[0] == 1:
    print('Conflict')
elif predictions[0] == 2:
    print('Rephrase')
elif predictions[0] == 3:
    print('No Relation')
    
    
print('--------------------------------------------------------')

predictions, raw_outputs = model.predict([["I think we shouldn't ignore this problem.", "This problem is not relevant for the people."]])

print("I think we shouldn't ignore this problem. // This problem is not relevant for the people.")
if predictions[0] == 0:
    print('Inference')
elif predictions[0] == 1:
    print('Conflict')
elif predictions[0] == 2:
    print('Rephrase')
elif predictions[0] == 3:
    print('No Relation')
    
print('--------------------------------------------------------')

predictions, raw_outputs = model.predict([["I think we shouldn't ignore this problem.", "Solving this problem we will be able to make significant advances."]])

print("I think we shouldn't ignore this problem. // Solving this problem we will be able to make significant advances.")
if predictions[0] == 0:
    print('Inference')
elif predictions[0] == 1:
    print('Conflict')
elif predictions[0] == 2:
    print('Rephrase')
elif predictions[0] == 3:
    print('No Relation')
    
print('--------------------------------------------------------')

predictions, raw_outputs = model.predict([["I think we shouldn't ignore this problem.", "My car is white."]])

print("I think we shouldn't ignore this problem. // My car is white.")
if predictions[0] == 0:
    print('Inference')
elif predictions[0] == 1:
    print('Conflict')
elif predictions[0] == 2:
    print('Rephrase')
elif predictions[0] == 3:
    print('No Relation')



eval_df = pd.read_csv('datasets/test.tsv', sep='\t')
eval_df.columns = ['text_a', 'text_b', 'labels']
eval_target = eval_df['labels'].tolist()

result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)
print(result)
preds = []
for pred in model_outputs:
    preds.append(pred.tolist().index(max(pred.tolist())))
    
eval_df['prediction'] = preds
eval_df.to_csv('predictions/US2016.csv', index=False)

print('Test F1-macro score:',f1_score(eval_target, preds,average='macro'))
print('Test precision score:',precision_score(eval_target, preds,average='macro'))
print('Test recall score:',recall_score(eval_target, preds,average='macro'))

eval_df = pd.read_csv('datasets/MM2012_m_relations.tsv', sep='\t')
eval_df.columns = ['text_a', 'text_b', 'labels']
eval_target = eval_df['labels'].tolist()

result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)
print(result)
preds = []
for pred in model_outputs:
    preds.append(pred.tolist().index(max(pred.tolist())))
    
eval_df['prediction'] = preds
eval_df.to_csv('predictions/MM2012.csv', index=False)

print('MM2012 F1-macro score:',f1_score(eval_target, preds,average='macro'))
print('MM2012 precision score:',precision_score(eval_target, preds,average='macro'))
print('MM2012 recall score:',recall_score(eval_target, preds,average='macro'))

eval_df = pd.read_csv('datasets/Bank_m_relations.tsv', sep='\t')
eval_df.columns = ['text_a', 'text_b', 'labels']
eval_target = eval_df['labels'].tolist()

result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)
print(result)
preds = []
for pred in model_outputs:
    preds.append(pred.tolist().index(max(pred.tolist())))
    
eval_df['prediction'] = preds
eval_df.to_csv('predictions/Bank.csv', index=False)

print('Bank F1-macro score:',f1_score(eval_target, preds,average='macro'))
print('Bank precision score:',precision_score(eval_target, preds,average='macro'))
print('Bank recall score:',recall_score(eval_target, preds,average='macro'))

eval_df = pd.read_csv('datasets/Empire_m_relations.tsv', sep='\t')
eval_df.columns = ['text_a', 'text_b', 'labels']
eval_target = eval_df['labels'].tolist()

result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)
print(result)
preds = []
for pred in model_outputs:
    preds.append(pred.tolist().index(max(pred.tolist())))
    
eval_df['prediction'] = preds
eval_df.to_csv('predictions/Empire.csv', index=False)

print('Empire F1-macro score:',f1_score(eval_target, preds,average='macro'))
print('Empire precision score:',precision_score(eval_target, preds,average='macro'))
print('Empire recall score:',recall_score(eval_target, preds,average='macro'))


eval_df = pd.read_csv('datasets/Money_m_relations.tsv', sep='\t')
eval_df.columns = ['text_a', 'text_b', 'labels']
eval_target = eval_df['labels'].tolist()

result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)
print(result)
preds = []
for pred in model_outputs:
    preds.append(pred.tolist().index(max(pred.tolist())))

eval_df['prediction'] = preds
eval_df.to_csv('predictions/Money.csv', index=False)
    
print('Money F1-macro score:',f1_score(eval_target, preds,average='macro'))
print('Money precision score:',precision_score(eval_target, preds,average='macro'))
print('Money recall score:',recall_score(eval_target, preds,average='macro'))

eval_df = pd.read_csv('datasets/Problem_m_relations.tsv', sep='\t')
eval_df.columns = ['text_a', 'text_b', 'labels']
eval_target = eval_df['labels'].tolist()

result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)
print(result)
preds = []
for pred in model_outputs:
    preds.append(pred.tolist().index(max(pred.tolist())))
    
eval_df['prediction'] = preds
eval_df.to_csv('predictions/Problem.csv', index=False)

print('Problem F1-macro score:',f1_score(eval_target, preds,average='macro'))
print('Problem precision score:',precision_score(eval_target, preds,average='macro'))
print('Problem recall score:',recall_score(eval_target, preds,average='macro'))

eval_df = pd.read_csv('datasets/Welfare_m_relations.tsv', sep='\t')
eval_df.columns = ['text_a', 'text_b', 'labels']
eval_target = eval_df['labels'].tolist()

result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)
print(result)
preds = []
for pred in model_outputs:
    preds.append(pred.tolist().index(max(pred.tolist())))
    
eval_df['prediction'] = preds
eval_df.to_csv('predictions/Welfare.csv', index=False)

print('Welfare F1-macro score:',f1_score(eval_target, preds,average='macro'))
print('Welfare precision score:',precision_score(eval_target, preds,average='macro'))
print('Welfare recall score:',recall_score(eval_target, preds,average='macro'))

'''