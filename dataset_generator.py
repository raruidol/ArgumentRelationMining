import pandas as pd
import os
import json
import numpy as np

#corpus = 'araucaria_662'
#corpus = 'IAC_1808'
#corpus = 'ArgMine'
#corpus = 'Microtext'


#corpus = 'US2016'
#corpus = 'Bank'
#corpus = 'Empire'
#corpus = 'MM2012'
#corpus = 'Money'
#corpus = 'Problem'
corpus = 'Welfare'

basePath = os.path.dirname(os.path.abspath(__file__))
data_path = basePath+'/datasets/'+corpus

# 'general' for a general dataset (proposition, label) [L, TA, YA, I, RA, CA, MA]
# 'relation' for a relational dataset (prop1, prop2, label) [RA, CA, MA]
# 'inference' for inference dataset (prop1, prop2, label) [AFExample, AFExpertOpinion, etc., Default Inference]
# 'conflict' for conflict dataset (prop1, prop2, label) [Logical contradiction, Pragmatic contrairness, etc., Default Conflict]
# 'rephrase' for rephrase dataset (prop1, prop2, label) [Specialisation, Generalisation, Instantiation, Default Rephrase]

TYPE = 'relation'

if TYPE == 'general':
    dataset = pd.DataFrame(columns=['cod', 'text', 'type'])

    for file in os.listdir(data_path):
        if file.endswith('.json'):
            print(file)
            with open(data_path+"/"+file) as json_file:
                arg_map = json.load(json_file)
                for node in arg_map['nodes']:
                    dataset = dataset.append({'cod': node['nodeID'], 'text': node['text'], 'type': node['type']}, ignore_index=True)

    dataset.describe()
    dataset.to_csv(basePath+'/datasets/'+corpus+'_general.csv', index=False)

elif TYPE == 'relation':

    dataset = pd.DataFrame(columns=['t1', 't2', 'type'])

    # Extracting tuples of sentences and their relation into dataframe from all the available argument maps

    for file in os.listdir(data_path):
        if file.endswith('.json'):
            print(file)
            with open(data_path+"/"+file) as json_file:
                arg_map = json.load(json_file)
                for node in arg_map['nodes']:
                    t1 = None
                    t2 = None
                    c1 = None
                    c2 = None
                    link = None
                    type = None
                    if node['type'] == 'CA' or node['type'] == 'RA' or node['type'] == 'MA':

                        # Link node detected of type CA, RA or MA
                        link = node['nodeID']
                        type = node['type']

                        # Finding edges from/towards the link node
                        for edge in arg_map['edges']:
                            ck1 = False
                            ck2 = False

                            if link == edge['toID']:
                                c1 = edge['fromID']
                                ck1 = True
                                #print('c1', c1)

                            elif link == edge['fromID']:
                                c2 = edge['toID']
                                ck2 = True
                                #print('c2', c2)

                            # Retrieving the text from the nodes
                            if (t1 == None and ck1 == True) or (t2 == None and ck2 == True):
                                for node2 in arg_map['nodes']:
                                    if node2['nodeID'] == c1 and node2['type'] == 'I':
                                        t1 = node2['text']
                                        #print('t1', t1)

                                    elif node2['nodeID'] == c2 and node2['type'] == 'I':
                                        t2 = node2['text']
                                        #print('t2', t2)

                        if t1 != None and t2 != None:
                            #print(c1, t1, c2, t2, type)
                            dataset = dataset.append({'t1': t1, 't2': t2, 'type': type}, ignore_index=True)

    # Generating NO samples

    # Dataframe conversion to list structure
    text1 = dataset['t1'].tolist()
    text2 = dataset['t2'].tolist()
    text_tuples = list(zip(text1, text2))
    target = dataset['type'].tolist()
    full_set = []
    for i in range(len(target)):
        full_set.append([text_tuples[i], target[i]])

    n_ra = 0
    n_ca = 0
    n_ma = 0
    sent_list = []
    rel_list = []

    # Counting number of items for each label, saving tuples of each type

    for element in full_set:

        rel_list.append(element[0])

        if element[0][0] not in sent_list:
            sent_list.append(element[0][0])
        if element[0][1] not in sent_list:
            sent_list.append(element[0][1])

        if element[1] == 'RA':
            n_ra += 1

        elif element[1] == 'CA':
            n_ca += 1

        elif element[1] == 'MA':
            n_ma += 1

    # NO samples 65%
    n_no = int(((n_ra+n_ca+n_ma)/0.35)*0.65)+1
    print(n_no)

    n = 0
    used_pointers = []
    while n < n_no:
        i = np.random.randint(len(sent_list))
        j = np.random.randint(len(sent_list))

        no_tup = (sent_list[i],sent_list[j])
        if (no_tup not in rel_list) and ((i,j) not in used_pointers) and i != j:
            used_pointers.append((i, j))
            n += 1
            dataset = dataset.append({'t1': no_tup[0], 't2': no_tup[1], 'type': 'NO'}, ignore_index=True)

    dataset = dataset.dropna()
    print(dataset['type'].value_counts())

    dataset.to_csv(basePath+'/datasets/'+corpus+'_relations.csv', index=False)

elif TYPE == 'inference':
    dataset = pd.DataFrame(columns=['c1', 't1', 'c2', 't2', 'type'])

    for file in os.listdir(data_path):
        if file.endswith('.json'):
            print(file)
            with open(data_path+"/"+file) as json_file:
                arg_map = json.load(json_file)
                for node in arg_map['nodes']:
                    t1 = None
                    t2 = None
                    c1 = None
                    c2 = None
                    link = None
                    type = None
                    if node['type'] == 'RA':
                        link = node['nodeID']
                        type = node['text']
                        for edge in arg_map['edges']:
                            if link == edge['toID']:
                                c1 = edge['fromID']
                            elif link == edge['fromID']:
                                c2 = edge['toID']
                            for node2 in arg_map['nodes']:
                                if node2['nodeID'] == c1 and node2['type'] == 'I':
                                    t1 = node2['text']
                                elif node2['nodeID'] == c2 and node2['type'] == 'I':
                                    t2 = node2['text']
                        #print(c1, t1, c2, t2, type)
                        dataset = dataset.append({'c1': c1, 't1': t1, 'c2': c2, 't2': t2, 'type': type}, ignore_index=True)

    dataset.describe()
    dataset.to_csv(basePath+'/datasets/'+corpus+'_inferences.csv', index=False)

elif TYPE == 'conflict':
    dataset = pd.DataFrame(columns=['c1', 't1', 'c2', 't2', 'type'])

    for file in os.listdir(data_path):
        if file.endswith('.json'):
            print(file)
            with open(data_path+"/"+file) as json_file:
                arg_map = json.load(json_file)
                for node in arg_map['nodes']:
                    t1 = None
                    t2 = None
                    c1 = None
                    c2 = None
                    link = None
                    type = None
                    if node['type'] == 'CA':
                        link = node['nodeID']
                        type = node['text']
                        for edge in arg_map['edges']:
                            if link == edge['toID']:
                                c1 = edge['fromID']
                            elif link == edge['fromID']:
                                c2 = edge['toID']
                            for node2 in arg_map['nodes']:
                                if node2['nodeID'] == c1 and node2['type'] == 'I':
                                    t1 = node2['text']
                                elif node2['nodeID'] == c2 and node2['type'] == 'I':
                                    t2 = node2['text']
                        #print(c1, t1, c2, t2, type)
                        dataset = dataset.append({'c1': c1, 't1': t1, 'c2': c2, 't2': t2, 'type': type}, ignore_index=True)

    dataset.describe()
    dataset.to_csv(basePath+'/datasets/'+corpus+'_conflicts.csv', index=False)

elif TYPE == 'rephrase':
    dataset = pd.DataFrame(columns=['c1', 't1', 'c2', 't2', 'type'])

    for file in os.listdir(data_path):
        if file.endswith('.json'):
            print(file)
            with open(data_path+"/"+file) as json_file:
                arg_map = json.load(json_file)
                for node in arg_map['nodes']:
                    t1 = None
                    t2 = None
                    c1 = None
                    c2 = None
                    link = None
                    type = None
                    if node['type'] == 'MA':
                        link = node['nodeID']
                        type = node['text']
                        for edge in arg_map['edges']:
                            if link == edge['toID']:
                                c1 = edge['fromID']
                            elif link == edge['fromID']:
                                c2 = edge['toID']
                            for node2 in arg_map['nodes']:
                                if node2['nodeID'] == c1 and node2['type'] == 'I':
                                    t1 = node2['text']
                                elif node2['nodeID'] == c2 and node2['type'] == 'I':
                                    t2 = node2['text']
                        #print(c1, t1, c2, t2, type)
                        dataset = dataset.append({'c1': c1, 't1': t1, 'c2': c2, 't2': t2, 'type': type}, ignore_index=True)

    dataset.describe()
    dataset.to_csv(basePath+'/datasets/'+corpus+'_rephrases.csv', index=False)
