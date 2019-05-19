import numpy as np

from isanlp.processor_sentence_splitter import ProcessorSentenceSplitter
from isanlp.ru.processor_tokenizer_ru import ProcessorTokenizerRu
from isanlp.ru.processor_mystem import ProcessorMystem
from isanlp import PipelineCommon

from sklearn.preprocessing import normalize


def acquiring(comp, model, true_label, model_words=None, skip_invalid_labels=True):
    ppl = PipelineCommon([
        (ProcessorTokenizerRu(), ['text'], {0 : 'tokens'}),
        (ProcessorSentenceSplitter(), ['tokens'], {0 : 'sentences'}),
        (ProcessorMystem(), ['tokens', 'sentences'], {'lemma' : 'lemma'})
    ])
    
    v_w1 = []
    v_w2 = []
    v_comp = []
    true_class = []
    
    if model_words is None:
        model_words = model
    
    indexes = []
    for i in comp.index:
        label = comp.loc[i, true_label]
        if skip_invalid_labels and label not in {0., 1.}:
            continue
            
        anns = ppl('{} {}'.format(comp.loc[i, 'Часть 1'], comp.loc[i, 'Часть 2']))['lemma'][0]
        
        try:
            #print('{}_{}'.format(anns[0], anns[1]))
            vec_w1 = model_words[anns[0]]
            vec_w2 = model_words[anns[1]]
            vec_comp = model['{}_{}'.format(anns[0], anns[1])]
            indexes.append(i)
        except KeyError:
            continue
        
        v_w1.append(vec_w1)
        v_w2.append(vec_w2)
        v_comp.append(vec_comp)
        true_class.append(label)
    
    print('Number of examples: ', len(v_w1))
    
    return np.array(v_w1), np.array(v_w2), np.array(v_comp), np.array(true_class), comp.loc[indexes]


def average_normalized(p1, p2):
    return normalize(p1) + normalize(p2)


def average_standard(p1, p2):
    return 0.5 * (p1 + p2)


def apply_distance(part1_vecs, part2_vecs, comp_vecs, f_distance, f_average):
    part_mean = f_average(part1_vecs, part2_vecs)
    distances = np.array([f_distance(part_mean[i], comp_vecs[i]) for i in range(part1_vecs.shape[0])])
    return -distances
