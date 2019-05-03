from isanlp.processor_sentence_splitter import ProcessorSentenceSplitter
from isanlp.ru.processor_tokenizer_ru import ProcessorTokenizerRu
from isanlp.ru.processor_mystem import ProcessorMystem
from isanlp import PipelineCommon
from isanlp.wrapper_multi_process_document import WrapperMultiProcessDocument


import msgpack
import os
import pandas as pd


class GeneratorSentences:
    def __init__(self, dir_path):
        self._dir_path = dir_path
        
    def __iter__(self):
        for fname in os.listdir(self._dir_path):
            file_path = os.path.join(self._dir_path, fname)
            
            with open(file_path, 'rb') as f:
                annots = msgpack.unpackb(f.read(), raw=False)
            
            if annots:
                for doc_annots in annots:
                    for sent in doc_annots:
                        yield sent
                        
                        
class GeneratorSentencesCompounds:
    def __init__(self, dir_path, compounds, strategy=1):
        self._dir_path = dir_path
        self._compounds = compounds
        self._strategy = strategy
        
    def __iter__(self):
        for fname in os.listdir(self._dir_path):
            file_path = os.path.join(self._dir_path, fname)
            
            with open(file_path, 'rb') as f:
                annots = msgpack.unpackb(f.read(), raw=False)
            
            if not annots:
                continue
            
            for doc_annots in annots:
                for sent in doc_annots:
                    additional_sent = []
                    
                    i = 0
                    while i < len(sent) - 1:
                        compound = '{}_{}'.format(sent[i], sent[i + 1])
                        if compound in self._compounds:
                            if self._strategy == 2:
                                additional_sent.append(sent[:i] + [compound] + sent[i + 2:])
                            elif self._strategy == 1:
                                sent = sent[:i] + [compound] + sent[i + 2:]
                                i += 1
                            else:
                                raise ValueError()
                        
                        i += 1

                    yield sent
                    for add_sent in additional_sent:
                        yield add_sent


def prepare_compounds(compounds_path):
    ppl = PipelineCommon([
        (ProcessorTokenizerRu(), ['text'], {0 : 'tokens'}),
        (ProcessorSentenceSplitter(), ['tokens'], {0 : 'sentences'}),
        (ProcessorMystem(), ['tokens', 'sentences'], {'lemma' : 'lemma'})
    ])
    
    df_compounds = pd.read_csv(compounds_path)

    compound_set = set()
    for i in df_compounds.index:
        compound = '{} {}'.format(df_compounds.loc[i, 'Часть 1'], df_compounds.loc[i, 'Часть 2'])
        lemmas = ppl(compound)['lemma'][0]
        compound_set.add('{}_{}'.format(lemmas[0], lemmas[1]))
    
    return compound_set


def train_dsm(compounds_path, data_path, save_path, model, epochs=5, strategy=1):
    compounds = prepare_compounds(compounds_path)
    generator = GeneratorSentencesCompounds(data_path, compounds, strategy)
    
    model.build_vocab(sentences=generator)
    model.train(sentences=generator, total_examples=model.corpus_count, epochs=epochs)
    model.save(save_path)
    
    return model
