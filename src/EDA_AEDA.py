from koeda import AEDA
from tqdm import tqdm


def AEDA_init(aeda_cfg):
    return AEDA(
        morpheme_analyzer=aeda_cfg.morpheme_analyzer,
        punc_ratio=aeda_cfg.punc_ratio,
        punctuations=aeda_cfg.punctuations,
    )


def AEDA_generator(sentence, aeda_cfg):
    aeda = AEDA_init(aeda_cfg)
    aeda_sentences = []
    if aeda_cfg.tqdm:
        sentence_loop = tqdm(sentence, desc="AEDA Process! ")
    else:
        sentence_loop = sentence
    for i in sentence_loop:
        result = aeda(i, repetition=aeda_cfg.generator.repetition)
        if aeda_cfg.generator.repetition == 1:
            aeda_sentences.append(result)
        else:
            aeda_sentences.extend(result)
    return aeda_sentences
