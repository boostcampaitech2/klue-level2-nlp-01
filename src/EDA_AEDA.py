from koeda import AEDA


def AEDA_init(aeda_cfg):
    return AEDA(
        morpheme_analyzer=aeda_cfg.morphme_analyzer,
        punc_ratio=aeda_cfg.punc_ratio,
        punctuations=aeda_cfg.punctuations,
    )


def AEDA_generator(aeda, sentence, aeda_generator_cfg):
    result = aeda(
        sentence, p=aeda_generator_cfg, repetition=aeda_generator_cfg.repetition
    )
    if aeda_generator_cfg.repetition == 1:
        return [result]
    return result
