from koeda import AEDA


def AEDA_init(aeda_cfg):
    return AEDA(
        morpheme_analyzer=aeda_cfg.morpheme_analyzer,
        punc_ratio=aeda_cfg.punc_ratio,
        punctuations=aeda_cfg.punctuations,
    )


def AEDA_generator(sentence, aeda_cfg):
    print(len(sentence))
    print(sentence[2])
    aeda = AEDA_init(aeda_cfg)
    result = aeda(
        sentence[:3], p=aeda_cfg.generator.p, repetition=aeda_cfg.generator.repetition
    )
    print(result)
    return result


aeda = AEDA(
    morpheme_analyzer="Okt", punc_ratio=0.3, punctuations=[".", ",", "!", "?", ";", ":"]
)
