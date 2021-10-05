from koeda import AEDA


def AEDA_init(aeda_cfg):
    return AEDA(
        morpheme_analyzer=aeda_cfg.morpheme_analyzer,
        punc_ratio=aeda_cfg.punc_ratio,
        punctuations=aeda_cfg.punctuations,
    )


def AEDA_generator(sentence, aeda_cfg):
    aeda = AEDA_init(aeda_cfg)
    print(sentence)
    re = []
    for sen in sentence:
        print("\n" + sen + "\n")
        result = aeda(
            sen, p=aeda_cfg.generator.p, repetition=aeda_cfg.generator.repetition
        )
        print(result)
        re.append(result)
    return result
