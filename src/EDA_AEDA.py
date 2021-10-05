import random

from koeda import AEDA
from tqdm import tqdm


SPACE_TOKEN = "\u241F"


def replace_space(text: str) -> str:
    return text.replace(" ", SPACE_TOKEN)


def revert_space(text: list) -> str:
    clean = " ".join("".join(text).replace(SPACE_TOKEN, " ").split()).strip()
    return clean


class myAEDA(AEDA):
    def _aeda(self, data: str, p: float) -> str:
        if p is None:
            p = self.ratio

        split_words = self.morpheme_analyzer.morphs(replace_space(data))
        words = self.morpheme_analyzer.morphs(data)

        new_words = []
        q = random.randint(1, int(p * len(words) + 1))
        qs_list = [
            index
            for index in range(len(split_words))
            if split_words[index] != SPACE_TOKEN
        ]
        qs = random.sample(qs_list, q)

        for j, word in enumerate(split_words):
            if j in qs:
                new_words.append(SPACE_TOKEN)
                new_words.append(
                    self.punctuations[random.randint(0, len(self.punctuations) - 1)]
                )
                new_words.append(SPACE_TOKEN)
                new_words.append(word)
            else:
                new_words.append(word)

        augmented_sentences = revert_space(new_words)

        return augmented_sentences


def AEDA_init(aeda_cfg):
    return myAEDA(
        morpheme_analyzer=aeda_cfg.morpheme_analyzer,
        punctuations=aeda_cfg.punctuations,
    )


def AEDA_generator(sentence, aeda_cfg):
    aeda = AEDA_init(aeda_cfg)
    aeda_sentences = []
    if aeda_cfg.tqdm:
        sentence_loop = tqdm(sentence, desc="AEDA Process! ")
    else:
        sentence_loop = sentence
    for sentence in sentence_loop:
        for _ in range(aeda_cfg.generator.repetition):
            result = aeda(
                sentence,
                p=random.uniform(
                    aeda_cfg.generator.punc_ratio.min, aeda_cfg.generator.punc_ratio.max
                ),
            )
            aeda_sentences.append(result)

    return aeda_sentences
