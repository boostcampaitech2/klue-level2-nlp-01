from konlpy.tag import Mecab

mecab = Mecab()


text = "klue re 대회 우리 모두 하루하루 배워가며 성장해나가요!"

print(mecab.pos(text))
