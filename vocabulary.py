import jamo

LEADS = [chr(i) for i in range(0x1100, 0x1113)]
VOWELS = [chr(i) for i in range(0x1161, 0x1176)]
TAILS = [chr(i) for i in range(0x11A8, 0x11C3)]
HANGUL = LEADS + VOWELS + TAILS
ENGLISH = [chr(i) for i in range(0x0061, 0x007B)]
PUNCT = list('.?!\'(),-:; ')
PAD = '_'
EOS = '>'

VOCAB = HANGUL + ENGLISH + PUNCT + [PAD] + [EOS]

CHAR2IDX = {n: i for i, n in enumerate(VOCAB)}
IDX2CHAR = {i: n for i, n in enumerate(VOCAB)}


def str2token(text):
    text = fix_string(text)
    token = [CHAR2IDX[c] for c in text]
    return token


def token2str(token):
    return j2h([IDX2CHAR[c] for c in token])


# 분리된 자모 문자열을 원래 텍스트로 변환
# jamo 라이브러리에서 제공하지 않는 기능 :(
def j2h(text):
    i = 0
    result = []
    while i < len(text):
        if i + 2 < len(text) and text[i] in LEADS and text[i + 1] in VOWELS and text[i + 2] in TAILS:
            result.append(jamo.j2h(text[i], text[i + 1], text[i + 2]))
            i += 3
        elif i + 1 < len(text) and text[i] in LEADS and text[i + 1] in VOWELS:
            result.append(jamo.j2h(text[i], text[i + 1]))
            i += 2
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


def fix_string(text):
    text = text.lower()
    text = text.replace("\"", "\'")
    text = text.replace("‘", "\'")
    text = text.replace("’", "\'")
    text = text.replace("“", "\'")
    text = text.replace("”", "\'")
    text = text.replace("—", "-")
    text = text.replace(PAD, "")
    text = text.replace(EOS, "")

    text = jamo.h2j(text)

    text = "".join([(c if c in VOCAB else "") for c in text])  # 사전에 없는 문자 제거

    text = text.replace("()", "")  # NIKL 데이터셋 대응 '(한자)'

    return text


def add_eos(token):
    token.append(CHAR2IDX[EOS])
    return token


def add_pad(token, count):
    for _ in range(count):
        token.append(CHAR2IDX[PAD])
    return token


def get_vocab_size():
    return len(VOCAB)
