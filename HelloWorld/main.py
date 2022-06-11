from sklearn.feature_extraction.text import CountVectorizer
import random 
import nltk 
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def convert_NNS(word_list: list[str]) -> list[str]:
    """
    词形还原。将常用名词复数形式(NNS)还原为单数形式。
    
    例如：
    input: ['She', 'eats', 'three', 'apples', 'in', 'several', 'days', 'days.']
    output: ['She', 'eats', 'three', 'apple', 'in', 'several', 'day', 'days.']
    """
    tags = nltk.pos_tag(word_list)
    res = []
    
    for word, tag in tags:
        if tag == 'NNS':
            word = lemmatizer.lemmatize(word, 'n')
        
        res.append(word)

    return res 


def clear_text(text: str) -> str:
    text = text.replace('[SEP]', ' ')
    text = text.replace('[CLS]', ' ')
    text = text.replace('[UNK]', ' ')
    text = text.replace('-', ' ')
    text = text.replace('##', ' ')

    words = text.split()
    words = convert_NNS(words)

    return '_'.join(words)

# a = 'parkinson [unk] s disease'
# print(clear_text(a))

with open('/home/Dataset/DuChenguang/MAG/conf_papers_ne_author_zh.json', 'r') as fp:
    txt = fp.read()
    txt_lower = txt.lower()

    index = 0 

    while True:
        index = txt_lower.find('[unk]', index)

        print(index)
        
        
        
        print(txt[index-50:index+50])

        index += 1 

    exit()

a = [
    ' '.join([''.join(random.choices('ABCDE_FG', k=1)) for _ in range(5)])
    for _ in range(6)
]
# a.append('耿皓 哈哈 😊 parkinson_[unk]_s_disease')

# print(a)


vectorizer = CountVectorizer(token_pattern=r'[^ ]+')
doc_term_mat = vectorizer.fit_transform(a)

row = doc_term_mat.tocoo().row 
col = doc_term_mat.tocoo().col 

print(a)
print(vectorizer.vocabulary_)
print(row)
print(col)

id2word = { 
    id: word for word, id in vectorizer.vocabulary_.items() 
}

# for i, j in zip(row, col):
#     word = id2word[j]
    
#     print(i, word)