import re
from konlpy.tag import Okt
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import linear_kernel
from rule_loader import rule_loader
from rule_process import rule_process
from konlpy.tag import Kkma

class Extraction_5W1H:
    
    def __init__(self, title = None, text = None):
        pass
    
    def extract(self, input_title, input_text):
        clean_text = self.cleansing(input_text)
        #print(clean_text)
        one_sentence = self.sentence_1(input_title, clean_text)
        #print(one_sentence)
        print()
        print('한문장 요약 : ', one_sentence[1])
        print()
        print('who : ', self.who(one_sentence[1]))
        print('when : ',self.when(one_sentence[1]))
        print('where : ',self.where(one_sentence[1]))
        print('what : ',self.what(one_sentence[1]))
        print('how : ',self.how(one_sentence[1]))
        print('why : ', self.why(one_sentence[1]))
        
    ######################## preprocessing ########################
    # cleansing
    def cleansing(self, text):
        t = text.split('.')
        result = []
        for i in t:
            t1 = i.replace(u'\xa0',' ') #\xa0제거
            t2 = re.sub('([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', '', string = t1) #email제거
            t3 = re.sub("\n","",t2) #\n제거
            t4 = re.sub('[^a-zA-Z가-힣0-9\.\% ]','',string = t3) #특수기호 제거
            t5 = t4.strip() #앞뒤문장 공백 제거
            t6 = " ".join(t5.split()) #문장 중간 중복된 공백 제거
            t7 = t6.replace(". ",".")
            t8 = re.split('(?<=[^0-9])[\.|\n]', t7) #.기준으로 문장 분리

            result.extend(t8)

        f_result = [r for r in result if len(r) > 1] # 비어있는 데이터 삭제

        return f_result
    
    # tokenizer
    def tokenizer(self, raw, pos=["Noun","Alpha","Verb","Number"], stopword=[], convertStrings=False):
        okt = Okt()
        return [
            word for word, tag in okt.pos(raw, norm=True, stem=True)
                if len(word) > 1 and tag in pos and word not in stopword
            ]

    # one sentence summarize
    def sentence_1(self, title, data):
        vectorize = HashingVectorizer(tokenizer = self.tokenizer, n_features=7)
        try:
            X = vectorize.fit_transform(data)
            srch_vector = vectorize.transform([title])
            cosine_similar = linear_kernel(srch_vector, X).flatten()
            #sim_rank_idx = cosine_similar.argsort()[::-1]

            return sorted(zip(cosine_similar, data), reverse=True)[0]
        
        except:
            
            return None
        
        
    ######################## for extracting 5W1H ########################
    # slot extraction    
    def slot_extraction(self, text):
        # rule 
        rl = rule_loader(logger=None)
        rl.load('./rule/when.rule') #when
        rl.load('./rule/where.rule') #where

        if not rl.generate_rules():
            exit()

        rp = rule_process(rules = rl.get_rules(), logger = None)
        rp.indexing()
        use_indexing = True

        result, variables, matched = rp.process(text, indexing = use_indexing)
        result = rp.merge_slot(result, text, rl.get_policy())

        return result    

    # extraction token
    def token(self, data):
        kkma = Kkma()
        try:
            return kkma.pos(data)
        except:
            return None

    # find Josa
    def josa(self, token):
        try:
            idx = []
            for i, t in enumerate(token):
                if t[1][0] == 'J':
                    idx.append([i, t])
            return idx
        except:
            None
    
    
    ######################## extraction 5W1H ########################
    # who
    def who(self, text): # one sentence
        words = text.split(" ")
        word = []
        result = []
        for w in words:
            if w[-1:] in ['은', '는', '이', '가']:
                word.append(w)
            
        result.append(word)
    
        return result
    
    # when
    def when(self, text): # one sentence
        slot = self.slot_extraction(text)
        when = []
        for _, s in enumerate(slot):
            if s['name'] == 'slot_timex3':
                when.append(s['text'])

        return when
    
    # where
    def where(self, text): # one sentence
        slot = self.slot_extraction(text)
        where = []
        for _, s in enumerate(slot):
            if s['name'] == 'slot_location':
                where.append(s['text'])

        if not where:
            words = text.split(" ")
            for w in words:
                if w[-2:] == '에서' and w[0] not in [str(i) for i in range(10)]:
                    where.append(w)

        return where

    #what
    def what(self, text): # one sentence
        text_token = self.token(text)
        text_josa = self.josa(text_token)
        result = []
        for i in range(len(text_josa)):
            pre = text_josa[i-1]
            now = text_josa[i]
            output = []
            if now[1][1] == 'JKO':
                for j in range(pre[0]+1, now[0]+1):
                    output.append(text_token[j][0])

            if output:
                result.append(''.join(output))

        return result
    
    # how
    def how(self, text): # clean_text
        sentence = [s for s in text if '위해 ' in s]
        result = []
        for j, s in enumerate(sentence):
            words = s.split(' ')
            target = [words.index(w) for w in words if '위해' in w]
            how = [words[w] for w in range(target[0]+1, len(words))]
            result.append(' '.join(how))
        return result

    # why
    def why(self, text): # one sentence
        sentence = [s for s in text if '위해 ' in s]
        result = []

        for j, s in enumerate(sentence):
            words = s.split(' ')
            target = [words.index(w) for w in words if '위해' in w]
            why = [words[w] for w in range(0, target[0]+1)]
            result.append(' '.join(why))

        return result
