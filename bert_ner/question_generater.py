from predict import ner_prediction, pos_process
from transformers import  BertTokenizer, TFBertModel

from konlpy.tag import Mecab # 형태소 단위로 나누기
import kss # 문장단위로 나누기
from hanspell import spell_checker # 띄어쓰기 + 맞춤법 교정
import random

import os
import warnings
from silence_tensorflow import silence_tensorflow

silence_tensorflow() # 텐서플로우 경고 무시
warnings.filterwarnings('ignore') # 경고 무시
# os.environ["CUDA_VISIBLE_DEVICES"] = "3" # 3번 GPU 사용

mecab = Mecab()
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

in_list = ['년','월','일','간','시','》','두',',','어','차','한'] # 태깅되어 빈칸이 되면 안되는 단어
out_list = ['그러나','이에','이런','그럼에도','따라서','상황',] # 태깅 안되었지만 빠져야하는 단어 (제거되지 않은 불용어)

# 빈칸 문제
def question_emoty(sentences):
    generated_question = []
    for sentence in sentences:
        question,answer = [],[]
        for word in sentence:
            if word[1] != '-':
                if word[0] in in_list:
                    question.append(word[0])
                else:
                    question.append(str('('+'_'*len(word[0])+')'))
                    answer.append(word[0])
            else:
                if word[0] in out_list:
                    pass
                else:
                    question.append(word[0])
        if answer: # 빈칸이 생성된 문장은 출력
            generated_question.append([question,answer])
    return generated_question

# OX 문제
def question_OX(sentences):
    tag_dict = {}
    Use_tag = {}
    unUse_tag = ['NUM_B','NUM_I','DAT_I','DAT_B']
    stop_word = ['일','을','를','개']

    # 태그별로 단어를 리스트로 저장
    for sentence in sentences:
        for word in sentence:
            
            if word[1] != '-':
                if word[0] not in stop_word:
                    if word[1] not in tag_dict:
                        tag_dict[word[1]] = [word[0]]
                    else:
                        tag_dict[word[1]].append(word[0])

    # 문장에서 같은 태그의 단어가 3가지 이상인 태그 선별
    for tag in tag_dict:
        if tag not in unUse_tag:
            if len(tag_dict[tag]) >= 3:
                Use_tag[tag] = tag_dict[tag]
    # OX 비율은 randint 함수로 랜덤으로 비율 선정 (1:1 비율로 수정)
    ox_ratio = {}
    for tag in Use_tag:
        O_ratio = random.randint(0,len(Use_tag[tag])//2)
        ox_ratio[tag] = O_ratio # 해당 태그에서 문제에 대한 답이 O일 비율은 O:X = O_ratio : len(Use_tag[tag]) - O_ratio


    tagSort_sentence = {}
    input_quesiton = []

    for sentence in sentences:
            for word in sentence:
                # 태그별로 단어가 3개 이상인 태그의 문장 추출
                # 중복을 방지하기 위해 이미 뽑은 문장은 제외함
                if word[1] in Use_tag:
                    if word[1] not in tagSort_sentence:
                        tagSort_sentence[word[1]] = [sentence]
                        break
                    else:
                        tagSort_sentence[word[1]].append(sentence)
                        break

    generated_question = []
    # # OX 문제 생성
    for tag in tagSort_sentence:
        ox_rate = ox_ratio[tag] # OX 비율
        o_idx = [] # 정답이 O인 문제의 인덱스

        over_list = [] # 뽑을 수의 중복을 방지
        for _ in range(ox_rate):
            tmp_num = random.randint(0,ox_rate)
            while tmp_num in over_list:
                tmp_num = random.randint(0,ox_rate)
            over_list.append(tmp_num)
            o_idx.append(tmp_num)
        
        for i in range(len(tagSort_sentence[tag])):
            question,answer = [], []
            if i in o_idx: # 만약 정답이 O인 문제면 그냥 출제한다.
                tmp_qusetion = tagSort_sentence[tag][i]
                for word in tmp_qusetion:
                    question.append(word[0])
                answer.append('O')
            else: # 정답이 X로 출력되어야 하는 문제는 같은 태그 내의 다른 단어로 바꿔서 출제한다.
                tmp_qusetion = tagSort_sentence[tag][i]
                answer_word = ""
                for word in tmp_qusetion:
                    if word[0] not in stop_word:
                        if word[1] == tag: # 해당 단어가 바꿔야하는 태그라면
                            word_set= list(set(Use_tag[tag]))
                            word_set.pop(word_set.index(word[0]))
                            i = random.randint(0,len(word_set)-1)
                            change_word = word_set[i] # 사용할 태그 랜덤으로 선택
                            question.append(change_word)
                            answer_word = word[0]
                        else:
                            question.append(word[0])
                    else:
                        question.append(word[0])
                        
                answer.append('X')
                answer.append(str("오답인 단어: "+change_word))
                answer.append(str('정답인 단어: '+answer_word ))
                
                
            generated_question.append([question,answer])
    return generated_question

def start(context):
    ktext_list = kss.split_sentences(context)
    kss_sentence = ner_prediction(ktext_list,max_seq_len=88, tokenizer=tokenizer, lang='ko')

    kss_qa= question_emoty(kss_sentence)
    empty_questions = []
    empty_answers = []
    for i in range(len(kss_qa)):
        tmp = "".join(kss_qa[i][0])
        # print("Question" ,i + 1, ": " , spell_checker.check(tmp).checked)
        # print("Answer" ,i + 1, ": " , kss_qa[i][1])
        # print()
        empty_questions.append(spell_checker.check(tmp).checked)
        empty_answers.append(kss_qa[i][1])

    # print("="*30)
    # print("OX 문제")
    # print("="*30)
    try:
        kss_qa= question_OX(kss_sentence)
        ox_questions = []
        ox_answers = []
        for i in range(len(kss_qa)):
            tmp = "".join(kss_qa[i][0])
            # print("Question" ,i + 1, ": " , spell_checker.check(tmp).checked)
            # print("Answer" ,i + 1, ": " , kss_qa[i][1])
            # print()
            ox_questions.append(spell_checker.check(tmp).checked)
            ox_answers.append(kss_qa[i][1])
    except:
        ox_questions = ['생성된 문제가 없습니다.','(문장이 너무 짧거나 인식할 수 있는 개체명의 종류가 충분하지 않습니다.)']
        ox_answers = ['-','-']

    return empty_questions,empty_answers, ox_questions, ox_answers





if __name__ == "__main__":
    context = input('텍스트를 입력해주세요.')
    start(context)



