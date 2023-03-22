from bs4 import BeautifulSoup
import urllib.request as req 
import argparse # 특정 웹사이트로 접속하기 위해
import re
from urllib.parse import quote
import json
from tqdm import tqdm


def crawling_list(url):
    res = req.urlopen(url).read()
    soup = BeautifulSoup(res, 'html.parser') #분석 용이하게 파싱
    find_tag = soup.findAll("a",{"class":"depth1-title"}) # "div",{"class":"section_body"}
    korean_hist = dict()
    for i in range(len(find_tag)):
        txt = re.sub(r"[^가-힣a-zA-Z0-9]","",str(find_tag[i].get_text())) # 특수문자 제거
        korean_hist[txt] = {"words":[],"word_size":0,"error_page":[]}
    return korean_hist

def search_size(visit_url,word):
    url = visit_url + word + "/" + "전체" + "?p=" + str(1) + "&"
    encoded_url = quote(url, safe=':/?&=') # 한글 주소 인코딩
    res = req.urlopen(encoded_url).read()
    soup = BeautifulSoup(res, 'html.parser')#분석 용이하게 파싱
    find_tag = soup.findAll("div",{"class":"count-text"})
    cnt = 0
    for i in range(len(find_tag)):
        cnt =  re.sub(r"[^0-9]","",str(find_tag[i].get_text()))
    return int(cnt) // 20 + 1

def visit_site(visit_url,word):
    word_list = []
    error_page = []
    target_cnt = search_size(visit_url,word)
    idx = 1
    print(word,"사전",target_cnt,"페이지 탐색")
    for idx in tqdm(range(1,target_cnt+1)):
        url = visit_url + word + "/" + "전체" + "?p=" + str(idx) + "&"
        encoded_url = quote(url, safe=':/?&=') # 한글 주소 인코딩
        try:
            res = req.urlopen(encoded_url).read()
            soup = BeautifulSoup(res, 'html.parser')#분석 용이하게 파싱
            find_tag = soup.findAll("div",{"class":"title"}) # "div",{"class":"section_body"}
            tmp = []
            for i in range(len(find_tag)):
                txt = re.sub(r"[^가-힣a-zA-Z0-9]","",str(find_tag[i].get_text()))
                if txt != "의견주제": # 특정 태그가 파싱된경우
                    tmp.append(txt)
            if tmp:
                word_list.extend(tmp)
            else:
                break
        except:
            error_page.append(idx)
    return word_list,error_page

def dict2json(hist_dict,savepath):
    # json 파일로 저장
    with open(savepath + 'korean_hist.json', 'w',encoding="UTF-8") as f : 
        json.dump(hist_dict, f, indent=4)

def readjson(savepath):
    with open(savepath + 'korean_hist.json', 'r') as f:
        data = json.load(f)
    print(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default='https://encykorea.aks.ac.kr/')
    parser.add_argument("--visit_url", type=str, default='https://encykorea.aks.ac.kr/Article/List/Type/')
    parser.add_argument("--savepath", type=str, default='/home/suyeon/code/capstone2/data_crawl/')
    args = parser.parse_args()

    korean_hist = crawling_list(args.url)
    for hist in tqdm(korean_hist):
        word_list,error_page = visit_site(args.visit_url,hist)
        korean_hist[hist]["words"].extend(word_list)
        korean_hist[hist]["word_size"] = len(word_list)
        korean_hist[hist]["error_page"].extend(error_page)

    dict2json(korean_hist,args.savepath)

    print("done")

    readjson(args.savepath)