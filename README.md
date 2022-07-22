# Extraction_5H1W

## Introduce
뉴스 본문 한줄 요약 후 육하원칙 추출

## :bust_in_silhouette: Participation Member

 [ 정영섭 ]  Professor   
 [ 김아경 ]  Student  : [ bzxz55@gmail.com ]   
 [ 이채윤 ]  Student  : [ dbs8438@gmail.com ]


 ## 📖 Introduction
[프로젝트 목적]  
뉴스 기사 본문을 세 줄 요약하고 육하원칙 추출

[프로젝트 필요성]  
핵심 문장으로 구성된 뉴스 요약문에 육하원칙이 추출되어 나온다면 요약문과 육하원칙을 보고 뉴스 기사의 핵심 내용을 한눈에 알아볼 수 있고 구체적으로 정보를 전달받을 수 있다.  
참고링크 : https://www.yna.co.kr/view/AKR20180205036951005?input=1195m

[데이터]  
크롤링 : https://news.daum.net/


 ## 🌐 Dependency Build Instructions

```
pip install konlpy
pip install scikit-learn
pip install 

scikit-learn==0.22.2.post1
konlpy==0.5.2
```

## 📝 How to use
slotminer directory 안에 run.py 파일과 Final.py 저장 후 아래 코드 실행
``` 
~/slotminer$ python3 run.py
```
뉴스 기사 제목과 본문을 순서대로 입력하면 who, when, where, what, how, why 순서대로 출력.

[주의사항]
뉴스 본문 입력 시, 문단 구분이 없어야 됨.

## 📄 Example
![image](https://user-images.githubusercontent.com/70522267/141249053-94cc2abd-eb39-4122-b1d3-bab15b5b58f7.png)


## 📄 Detail Information
[Git_lab](https://gitlab.com/veronica1/text-mining)
