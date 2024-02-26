# Bag-of-Words-Meets-Bags-of-Popcorn

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=white)
![Scikit](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Numpy](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)




## Movie review <br/> 
Train and predict positive/negative <br/>


# Random Forest
![Random Forest](https://github.com/Roni81/Bag-of-Words-Meets-Bags-of-Popcorn/blob/main/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202024-02-27%20002300.png)



# Random Forest + Gradient Boosting
![Random Forest + Gradient Boosting](https://github.com/Roni81/Bag-of-Words-Meets-Bags-of-Popcorn/blob/main/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202024-02-27%20002328.png)



## Code
### Import Library
```python 
import pandas as pd
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
SEED = 33
```
### Read Data
```python
DATA = '.\data'   # Data directory
```
```python
train = pd.read_csv(os.path.join(DATA, 'train.tsv'), delimiter='\t')
test = pd.read_csv(os.path.join(DATA, 'test.tsv'), delimiter='\t')
unlabled_train = pd.read_csv(os.path.join(DATA, 'unlabeledTrain.tsv'), delimiter='\t', on_bad_lines='skip')
```
```python
print(train.shape)
train.head()
```
(25000, 3)
<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>sentiment</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5814_8</td>
      <td>1</td>
      <td>With all this stuff going down at the moment w...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2381_9</td>
      <td>1</td>
      <td>\The Classic War of the Worlds\" by Timothy Hi...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7759_3</td>
      <td>0</td>
      <td>The film starts with a manager (Nicholas Bell)...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3630_4</td>
      <td>0</td>
      <td>It must be assumed that those who praised this...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9495_8</td>
      <td>1</td>
      <td>Superbly trashy and wondrously unpretentious 8...</td>
    </tr>
  </tbody>
</table>
</div>

```python
print(test.shape)
test.head()
```

(25000, 2)
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12311_10</td>
      <td>Naturally in a film who's main themes are of m...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8348_2</td>
      <td>This movie is a disaster within a disaster fil...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5828_4</td>
      <td>All in all, this is a movie for kids. We saw i...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7186_2</td>
      <td>Afraid of the Dark left me with the impression...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12128_7</td>
      <td>A very accurate depiction of small time mob li...</td>
    </tr>
  </tbody>
</table>
</div>

```python
print(unlabled_train.shape)
unlabled_train.head()
```
(49998, 2)

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9999_0</td>
      <td>Watching Time Chasers, it obvious that it was ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45057_0</td>
      <td>I saw this film about 20 years ago and remembe...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15561_0</td>
      <td>Minor Spoilers&lt;br /&gt;&lt;br /&gt;In New York, Joan Ba...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7161_0</td>
      <td>I went to see this film with a great deal of e...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>43971_0</td>
      <td>Yes, I agree with everyone on this site this m...</td>
    </tr>
  </tbody>
</table>
</div>


### Import Library
```python
from bs4 import BeautifulSoup   # Beutiful Soup
from nltk.corpus import stopwords  # Stop Words
```

```python
eng_stopwords = stopwords.words('english')  # NLTK 라이브러리를 사용하여 영어 Stopwords를 변수 eng_stopwords에 담음
```
### Import Library Regex
```python
import re
```
```python
def preprocessing(sentence):
    soup = BeautifulSoup(sentence, 'html.parser')
    cleaned = re.sub('[^a-zA-Z]',' ', soup.text)    # 정규표현식으로 a~z A~Z를 제외한 특수문자들을 공백 대치
    cleaned = cleaned.lower()
    cleaned = [word for word in cleaned.split() if word not in eng_stopwords] # 모든 문장을 단어로 나누고 eng_stopwords와 비교

    return ' '.join(cleaned)  #eng_stopwords에 해당되지 않는 단어를 공백 ' '과 함께 join함
```
```python
preprocessing(sample)  #함수 적용 테스트
```
'stuff going moment mj started listening music watching odd documentary watched wiz watched moonwalker maybe want get certain insight guy thought really cool eighties maybe make mind whether guilty innocent moonwalker part biography part feature film remember going see cinema originally released subtle messages mj feeling towards press also obvious message drugs bad kay visually impressive course michael jackson unless remotely like mj anyway going hate find boring may call mj egotist consenting making movie mj fans would say made fans true really nice actual feature film bit finally starts minutes excluding smooth criminal sequence joe pesci convincing psychopathic powerful drug lord wants mj dead bad beyond mj overheard plans nah joe pesci character ranted wanted people know supplying drugs etc dunno maybe hates mj music lots cool things like mj turning car robot whole speed demon sequence also director must patience saint came filming kiddy bad sequence usually directors hate working one kid let alone whole bunch performing complex dance scene bottom line movie people like mj one level another think people stay away try give wholesome message ironically mj bestest buddy movie girl michael jackson truly one talented people ever grace planet guilty well attention gave subject hmmm well know people different behind closed doors know fact either extremely nice stupid guy one sickest liars hope latter'

```python
all_review = pd.concat([train['review'], unlabled_train['review'], test['review']])  #전체 리뷰를 합침
all_review
```
0        With all this stuff going down at the moment w...
1        \The Classic War of the Worlds\" by Timothy Hi...
2        The film starts with a manager (Nicholas Bell)...
3        It must be assumed that those who praised this...
4        Superbly trashy and wondrously unpretentious 8...
                               ...                        
24995    Sony Pictures Classics, I'm looking at you! So...
24996    I always felt that Ms. Merkerson had never got...
24997    I was so disappointed in this movie. I am very...
24998    From the opening sequence, filled with black a...
24999    This is a great horror film for people who don...
Name: review, Length: 99998, dtype: object

```python
all_review_clean = all_review.apply(preprocessing) # 전체 리뷰를 함수에 적용
all_review_clean
```
0        stuff going moment mj started listening music ...
1        classic war worlds timothy hines entertaining ...
2        film starts manager nicholas bell giving welco...
3        must assumed praised film greatest filmed oper...
4        superbly trashy wondrously unpretentious explo...
                               ...                        
24995    sony pictures classics looking sony got rights...
24996    always felt ms merkerson never gotten role fit...
24997    disappointed movie familiar case read mark fuh...
24998    opening sequence filled black white shots remi...
24999    great horror film people want vomit retching g...
Name: review, Length: 99998, dtype: object

## Count_Vectorizer
### 1st 일반적인 Count_Vectorizer

```python
from sklearn.feature_extraction.text import CountVectorizer
# sklearn.feature_extraction.text 모듈에서 CountVectorizer를 import
```
```python
cv = CountVectorizer(analyzer='word', max_features=5000)
# analyzer='word' 단어 단위로 텍스트를 분석하도록 지정 (다른 옵션 char, char_wb)
# max_features=5000 생성할 수 있는 최대 특징 수를 지정
```
```python
all_review_cv = cv.fit_transform(all_review_clean)
# 텍스트 데이터를 분석하여 특징 벡터로 변환
```
```python
all_review_cv.shape # 변환된 데이터의 사이즈 확인
```
```python
all_review_cv.toarray() # 변환된 데이터를 행열로 변
```
                                                                                                                                                     

### 2nd n-gram 적용 CountVectorizer
```python
ng_cv = CountVectorizer(analyzer='word', ngram_range=(1, 2),max_features=5000)
```
```python
all_review_ng_cv = ng_cv.fit_transform(all_review_clean)
```
```python
print("N-gram 행렬:")
print(all_review_ng_cv.toarray())
```
N-gram 행렬:
[[0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]]



### Split Train, Test 
```python
train_sentences = all_review_cv[:len(train)]
test_setences = all_review_cv[-len(test):]
```
```python
train_sentences.shape, test_setences.shape
```
((25000, 5000), (25000, 5000))
```python
train_labels = train['sentiment']
train_labels.shape
```
(25000,)


## Model 1
### RandomForestClassifier
```python
from sklearn.ensemble import RandomForestClassifier
```
```python
RFC = RandomForestClassifier(n_estimators=1000, max_depth=8, n_jobs=-1)
```
```python
RFC.fit(train_sentences, train_labels)
```
RandomForestClassifier<br/>
RandomForestClassifier(max_depth=8, n_estimators=1000, n_jobs=-1)
```python
prediction = RFC.predict(test_setences)
```
```python
prediction.shape
```
(25000,)
```python
prediction[:10]
```
array([1, 0, 1, 1, 1, 0, 0, 1, 0, 1], dtype=int64)

## Model 2
### VotingClassifier, RandomForestClassifier, GradientBoostingClassifier

```python
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
```python
rf_clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs= -1) # RandomForestClassifier 분류기 생성
gb_clf = GradientBoostingClassifier(n_estimators=15, random_state=42) # GradientBoostingClassifier 분류기 생성
```
```python
voting_clf = VotingClassifier(estimators=[('rfc', rf_clf),('gbc'gb_clf)], voting='soft')
# Voting Classifier 생성
```
```python
voting_clf.fit(train_sentences, train_labels)
```
```python
y_pred = voting_clf.predict(test_setences)
```
```python
prediction = voting_clf.predict(test_setences)
```
```python
prediction.shape #예측된 데이터의 행열 확인
```

### Submission
```python
submission =pd.read_csv(os.path.join(DATA, 'sampleSubmission.csv')) #Read Submission
```
```python
submission.head()
```
<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12311_10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8348_2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5828_4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7186_2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12128_7</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

```python
submission['sentiment'] = prediction
```
```python
submission['sentiment'].value_counts()
```
sentiment<br/>
1    13080 <br/>
0    11920 <br/>
Name: count, dtype: int64

```python
import datetime
```
```python
timestring = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
```
```python
filename = f'submission-{timestring}.csv'
filename
#파일명에 날짜시간 추가 및 테스트
```
'submission-2024-02-14-17-45-31.csv'

```python
submission.to_csv(os.path.join(DATA, filename),index = False)
#submission data 저장
```





