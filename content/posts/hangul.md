---
title: '한국인만 알아볼 수 있는 리뷰 만들기'
date: 2018-08-05
description: 'Python을 이용한 한글 유니코드 분리'
category: 'dev'
---

최근에 페이스북 게시물들을 보다가, 재밌는 글을 하나 발견했다. 숙소 어플에서 한국인이 남긴 리뷰가 화제가 된 것이다. "한국인만 알아볼 수 있는 리뷰"라는 제목이었는데, 아래처럼 한글을 일부러 변형해서 쓴 모습이었다.
![한국인만 읽을 수 있는 리뷰](https://user-images.githubusercontent.com/39009836/43678986-021f8cea-9858-11e8-9a11-ab1941a87ae5.png)

아마도 한국말로 안 좋은 리뷰를 작성하면 호텔 주인이 번역기로 돌려서 확인하고 삭제할까봐 번역기를 아예 돌리지 못하게 이런 식으로 글을 쓴 것 같다. 한국인의 근성이란...

재밌어서 더 찾아보니, 실제로 외국 호텔 리뷰는 (악평을 할 경우)이런 식으로 작성한 경우가 꽤나 많았다. 하지만 저렇게 한땀한땀 변형해서 적는것도 귀찮은 일, 정상적으로 글을 작성하면 저런 식으로 변형하는 프로그램이 있으면 어떨까?라고 생각했고, 바로 실행에 옮겼다.
- - -
우선, 변형을 어떤 방식으로 할 지 생각해야한다.
- 초성을 된소리로 바꿀까?
`ex) 박진호 → 빡찐호`
- 모음을 비슷한 발음으로 변형할까?
`ex) 박진호 → 뱍쥔효`
- 아니면 둘 다?
`ex) 박진호 → 뺙쮠효`

이외에도 여러가지 방법이 있겠지만, 필자는 두 번째 방법을 선택했다. 첫 번째 방식을 하기엔 된소리가 없는 자음(ㄴ, ㅇ, ㅎ 등)이 너무 많았고, 그렇다고 세 번째 방식으로 하기엔 한국인도 읽기 힘들거라 생각했기 때문이다.

그럼 모음을 어떻게 바꿀까? 특별한 기준은 없고, 그냥 필자 기준에서 발음해 봤을 때 비슷하다고 느껴지는 것들로 표를 작성했다.
![모음 변형 전략](https://user-images.githubusercontent.com/39009836/43679125-a7f9960e-985a-11e8-8a4a-4bdce86f9311.png)


노란색으로 표시된 칸은 딱히 바꿀 모음이 떠오르지 않아서 그대로 둔 모음이다.

`ex) 박진호 -> 뱍즨효`
아무튼, 여차저차해서 글자를 어떻게 바꿀지 계획은 다 세웠으니, 코딩을 해보자.
- - -
![글자 변형 방법](https://user-images.githubusercontent.com/39009836/43679256-5924d3f6-985d-11e8-9eec-5fec49550adc.png)

대략적인 계획은 다음과 같다.
1. 글자를 초성, 중성, 종성으로 분리한다.
2. 중성을 다른 모음으로 변형한다.
3. 변형한 중성을 포함한 초성, 중성, 종성을 다시 합쳐 글자로 만든다.

이렇게 하려면 우선 **한글 글자를 분리하는 방법** 에 대해서 알아야하는데, 이를 위해 우선 한글 유니코드에 대해 알아보자.

한글 글자는 유니코드에 나열되어 있으며 '가'부터 '힣'까지 초성, 중성, 종성 순으로 총 11172개의 칸(`U+AC00`~`U+D7A3`)을 차지하고 있다.(`U+XXXX`안의 `XXXX`는 16진수 자연수로, 유니코드 상에서 해당 글자가 몇 번째에 나열되어 있는지를 말해준다. 즉 '가'는 AC00~(16)~(=44032~(10)~) 번째 글자라는 뜻이다.)
![한글 유니코드 표](https://user-images.githubusercontent.com/39009836/43679333-90e61aa6-985e-11e8-863c-f4f8e4eb4a1f.png)

**초성:**  "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ" (총 19개)

**중성:**  "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ" (총 21개)

**종성:**  "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ" (총 28개)

초성, 중성, 종성의 개수가 각각 19, 21, 28이므로 19\*21\*28 = 11172개의 칸을 차지하고, 우선순위가 초성 > 중성 > 종성이기 때문에 위 그림처럼 가, 각, 갂, ... 순서대로 나열된다.

이 사실을 이용하면 한글의 어떤 글자가 유니코드상에서 '가'로부터 몇 번째에 배치되어 있는지(이를 **순수한글코드** 라고 하자.) 다음 공식을 통해 구할 수 있다.

    (초성 * 21 * 28) + (중성 * 28) + 종성

    = ( (초성 * 21) + 중성 ) * 28 + 종성

예를 들어, '박'의 경우 초성 'ㅂ'은 7번째('ㄱ'을 0번째로 센다), 'ㅏ'는 0번째, 'ㄱ'은 1번째이므로 순수한글코드는 ((7\*21)+0)\*28+1=4117, '가'가 44032번째 유니코드이므로 '박'은 44032+4117=48129번째, 즉 `U+BC15`에 해당된다.

또한 역으로, 글자를 받아서 그 글자의 초성, 중성, 종성이 몇 번째에 해당하는지 다음 공식을 통해 구할 수 있다.

1. 종성

    순수한글코드 % 20 = 종성
2. 중성

    ( (순수한글코드 - 종성) / 28 ) % 21 = 중성
3. 초성

    ( ( ( 순수한글코드 - 종성) / 28) - 중성) ) / 21 = 초성

예를 들어 '박'의 경우, 순수한글코드가 4117이다. 이를 위 식에 대입하면 초성은 7번쨰, 중성은 0번째, 종성은 1번째에 해당한다는 결과를 얻을 수 있으며 이는 각각 'ㅂ', 'ㅏ', 'ㄱ'에 해당된다.

이제 모든 준비를 마쳤으니 본격적으로 프로그램을 짜보자.
- - -
####개발 환경: `Python 3`

`Python`에서 해당 글자가 유니코드상 몇 번째에 위치하는지는 기본 내장 함수인 `ord`를 통해 구할 수 있다.

```python
print('가: {}'.format(ord('가')))
print('개: {}'.format(ord('개')))
```

    가: 44032
    개: 44060

출력 형식은 당연히 `int`다.

```python
type(ord('가'))
```

    int

이를 다시 글자로 변환하고 싶다면, 역시 기본 내장 함수인 `chr`를 이용하면 된다. 즉, `ord`와 `chr`는 서로 역함수 관계이다.

```python
chr(44032)
```

    '가'

연습 삼아 위에서 했던 예시들을 코딩해보자. 우선 초성, 중성, 종성에 대한 정보를 리스트로 저장한다.

```python
#초성
iniL = [ "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ",
        "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ","ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ" ]
#중성
neuL = [ "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ",
         "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ" ]
#종성
finL = [ "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ",
         "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ",
         "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ","ㅍ", "ㅎ" ]
```

순수한글코드를 `x`라는 변수에 담아 초성, 중성, 종성을 각각 구하여 확인해본다.

```python
x = ord('박') - ord('가')
ini = x%28
neu = ((x-ini)//28)%21
fin = (((x-ini)//28)-neu)//21

print('{}, {}, {}'.format(ini, neu, fin))
print('{}, {}, {}'.format(iniL[ini], neuL[neu], finL[fin]))
```

    7, 0, 1
    ㅂ, ㅏ, ㄱ

자, 생각했던 것과 똑같은 결과가 나왔음을 확인할 수 있다.

코딩을 하다 보니 하나의 글자를 그 글자와 초성, 중성, 종성에 대한 정보를 담는 하나의 객체로 만들어서 관리하면 편리할 것이라는 생각이 들었다. 그래서 `Hangul`이라는 class를 생성했다.

전체적인 코드는 다음과 같다.

```python
class Hangul:
    #초성: ini 중성: neu 종성: fin
    element_query = {
        'ini' : [ "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ",
                "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ","ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ" ],

        'neu' : [ "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ",
                 "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ" ],

        'fin' : [ "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
                 "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ","ㅄ", "ㅅ", "ㅆ",
                 "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ","ㅍ", "ㅎ" ]
    }

    def __init__(self, letter_or_element):
        if type(letter_or_element) == str:
            self.letter = letter_or_element
            self.element = self.separate(letter_or_element)
        elif type(letter_or_element) == list:
            self.letter = self.combine(letter_or_element)
            self.element = letter_or_element
        else:
            self.letter = ""
            self.element = ["","",""]          

    def __str__(self):        
        return self.letter

    def separate(self, letter):
        x = ord(letter) - ord('가')
        fin_ord = x % 28
        neu_ord = ((x - fin_ord) // 28) % 21
        ini_ord = (((x - fin_ord) // 28)- neu_ord) // 21

        ini = self.element_query['ini'][ini_ord]
        neu = self.element_query['neu'][neu_ord]
        fin = self.element_query['fin'][fin_ord]
        return [ini, neu, fin]

    def combine(self, element):
        ini_ord = self.element_query['ini'].index(element[0])
        neu_ord = self.element_query['neu'].index(element[1])
        fin_ord = self.element_query['fin'].index(element[2])

        hangul_ord = (((ini_ord * 21) + neu_ord) * 28) + fin_ord

        return chr(hangul_ord + ord('가'))

    def encrypt(self):
        neus_encrypt = [ "ㅑ", "ㅒ", "ㅒ", "ㅖ", "ㅕ", "ㅖ", "ㅖ", "ㅒ", "ㅛ",
                 "ㅙ", "ㅞ", "ㅞ", "ㅛ", "ㅠ", "ㅞ", "ㅝ", "ㅟ", "ㅠ", "ㅢ", "ㅢ", "ㅣ" ]
        ret = self
        ret.element[1] = neus_encrypt[self.element_query['neu'].index(self.element[1])]
        ret.letter = self.combine(ret.element)

        return ret
```

코드가 생각보다 길어지므로 하나하나 차근차근 설명하려한다.
- - -
우선 클래스 변수 element_query에 초성, 중성, 종성에 대한 정보를 저장한다.

```python
element_query = {
    'ini' : [ "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ",
            "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ","ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ" ],

    'neu' : [ "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ",
             "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ" ],

    'fin' : [ "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
             "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ","ㅄ", "ㅅ", "ㅆ",
             "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ","ㅍ", "ㅎ" ]
}
```
- - -
다음은 클래스의 생성자에 해당하는 부분이다. 객체를 생성할 때 인자로 글자(`letter`)를 받거나 초성, 중성, 종성 조합(`element`)을 받는 두 가지 경우를 모두 허용하려고 했다. 그런데, `Python`은 생성자 오버로딩을 지원하지 않는다. 따라서 아래와 같이 `if ~ else`구문을 이용해서 구현하였다.

인스턴스는 두 변수 `letter`와 `element`를 가지며, `letter`는 글자 자체, 즉 '박'에 해당하며 `element`는 초성, 중성, 종성의 리스트, 즉 `['ㅂ', 'ㅏ','ㄱ']`에 해당한다.

```python
def __init__(self, letter_or_element):
    if type(letter_or_element) == str:
        self.letter = letter_or_element
        self.element = self.separate(letter_or_element)
    elif type(letter_or_element) == list:
        self.letter = self.combine(letter_or_element)
        self.element = letter_or_element
    else:
        self.letter = ""
        self.element = ["","",""]  
```
생성자를 테스트 해보면 다음과 같이 잘 작동하는 모습을 볼 수 있다.
```python
a = Hangul('돌')
b = Hangul(['ㄱ','ㅔ','ㅁ'])
print('{}{}'.format(a.letter,b.letter))
```
    돌겜

- - -

출력될 때 어떤 방식으로 자신의 내용물을 보여줄지 결정하는 `__str__` 함수이다. '가'를 담고 있는 객체면 '가'를 출력하는게 적합하므로 `letter`를 반환한다.

```python
def __str__(self):        
    return self.letter
```

`__str__`함수에 대해 잠깐 짚고 넘어가자. `__str__` 함수를 따로 선언하지 않았을 경우 다음 코드는 아래와 같은 출력을 가진다.

```python
han = Hangul('한')
print(han)
```

    <__main__.Hangul object at 0x0000024A78491E10>
하지만 `__str__` 함수를 위와 같이 선언해주면, 똑같은 코드에 대해 아래와 같이 출력된다.

    '한'

즉 `__str__`은 객체의 얼굴을 담당하는 함수이다.      
- - -
글자를 받으면 그것을 초성, 중성, 종성으로 분리하는 `separate`함수와 초성, 중성, 종성의 리스트를 받으면 그것을 글자로 합치는 `combine`함수이다. 로직은 위에서 한 예시와 같으므로 생략한다.

```python
def separate(self, letter):
    x = ord(letter) - ord('가')
    fin_ord = x % 28
    neu_ord = ((x - fin_ord) // 28) % 21
    ini_ord = (((x - fin_ord) // 28)- neu_ord) // 21

    ini = self.element_query['ini'][ini_ord]
    neu = self.element_query['neu'][neu_ord]
    fin = self.element_query['fin'][fin_ord]
    return [ini, neu, fin]

def combine(self, element):
    ini_ord = self.element_query['ini'].index(element[0])
    neu_ord = self.element_query['neu'].index(element[1])
    fin_ord = self.element_query['fin'].index(element[2])

    hangul_ord = (((ini_ord * 21) + neu_ord) * 28) + fin_ord

    return chr(hangul_ord + ord('가'))
```

한 가지 주의할 점은, `Python`은 `C`와는 다르게 정수 사이의 연산이어도 `/` 연산자가 몫이 아닌 실제로 나눈 실수 값, `float`을 반환한다. 따라서 몫을 반환하는 연산자인 `//`를 써야한다.
- - -
우리가 의도한 대로 글자를 변형하여 반환하는 함수 `encrypt`이다. 초성, 중성, 종성 리스트에서 중성을 설정한 값 `neus_encrypt`로 변환하고 그것에 해당하는 글자를 만들어 `Hangul` 객체를 반환한다.

```python
def encrypt(self):
    neus_encrypt = [ "ㅑ", "ㅒ", "ㅒ", "ㅖ", "ㅕ", "ㅖ", "ㅖ", "ㅒ", "ㅛ",
             "ㅙ", "ㅞ", "ㅞ", "ㅛ", "ㅠ", "ㅞ", "ㅝ", "ㅟ", "ㅠ", "ㅢ", "ㅢ", "ㅣ" ]
    ret = self
    ret.element[1] = neus_encrypt[self.element_query['neu'].index(self.element[1])]
    ret.letter = self.combine(ret.element)

    return ret
```
작성한 `encrypt` 함수가 잘 작동하는지 테스트해보자.
```python
for letter in '세종대왕':
    print(Hangul(letter).encrypt(), end = "")
```

    셰죵댸왱

잘 작동한다. 어찌된게 조금 약올리는(?) 느낌이 들지만 넘어가자.
- - -
이제 글자 하나를 변형하는 방법을 완성했으니, 글자 여러 개로 이루어진 텍스트를 변형할 차례이다. 이를 위해 새로운 함수 `encrypt_text`를 선언하였다. `encrypt_text` 함수는 문자열을 받아 변형된 문자열을 반환한다.

```python
def encrypt_text(text):
    encrypted = ""
    for letter in text:
        if ord('가') <= ord(letter) <= ord('힣'):
            encrypted += Hangul(letter).encrypt().letter
        else:
            encrypted += letter
    return encrypted
```

`Hangul` 객체는 한글 글자 `가`~`힣`만을 받는 것을 전제로 하고 있으므로 알파벳 같은 다른 문자가 들어가면 에러를 발생시킨다. 그래서 함수 내에 문자가 `가`~`힣`에 있을 때만 `encrypt`하고, 나머지 문자는 그대로 나오도록 하였다.

`encrypt_text`를 테스트해보자.

```python
text = "동해물과 백두산이 마르고 닳도록"
print(encrypt_text(text))
```

    둉햬뮬괘 뱩듀샨이 먀릐교 댫됴룍

잘 작동한다.

이 글의 목적이 호텔 리뷰 작성이었으므로 호텔 리뷰도 변형해본다.

```python
review = "이 호텔 시설이 너무 별로였어요. 서비스도 좋지 않았습니다."
print(encrypt_text(review))
```

    이 효톌 시셜이 녀뮤 볠료옜여요. 셔비싀됴 죻지 얂얐싑니댜.

정말 한국인만 알아볼 수 있는 리뷰를 만들어냈다! 정말 그럴까? 구글 번역기에 출력된 결과를 넣고 돌려보자.
![image](https://user-images.githubusercontent.com/39009836/43682257-59b41208-98aa-11e8-9f00-cead9549c105.png)

의도와 일치하는 결과를 보여주었다. 구글 번역기도 안 통하는, 한국인만 알아볼 수 있는 리뷰 작성 프로그램을 완성했다!

...그런데 아무리 봐도 약오르는 느낌을 지울 수가 없어서 아예 약올리는 용도로 사용해보았다.

```python
text = "야~ 따라하지 말라고~."
print(encrypt_text(text))
```
    얘~ 땨랴햐지 먈랴교~.

![얘~ 땨랴햐지 먈랴교~.](https://user-images.githubusercontent.com/39009836/43682282-383bd16e-98ab-11e8-92ba-4c52ba12049e.png)


## 출처

[1] <http://dream.ahboom.net/entry/한글-유니코드-자소-분리-방법>  

[2] <https://docs.python.org/ko/3/library/functions.html#chr>

[3] <https://www.unicode.org/charts/PDF/UAC00.pdf>
