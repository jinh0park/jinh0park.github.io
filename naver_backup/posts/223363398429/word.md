## [2359] 수강신청, 티켓팅 연습 사이트 만들기

https://jinh0park.github.io/2359/

[23:59 - 수강신청, 티켓팅 연습](https://jinh0park.github.io/2359/) : 23:59: 54 소수점 표시 Click! 00초에 버튼 클릭!

​

스톱워치를 구현하자마자... 간단하고 재밌는 웹 앱 아이디어가 떠올라서 만들어버렸다. 형법 선행해야하는데 ㅜㅜ 오늘부터 시간 정해놓고 하루에 한시간만 코드 짜야겠다...

​

소스코드는 한  파일 안에 작성했다. (App.js)

```
import "./App.css";
import React, { useState, useEffect } from "react";

function App() {
  const [currentTime, setCurrentTime] = useState(0);
  const [delta, setDelta] = useState(null);
  const [showms, setShowms] = useState(false);

  const formatTime = (t, showms) => {
    //eslint-disable-next-line
    const ss = (50 + (parseInt(t / 1000) % 10)).toLocaleString("en-US", {
      minimumIntegerDigits: 2,
      useGrouping: false,
    });
    const ms = (t % 1000).toLocaleString("en-US", {
      minimumIntegerDigits: 3,
      useGrouping: false,
    });
    if (showms) return `${ss}:${ms}`;
    return `${ss}`;
  };

  useEffect(() => {
    setInterval(() => {
      setCurrentTime(new Date().getTime());
    }, 10);
  }, []);

  const onClick = () => {
    const d = 10000 - (currentTime % 10000);
    setDelta((d > 5000 ? d - 10000 : d) / 1000);
  };

  const result = (d) => {
    if (d > 0) {
      return `${d}초 빨랐습니다...`;
    } else if (d === 0) {
      return "정확합니다!";
    } else {
      return `${-d}초 늦었습니다...`;
    }
  };

  return (
    <div
      className={`bg-${parseInt(parseInt(currentTime / 1000) % 10)} container`}
    >
      <div>
        <p style={{ fontSize: "50px" }}>
          23:59:{formatTime(currentTime, showms)}
        </p>
        <p>
          <input
            type="checkbox"
            defaultChecked={showms}
            onChange={(e) => {
              setShowms(e.target.checked);
            }}
          />
          소수점 표시
        </p>
        <p>
          <button style={{ width: "300px", height: "100px" }} onClick={onClick}>
            Click!
          </button>
        </p>
        <p>{delta !== null ? result(delta) : "00초에 버튼 클릭!"}</p>
      </div>
    </div>
  );
}

export default App;

```

​

00초에 가까워질때마다 배경 색이 초록색에서 빨간색으로 변하는 것은, css에서 className에 번호를 붙여 10개까지 그라데이션을 부여하였다.

https://coolors.co/gradient-palette/2b4584-4a9e48?number=7

[Create a Gradient palette - Coolors](https://coolors.co/gradient-palette/2b4584-4a9e48?number=7) : Create a gradient palette between two colors.

그라데이션 만드는 사이트는 위의 coolors.co 추천!!

```
div.bg-0 {
  background: #8BFF85;
}
div.bg-1 {
  background: #98ED80;
}
div.bg-2 {
  background: #A5DB7C;
}
div.bg-3 {
  background: #B2C877;
}
div.bg-4 {
  background: #BFB672;
}
div.bg-5 {
  background: #CBA46E;
}
div.bg-6 {
  background: #D89269;
}
div.bg-7 {
  background: #E57F64;
}
div.bg-8 {
  background: #F26D60;
}
div.bg-9 {
  background: #FF5B5B;
}

div.container {
  width: 100vw;
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
}

p {
  text-align: center;
}

body {
  font-family: "Nanum Gothic", sans-serif;
}
```

그나저나 Github repo 별로 gh-page를 만들 수 있는 걸 처음 알았다. 깃헙도 나날이 발전하는구나... 

 해시태그 : 