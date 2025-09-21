---
title: 'React로 정확한 스톱워치 만들기'
date: 2024-02-23
description: '오차 누적 없는 정확한 시간 계산 방법'
category: 'dev'
---

답안지 작성 사이트를 만들던 중 스톱워치 기능이 필요하여 검색해보았다.

구글에 "react stopwatch"를 치면 블로그 글이 여러개 나오는데, 그대로 따라해 작동 시켜보니 치명적인 문제점이 있었다. 바로 실제 시간과 일치하지 않는다는 것!!

웹 상의 많은 코드들이 setInterval을 이용해 1초마다 시간 값을 +1 해주는 방식으로 스톱워치를 구동하고 있었는데, setInterval의 interval을 1초(1000밀리초)로 설정해도 항상 1000밀리초마다 루프가 돌아간다는 보장이 없기 때문에 해당 문제가 발생한다.

```javascript
const func = () => {
    localStorage.setItem("1", "1");
};
const time = new Date().getTime();
const interval = setInterval(() => {
    func();
    console.log(new Date().getTime() - time);
}, 1000);
```

가령 위 코드를 크롬 콘솔창에 넣고 실행해보면, 1초(1000밀리초)마다 "현재 시각 - 처음 시각"이 밀리초 단위로 출력되어야 할 것이다. 만약 setInterval이 정확히 1000밀리초 단위로 반복된다면, 콘솔창에는 1000, 2000, 3000, ... 등 1000의 배수가 찍힐 것이다.

```javascript
import React, { useState, useEffect, useRef } from "react";

function Stopwatch() {
    const [initialTime, setInitialTime] = useState(0);
    const [currentTime, setCurrentTime] = useState(0);
    const [savedTime, setSavedTime] = useState(0);

    const [isRunning, setIsRunning] = useState(false);
    const interval = useRef();

    const formatTimeDelta = (timedelta) => {
        //eslint-disable-next-line
        const [ms, ss, mm, hh] = [
            parseInt(timedelta) % 100,
            parseInt(timedelta / 1000) % 60,
            parseInt(timedelta / (60 * 1000)) % 60,
            parseInt(timedelta / (60 * 60 * 1000)),
        ].map((x) =>
            x.toLocaleString("en-US", {
                minimumIntegerDigits: 2,
                useGrouping: false,
            })
        );
        return `${hh}:${mm}:${ss}:${ms}`;
    };

    useEffect(() => {
        const time = new Date().getTime();
        setCurrentTime(time);
        setInitialTime(time);
    }, []);

    useEffect(() => {
        if (isRunning) {
            const time = new Date().getTime();
            setInitialTime(time);
            setCurrentTime(time);
            interval.current = setInterval(() => {
                setCurrentTime(new Date().getTime());
            }, 10);
        } else {
            clearInterval(interval.current);
            setSavedTime((t) => t + currentTime - initialTime);
            const time = new Date().getTime();
            setCurrentTime(time);
            setInitialTime(time);
        }
        // currentTime, initialTime은 !isRunning일 때 렌더링 되지 않으므로, deps에 넣기 불필요
        // eslint-disable-next-line
    }, [isRunning]);

    const onRun = () => {
        setIsRunning((state) => !state);
    };

    const onStop = () => {
        const time = new Date().getTime();
        setIsRunning(false);
        setCurrentTime(time);
        setInitialTime(time);
        setSavedTime(0);
    };

    return (
        <div>
            <p>{formatTimeDelta(currentTime - initialTime + savedTime)}</p>
            <p>
                <button onClick={onRun}>{isRunning ? "PAUSE" : "PLAY"}</button>
                <button onClick={onStop}>STOP</button>
            </p>
        </div>
    );
}

export default Stopwatch;
```

대부분은 직관적으로 이해하기 어렵지 않은 코드이다. 다만 currentTime, initialTime 외 savedTime을 도입했는데, 이는 Pause 기능을 구현하기 위함이다.

currentTime과 initialTime만 state로 관리를 하고 그 차이를 렌더링하면, 스톱워치를 Pause하고 있는 동안에도 내부적으로 계속 시간이 카운트된다. 따라서, Pause하는 순간 savedTime에 시간을 누적하고, 다시 Play를 누르면 currentTime, initialTime을 초기화하면 문제 없이 일시정지(Pause)를 구현할 수 있다.

한 가지 주의할 것은, 만약 스톱워치를 밀리초 단위까지 보여줄 경우 밀리초 부분은 너무 빨리 바뀌어서 텍스트가 덜덜 떨리는 것 처럼 보인다.

이는 밀리초 부분을 과감히 포기하거나, 모노스페이스 폰트를 활용하여 해결할 수 있을 것이다. 아니면 시/분/초/밀리초 별로 고정된 크기의 `<span>` 에 분리하는 방법도 있을 듯?

---

완성된 코드와 실제 데모는 아래 Codepen 링크에서 확인할 수 있다.

[React Stopwatch](https://codepen.io/jinh0park/pen/eYXwzBr)
