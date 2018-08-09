[CIKM AnalytiCup 2018](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.11165320.5678.1.61a15eb5olItEx&raceId=231661) – 阿里小蜜机器人跨语言短文本匹配算法竞赛 - 深度模型部分

### 方案说明：

在此次比赛中，深度模型线下测试loss能达到0.32左右，但是线上loss只能到0.40左右，原因可能是因为语料过少，导致模型过拟合。

所以我们团队采用的方法是将深度模型训练的结果当做feature，拿到传统模型上训练。

传统模型方案：[Are you happy](https://github.com/SJHBXShub/Are-you-happy)

### 比赛成绩：
- 第二阶段 18/1027

### 参考文献：
- ELA model ：[Enhanced LSTM for Natural Language Inference](https://arxiv.org/pdf/1609.06038)
- CNN3: [Is That a Duplicate Quora Question](https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur/)
