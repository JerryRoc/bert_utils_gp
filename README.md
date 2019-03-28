# bert-utils-gp

基于Google开源的[BERT](https://github.com/google-research/bert)代码，进行了进一步的简化，方便生成句向量

1. 下载BERT的pre-trained模型

2. 把下载好的模型添加到本目录下

3. 句向量生成

注意参数必须是一个list。

```
from bert_utils_gp import BertEncoder, args
bert = BertEncoder(args)
bert.encode(['hello world'])
```