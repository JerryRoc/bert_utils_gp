# bert-utils-gp

����Google��Դ��[BERT](https://github.com/google-research/bert)���룬�����˽�һ���ļ򻯣��������ɾ�����

1. ����BERT��pre-trainedģ��

2. �����غõ�ģ����ӵ���Ŀ¼��

3. ����������

ע�����������һ��list��

```
from bert_utils_gp import BertEncoder, args
bert = BertEncoder(args)
bert.encode(['hello world'])
```