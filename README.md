Sentiment Analysis
=====================
Tensorflow implementation of LSTM model for Sentiment Analysis.

**Setup**

* Tensorflow, version = r1.0.1
* Python 3.6

sentiment analysis: sequence classification

**generate_data.py**

数据预处理，将原始数据（/raw文件夹下的文件）处理成句子+情感标签的形式（/data）

**data_utils.py**

将数据处理方便模型读取的形式

**sa_model.py**

sentiment analysis任务的LSTM模型实现

**run_sa.py**

LSTM模型的训练

**sa_inference.py**

用已经训练好的模型对用户的输入进行情感分类

**Contact**

* Feel free to email liuaiting37@gmail.com for any pertinent questions/bugs regarding the code.