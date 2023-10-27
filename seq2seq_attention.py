import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
from tensorflow import keras
import unicodedata
import re
from sklearn.model_selection import train_test_split

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)
en_spa_file_path = "./data_spa_en/spa.txt"


def unicode_to_ascii(s):
    '''
    英语大部分都是ascii，但是西班牙语有些特殊字符是unicode，需要进行转换,
    用ASCII原因是因为unicode的数据比较大
    :param s:需要转换的词
    :return:NFD表示如果有一个unicode是多个ASCII组成的，就把这个ascii拆开，Mn是重音，如果不是重音，转化。
    '''
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


en_sentence = 'Then what?'
sp_sentence = '¿Entonces qué?'  # 调用上面那个函数处理时候，把e和那个重音符号分开，然后忽略重音符号，转化
print(unicode_to_ascii(en_sentence))
print(unicode_to_ascii(sp_sentence))


def preprocess_sentence(s):
    '''
    标点符号与词语分开，去掉部分空格
    :return:
    '''
    s = unicode_to_ascii(s.lower().strip())  # 变小写，去空格
    s = re.sub(r"([?,.!¿])", r" \1 ", s)  # 中括号代表匹配到任何一个都转化，_\1_是空格。标点前后加空格，但这样加了后可能有多余空格存在，还要去除
    s = re.sub(r'[" "]+', " ", s)  # 一个或者多个空格时候都用一个空格替代
    s = re.sub(r"[^a-zA-Z?,!¿.]", " ", s)  # 除了a-zA-Z?,!¿之外的全给他替换成空格
    s = s.rstrip().strip()  # rstrip去掉前面空格和strip去掉后面空格
    s = '<start> ' + s + ' <end>'  # 添加特殊字符
    return s


print(preprocess_sentence(en_sentence))
print(preprocess_sentence(sp_sentence))


def parse_data(filename):
    '''
    分开英语和西班牙语，并且把数据转换好，有标签，ASCII编码，无乱七八糟终字符
    :param filename:
    :return:
    '''
    lines = open(filename, encoding = "UTF-8").read().strip().split('\n')  # 读所有的行,去掉后面的空格*******************
    sentence_pairs = [line.split('\t') for line in lines]  # 对每个字符进行用\t分割
    preprocess_sentence_pairs = [
        (preprocess_sentence(en), preprocess_sentence(sp))
        for en, sp in sentence_pairs
    ]
    return zip(*preprocess_sentence_pairs)  # 解包把元组解开，然后对应位置再拼起来,因为上面的是元组一对对的，en一组，sp一组


en_dataset, sp_dataset = parse_data(en_spa_file_path)
print(en_dataset[-1])
print(sp_dataset[-1])


def tokenizer(lang):  # 词到ID的转化
    lang_tokenizer = keras.preprocessing.text.Tokenizer(num_words = None,  # 设置词表的长度，没有代表不限制 Tokenizer 分词
                                                        filters = '', split = ' ')  # filters为黑名单，为空。split以什么为分割
    lang_tokenizer.fit_on_texts(lang)  # 统计词频，生成词表 num_words参数会对词表删减
    tensor = lang_tokenizer.texts_to_sequences(lang)  # 文本转成ID？ tensor 是编码的id
    tensor = keras.preprocessing.sequence.pad_sequences(tensor, padding = "post")  # 做padding操作，在后面
    return tensor, lang_tokenizer


input_tensor, input_tokenizer = tokenizer(sp_dataset)  # 为了快速拟合,从sp----->en
output_tensor, output_tokenizer = tokenizer(en_dataset)


def max_len(tensor):  # 数据中最长的样本的长度
    return max(len(t) for t in tensor)


max_len_input = max_len(input_tensor)
max_len_output = max_len(output_tensor)
print(max_len_input)
print(max_len_output)

# 切分数据集
input_train, input_eval, output_train, output_eval = train_test_split(input_tensor, output_tensor, test_size = 0.2)
print(len(input_train), len(input_eval), len(output_train), len(output_eval))


def convert(example, tokenizer):
    for t in example:
        if t != 0:
            print('%d--->%s' % (t, tokenizer.index_word[t]))


convert(input_train[0], input_tokenizer)
print()
convert(output_train[0], output_tokenizer)


# 转换成功，现在可以构建数据集了
def make_dataset(input_tensor, output_tensor, batch_size, epochs, shuffle):
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, output_tensor))
    if shuffle:
        dataset = dataset.shuffle(30000)
    dataset = dataset.repeat(epochs).batch(batch_size, drop_remainder = True)
    return dataset


batch_size = 2
epochs = 20
train_dataset = make_dataset(input_train, output_train, batch_size, epochs, True)
eval_dataset = make_dataset(input_eval, output_eval, batch_size, 1, False)
for x, y in train_dataset.take(1):
    print(x.shape)
    print(y.shape)
    print(x)
    print(y)

embedding_units = 256  # 每个word转成embedding是多少，input和output一样，好像也就是encode和decode
units = 1024  # 循环神经网络的size大小，encode和decode一样
input_vocab_size = len(input_tokenizer.word_index) + 1
output_vocab_size = len(output_tokenizer.word_index) + 1


class Encoder(keras.Model):
    def __init__(self, vocab_size, embedding_units, encoding_units, batch_size):
        # encoding_units是LSTM或者RNN的大小
        super(Encoder, self).__init__()  # 调用父类函数
        self.batch_size = batch_size
        self.encoding_units = encoding_units
        self.embedding = keras.layers.Embedding(vocab_size, embedding_units)  # 这些定义的层次结构当成函数来使用
        self.gru = keras.layers.GRU(self.encoding_units,  # GRU是lstm的变种，合并了遗忘门和输入门，遗忘门和输入门加起来等于1
                                    return_sequences = True,  # 因为attention的计算需要Encoder的每一个状态的输出，所以这里为true
                                    return_state = True,  # lstm在最后的输出门还要经过一个变换，而GRU则只有一个输出
                                    recurrent_initializer = "glorot_uniform")

    def call(self, x, hidden):
        # hidden是隐含状态的初始化，encoder也是一个lstm，其间的隐含状态
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialized_hidden_state(self):
        # 可以初始化一个全0的隐含状态，传给call函数
        return tf.zeros((self.batch_size, self.encoding_units))  # batch * 1024


encoder = Encoder(input_vocab_size, embedding_units, units, batch_size)
sample_hidden = encoder.initialized_hidden_state()  # 初始化的hidden状态
sample_output, sample_hidden = encoder.call(x, sample_hidden)
# sample_output-->encoder_outputs    sample_hidden-->decoder_hidden(状态)
print(sample_output.shape)  # (64, 16, 1024) 1024就是网络的units,16是长度
print(sample_hidden.shape)  # (64, 1024)


class BahdanauAttention(keras.Model):
    def __init__(self, units):  # 这边的units 以及构造函数里面的是公式里面经过tanh的units，而下面注释是网络传递层数1024
        super(BahdanauAttention, self).__init__()
        self.W1 = keras.layers.Dense(units)
        self.W2 = keras.layers.Dense(units)  # 这两矩阵是为给encode、和decode的output分别做全连接,公式
        self.V = keras.layers.Dense(1)

    def call(self, decoder_hidden, encoder_outputs):
        '''
        decoder_hidden.shape = (batch_size,units),encoder_output.shape = (batch_size,lengths,units)
        两者的维度不匹配，所以要进行对decoder_hidden的扩展维度,这里在第一个维度扩展
        :param decoder_hidden: decode里面每一个隐含状态
        :param encoder_outputs: encode每一步输出
        :return:
        '''
        decoder_hidden_with_time_axis = tf.expand_dims(decoder_hidden, 1)  # decoder_hidden.shape = (batch_size,1,units)
        score = self.V(tf.nn.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden_with_time_axis)))
        # 在没经过V之前算式shape =(batch_size,lengths,units) ， 经过之后(batch_size,lengths,1)
        # 然后得到了score就可以在lengths那个维度上算attention的权重了
        attention_weights = tf.nn.softmax(score, axis = 1)  # 这个attention的shape是(batch_size,lengths,1)
        context_vector = attention_weights * encoder_outputs
        # 上面的乘法两个维度并不匹配(batch_size,lengths,1)*(batch_size,lengths,units)这里会把1广播成units维度大小
        # 所以最后这个context_vector维度是(batch_size,lengths,units)，这里是对encoder_outputs加权
        context_vector = tf.reduce_sum(context_vector, axis = 1)  # context_vector维度是(batch_size,units)，对length这个维度进行求和
        return context_vector, attention_weights  # 返回这个权重是为衡量在当前这一步的decoder_hidden，encoder_outputs关系是什么样的


attention_model = BahdanauAttention(10)
attenton_results, attention_weight = attention_model(sample_hidden, sample_output)  # call方法
print(attenton_results.shape)
print(attention_weight.shape)


class Decoder(keras.Model):
    def __init__(self, vocab_size, embedding_units, decoding_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.decoding_units = decoding_units
        self.embedding = keras.layers.Embedding(vocab_size, embedding_units)  # embedding layer
        self.gru = keras.layers.GRU(self.decoding_units,  # GRU是lstm的变种，合并了遗忘门和输出门
                                    return_sequences = True,
                                    return_state = True,
                                    recurrent_initializer = "glorot_uniform")
        self.fc = keras.layers.Dense(vocab_size)  # 输出我预测的某个词
        self.attention = BahdanauAttention(self.decoding_units)  # 每一步都会被调用至于为啥是decoding_units有待研究

    def call(self, x, hidden, encoding_outputs):
        '''

        :param x: decoder的输入
        :param hidden: 与输入x对应的上一个隐含状态，在decode的网络中传递的隐含状态
        :param encoding_outputs: encode的输出，有点迷啊这里
        :return:
        '''
        context_vector, attention_weights = self.attention(hidden, encoding_outputs)
        # context_vector.shape = (batch_size,units)
        # 先对x进行embedding ，在做之前x.shape=(batch_size,1),由于这边是单步的decode，对encoding_outputs做一个预测？迷
        # 完了以后shape为（batch_size,1,embedding_units）
        x = self.embedding(x)
        # 这一步给根据我画的图的final input理解，x和context_vector进行拼接然后送给decode的GRU，但是维度不匹配，还要转换
        combined_x = tf.concat([tf.expand_dims(context_vector, axis = 1), x], axis = -1)
        # 里面的expand_dims先扩展，然后拼接，利用batch size这个维度也就是-1索引的最后一个维度拼接
        output, state = self.gru(combined_x)  # 两个大小分别(batch_size ,1,decoding_units),(batch_size,decoding_units)
        # 下面再对output进行reshape
        output = tf.reshape(output, (-1, output.shape[2]))  # 做完后变成 (batch_size,decoding_units)
        # 再去做全连接
        output = self.fc(output)  # 此时维度变成(batch_size,vocab_size)
        return output, state, attention_weights


decoder = Decoder(output_vocab_size, embedding_units, units, batch_size)
outputs = decoder(tf.random.uniform((batch_size, 1)), sample_hidden, sample_output)
decoder_output, decoder_hidden, decoder_aw = outputs
print()
print(decoder_output.shape)  # (64, 4935)
print(decoder_hidden.shape)  # (64, 1024)
print(decoder_aw.shape)  # (64, 16, 1)

optimizer = keras.optimizers.Adam()
# 在众多的词语ID里面预测哪一个才是正确是词语ID，是一个分类问题
loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits = True,  # 这边设置成true是因为上面在经过全连接时候并没有使用激活函数
                                                         reduction = 'none')  # 这边设置成None因为要进行其他操作，看下面，与分布式求和有关
checkpoint_dir = './training_checkpoints_big_data'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer = optimizer,
                                 encoder = encoder,
                                 decoder = decoder)


# 将 输入 传送至 编码器，编码器返回 编码器输出 和 编码器隐藏层状态。
# 将编码器输出、编码器隐藏层状态和解码器输入（即 开始标记）传送至解码器。
# 解码器返回 预测 和 解码器隐藏层状态。
# 解码器隐藏层状态被传送回模型，预测被用于计算损失。
# 使用 教师强制 （teacher forcing） 决定解码器的下一个输入。
# 教师强制 是将 目标词 作为 下一个输入 传送至解码器的技术。
# 最后一步是计算梯度，并将其应用于优化器和反向传播。


def loss_func(real, pred):
    '''

    :param real: 真实值
    :param pred: 预测值
    :return:
    '''
    # mask 是为了让padding的值不参与损失函数，或者说是把那些损失函数的值给过滤掉
    mask = tf.math.logical_not(tf.math.equal(real, 0))  # 先判断是不是0，是0也就是padding的部分，值为1。然后取反为零
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype = loss_.dtype)  # 做了类型转换才可以相乘，改为float
    loss_ *= mask  # 乘完以后里面的loss的padding就没有了
    # 这个mask机制是NLP领域常用的东西，就是把padding的部分给去掉
    return tf.reduce_mean(loss_)  # 这边就是上面那个reduction为啥设置成none，是因为想求了平均，过滤padding乘完mask之后在做聚合


# 说是上面是计算单步损失函数，下面是多步损失函数的,并转化为图结构加速
@tf.function
def train_step(inp, targ, encoding_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        encoding_outputs, encoding_hidden = encoder(inp, encoding_hidden)  # 这个encoding_hidden就是初始状态，调用上面那个类call
        # 上面返回的encoding_hidden也就是第一个的decoding_hidden
        decoding_hidden = encoding_hidden
        # 举个例子：decoding每一步的输入都是GT？
        '''
        <start> I am here <end>
        1.<start>->I
        2.I->am
        3.am->here
        4.here-><end>
        采用lstm情况下，由于网络具有记忆，之前所有的信息在下一次输入都会有涉及。i 也有start的信息
        然后由于end的下一个没有字符去预测了，所以不需要进行损失函数计算，所以索引到-1
        '''
        for t in range(0, targ.shape[1] - 1):  # 循环少一次就是end少了一次
            decoding_input = tf.expand_dims(targ[:, t], 1)  # 直接从targ里面取，但是取出来就是个向量，要扩展成矩阵
            # targ[:, t] 表示batch size 那个维度要全取，后面维度只取一个数
            predictions, decoding_hidden, _ = decoder(decoding_input, decoding_hidden, encoding_outputs)
            # 再调用上面那个类
            loss += loss_func(targ[:, t + 1], predictions)  # 得到了多步的损失函数值
    # 每个batch-size都平均的损失函数,有时候每个人设置的batch不一样没法评估大小，
    batch_loss = loss / int(targ.shape[0])  # 就使为了统一能比大小
    variables = encoder.trainable_variables + decoder.trainable_variables  # 列表相加，列表合并,非对应元素相加
    gradients = tape.gradient(loss, variables)  # 用哪个loss无所谓，就是一个系数不同
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


# 训练模型

epochs = 10
steps_per_epoch = len(input_tensor) // batch_size

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))  # 载入最新检查点
for epoch in range(epochs):
    start = time.time()
    encoding_hidden = encoder.initialized_hidden_state()  # 每次训练前初始化这个
    total_loss = 0
    for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, encoding_hidden)
        total_loss += batch_loss
        if batch % 100 == 0:
            print("Epoch {} Batch {} Loss{:.4f}".format(epoch + 1, batch, batch_loss.numpy()))
    # 每 2 个周期（epoch），保存（检查点）一次模型
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)
    print("Epoch {} Loss {:.4f}".format(epoch + 1, total_loss / steps_per_epoch))  # 总信息
    print("Time take for 1 epoch {} sec\n".format(time.time() - start))

# encoder.save_weights("encoder.h5")
# decoder.save_weights("decoder.h5")

# 和train不一样的是，这边的evaluate 在decoding时候下一步的输入是上一步的输出，而在train中下一步的输入就是GT
def evaluate(input_sentence):
    attention_matrix = np.zeros((max_len_output, max_len_input))  # 保存注意力权重，注意力（或者说是输出？）和输入有关系，有输出个数个这样的注意力向量构成矩阵
    input_sentence = preprocess_sentence(input_sentence)  # 正则预处理
    # 转换成ID
    inputs = [input_tokenizer.word_index[token] for token in input_sentence.split(' ')]
    # 做padding
    inputs = keras.preprocessing.sequence.pad_sequences([inputs],
                                                        maxlen = max_len_input,
                                                        padding = "post")  # 在后面padding
    # 转换成tensor
    inputs = tf.convert_to_tensor(inputs)
    # 保存翻译结果
    results = ''
    # encoding_hidden = encoder.initialized_hidden_state()
    encoding_hidden = tf.zeros((1, units))  # batch size 是一，而上面的是训练时候用的64 ，用就报错维度不匹配

    encoding_outputs, encoding_hidden = encoder(inputs, encoding_hidden)
    decoding_hidden = encoding_hidden  # 得到的就是decoding_hidden初始值
    # 如何做预测：
    # <start>-> A
    # A->B->C 与文本生成很像
    decoding_input = tf.expand_dims(
        [output_tokenizer.word_index['<start>']], 0)  # 取的是ID
    # 数据必须是二维的所以要扩展维度，扩展完是shape=（1，1）
    for t in range(max_len_output):
        predictions, decoding_hidden, attention_weights = decoder(
            decoding_input, decoding_hidden, encoding_outputs)
        # 这边的得到的attention_weights是(batch_size,input_length,1)
        # 也就是(1,16,1)也许不是16 需要存到开始的向量里面去，前后两维度消除，需要reshape
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_matrix[t] = attention_weights.numpy()  # 由于是tensor 需要用numpy把值取出来
        # 有了predictions就可以找出概率最大的作为下一步输入了
        # 这里predictions的shape：（batch_size,vocab_size）
        # 也就是(1,4935) 然后取出4935中那个值的概率是最大的
        predicted_id = tf.argmax(predictions[0]).numpy()
        # 保存翻译结果
        # 如果遇到结束就终止，否则更新decoding-input
        results += output_tokenizer.index_word[predicted_id] + ' '  # 获取ID对应的词，加上空格
        if output_tokenizer.index_word[predicted_id] == '<end>':  # 如果遇到end就直接结束，否则更新decoder的输入，用上一次预测的输出
            return results, input_sentence, attention_matrix
        decoding_input = tf.expand_dims([predicted_id], 0)  # 仍需扩展，和上面一开始扩展一样的
    return results, input_sentence, attention_matrix


# 注意关系可视化
def plt_attention(attention_matrix, input_sentence, predicted_sentence):
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention_matrix, cmap = "viridis")  # 配色方案，对不同的值施加不同的颜色，看起来更直观
    font_dict = {'fontsize': 14}
    ax.set_xticklabels([''] + input_sentence, fontdict = font_dict, rotation = 90)  # 翻译文字添加，每个word做个90°翻转
    ax.set_yticklabels([''] + predicted_sentence, fontdict = font_dict)  # 加个‘’ 是为了在设置x，y的时候都是从第一个值开始
    plt.show()


def translate(input_sentence):
    results, input_sentence, attention_matrix = evaluate(input_sentence)
    print("input: %s" % (input_sentence))
    print("predicted: %s" % (results))
    # 预处理attention_matrix：input的做padding不打印出来
    # output的没达到max-len的也去掉 前面的维度是output的，先是result
    attention_matrix = attention_matrix[:len(results.split(' ')),  # result长度之外的维度去掉
                       :len(input_sentence.split(' '))]  # input 超出的维度去掉
    plt_attention(attention_matrix, input_sentence.split(' '),
                  results.split(' '))


# 恢复检查点目录 （checkpoint_dir） 中最新的检查点
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
translate(u'Siempre te amaré.')
# 作用：后面字符串以 Unicode 格式 进行编码，一般用在中文字符串前面，防止因为源码储存格式问题，导致再次使用时出现乱码。
# PS：不是仅仅是针对中文, 可以针对任何的字符串，代表是对字符串进行。一般英文字符在使用各种编码下,，基本都可以正常解析, 所以一般不带u
translate('Es un buen día.')

