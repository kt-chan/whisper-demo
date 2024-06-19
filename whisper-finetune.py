#在微调 Whisper 模型时，我们会用到几个流行的 Python 包。我们使用 datasets 来下载和准备训练数据，使用 transformers 来加载和训练 Whisper 模型。另外，我们还需要 soundfile 包来预处理音频文件，evaluate 和 jiwer 来评估模型的性能。最后，我们用 gradio 来为微调后的模型构建一个亮闪闪的演示应用。

!pip install datasets>=2.6.1
!pip install git+https://github.com/huggingface/transformers
!pip install librosa
!pip install evaluate>=0.30
!pip install jiwer
!pip install gradio

#使用 🤗 Datasets 来下载和准备数据非常简单。仅需一行代码即可完成 Common Voice 数据集的下载和准备工作。由于印地语数据非常匮乏，我们把 训练集 和 验证集合并成约 8 小时的训练数据，而测试则基于 4 小时的 测试集:

from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()

common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test", use_auth_token=True)

print(common_voice)

大#多数 ASR 数据集仅包含输入音频样本 ( audio) 和相应的转录文本 ( sentence)。 Common Voice 还包含额外的元信息，例如 accent 和 locale，在 ASR 场景中，我们可以忽略这些信息。为了使代码尽可能通用，我们只考虑基于输入音频和转录文本进行微调，而不使用额外的元信息:

common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

#Transformers Whisper 特征提取器仅用一行代码即可执行填充和声谱图变换两个操作！我们使用以下代码从预训练的 checkpoint 中加载特征提取器，为音频数据处理做好准备.我们可以通过对 Common Voice 数据集的第一个样本进行编解码来验证分词器是否正确编码了印地语字符。
#在对转录文本进行编码时，分词器在序列的开头和结尾添加“特殊标记”，其中包括文本的开始/结尾、语种标记和任务标记 (由上一步中的参数指定)。在解码时，我们可以选择“跳过”这些特殊标记，从而保证输出是纯文本形式的:

from transformers import WhisperFeatureExtractor, WhisperTokenizer

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")
input_str = common_voice["train"][0]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input: {input_str}")
print(f"Decoded w/ special: {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal: {input_str == decoded_str}")

#为了简化使用，我们可以将特征提取器和分词器 包进 到一个 WhisperProcessor 类，该类继承自 WhisperFeatureExtractor 及 WhisperTokenizer，可根据需要用于音频处理和模型预测。
from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")
print(common_voice["train"][0])

#我们将使用 dataset 的 cast_column 方法将输入音频转换至所需的采样率。该方法仅指示 datasets 让其在首次加载音频时 _即时地_对数据进行重采样，因此并不会改变原音频数据:

from datasets import Audio
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

#重新打印下 Common Voice 数据集中的第一个音频样本，可以看到其已被重采样:

print(common_voice["train"][0])

# 我们将使用 dataset 的 cast_column 方法将输入音频转换至所需的采样率。
# 该方法仅指示 datasets 让其在首次加载音频时 _即时地_对数据进行重采样，因此并不会改变原音频数据:

from datasets import Audio
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
# 重新打印下 Common Voice 数据集中的第一个音频样本，可以看到其已被重采样:
print(common_voice["train"][0])



#现在我们编写一个函数来为模型准备数据:
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=4)
