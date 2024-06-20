#在微调 Whisper 模型时，我们会用到几个流行的 Python 包。我们使用 datasets 来下载和准备训练数据，使用 transformers 来加载和训练 Whisper 模型。另外，我们还需要 soundfile 包来预处理音频文件，evaluate 和 jiwer 来评估模型的性能。最后，我们用 gradio 来为微调后的模型构建一个亮闪闪的演示应用。
# !pip install datasets>=2.6.1
# !pip install git+https://github.com/huggingface/transformers
# !pip install librosa
# !pip install evaluate>=0.30
# !pip install jiwer
# !pip install gradio

#使用 🤗 Datasets 来下载和准备数据非常简单。仅需一行代码即可完成 Common Voice 数据集的下载和准备工作。由于印地语数据非常匮乏，我们把 训练集 和 验证集合并成约 8 小时的训练数据，而测试则基于 4 小时的 测试集:
import os, sys, time
import pprint
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "npu:0"
torch_dtype = torch.float16

print('argument list: ', sys.argv)

from datasets import load_dataset, DatasetDict
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = "/root/demo/data/"
model_name = 'whisper-large-v3'
model_id = "/mnt/remote/models/whisper/whisper-large-v3/"

common_voice = DatasetDict()
common_voice["train"] = load_dataset(data_dir + r"./mozilla-foundation/common_voice_11_0", "yue",
                                     split="train+validation", trust_remote_code=True)
common_voice["test"] = load_dataset(data_dir + r"./mozilla-foundation/common_voice_11_0", "yue", split="test",
                                    trust_remote_code=True)
common_voice = common_voice.remove_columns(
    ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

# #Transformers Whisper 特征提取器仅用一行代码即可执行填充和声谱图变换两个操作！我们使用以下代码从预训练的 checkpoint 中加载特征提取器，为音频数据处理做好准备.我们可以通过对 Common Voice 数据集的第一个样本进行编解码来验证分词器是否正确编码了印地语字符。
# #在对转录文本进行编码时，分词器在序列的开头和结尾添加“特殊标记”，其中包括文本的开始/结尾、语种标记和任务标记 (由上一步中的参数指定)。在解码时，我们可以选择“跳过”这些特殊标记，从而保证输出是纯文本形式的:
#
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
tokenizer = WhisperTokenizer.from_pretrained(model_id, language="Cantonese", task="transcribe")
# input_str = common_voice["train"][0]["sentence"]
# labels = tokenizer(input_str).input_ids
# decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
# decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

# this is not required yet, just for inference usage
# 为了简化使用，我们可以将特征提取器和分词器 包进 到一个 WhisperProcessor 类，
# 该类继承自 WhisperFeatureExtractor 及 WhisperTokenizer，可根据需要用于音频处理和模型预测。
processor = WhisperProcessor.from_pretrained(model_id, language="Cantonese", task="transcribe", skip_special_tokens=True)
processor.tokenizer.skip_special_tokens = True
# print(common_voice["train"][0])

# 我们将使用 dataset 的 cast_column 方法将输入音频转换至所需的采样率。
# 我们将使用 dataset 的 cast_column 方法将输入音频转换至所需的采样率。
# 该方法仅指示 datasets 让其在首次加载音频时 _即时地_对数据进行重采样，因此并不会改变原音频数据:
from datasets import Audio

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))


# print(common_voice)

# 现在我们编写一个函数来为模型准备数据:
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

# @DEBUG
# Sample test with first 100 records
# remove these two lines for official training
# common_voice = {split: dataset.take(100) for split, dataset in common_voice.items()}
# common_voice = DatasetDict(common_voice)

print('Running long task on converting dataset ... ')
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=64)
print(common_voice)

# Load a Pre-Trained Checkpoint
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(model_id)
model.generation_config.language = "yue"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

# The data collator for a sequence-to-sequence speech model is unique in the sense that it treats the input_features and labels independently
# the input_features must be handled by the feature extractor and the labels by the tokenizer.
# We can leverage the WhisperProcessor we defined earlier to perform both the feature extractor and the tokenizer operations:
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# Define the Training Arguments
from transformers import Seq2SeqTrainingArguments

## this would roughly run 40 epoch for training data size of 2000
training_args = Seq2SeqTrainingArguments(
    output_dir="./"+model_name+"-yue",  # change to a repo name of your choice
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=2,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)



from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

print('Running long task on training the model  ... ')
trainer.train()