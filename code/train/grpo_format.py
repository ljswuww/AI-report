import os
import re
import json
import torch
import numpy as np
from typing import List
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from collections import Counter

# -------------------------- 环境配置 --------------------------

os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"
os.environ["HF_ENDPOINT"] = "[https://hf-mirror.com](https://hf-mirror.com)"

# -------------------------- 路径配置 --------------------------

OUTPUT_DIR = "/data/ai_report/lijiusong/output/model/GRPO_llmJD_CTN4"
MODEL_PATH = "/run/ai_report/lijiusong/output/model/GRPO_llmJD_CTN2/checkpoint-2600"

# -------------------------- 简化的提示词 --------------------------

SYSTEM_PROMPT = """You are a professional expert in text error diagnosis and correction, with specialized expertise in the field of AECQ automotive-grade chip testing.

### STRICT FORMAT REQUIREMENT (MUST FOLLOW EXACTLY) ###
You MUST respond in the following XML-like format with NO extra content, NO explanations, NO additional text:

<reasoning>
[Your reasoning process here]
</reasoning>
<error_position>
[Smallest possible error-containing segment (write "none" if no error)]
</error_position>
<specific_error_reason>
[Brief error reason (write "none" if no error)]
</specific_error_reason>
<corrected_text>
[Complete revised text]
</corrected_text>

### EXAMPLE ###
Input text: "The ambient operating temperature range for Grade 3 in AEC-Q100 Rev-H is -40°C to +95°C."
Your response should be:
<reasoning>
The given temperature range is incorrect according to the AEC-Q100 Rev-H specification. The standard specifies that Grade 3 has an ambient operating temperature range of -40°C to +85°C, not +95°C.
</reasoning>
<error_position>
The ambient operating temperature range for Grade 3 in AEC-Q100 Rev-H is -40°C to +95°C.
</error_position>
<specific_error_reason>
The upper temperature limit for Grade 3 is specified as +85°C, but +95°C is incorrectly stated in the text.
</specific_error_reason>
<corrected_text>
The ambient operating temperature range for Grade 3 in AEC-Q100 Rev-H is -40°C to +85°C.
</corrected_text>

### TASKS ###
1. Judge if there is an error in the text based on what you know.
2. Precisely extract the error-containing sentence.Extract the smallest complete sentence (or clause) containing the error.
3. Explain the specific cause of the error briefly.
4. Generate the complete revised text.

### FORMAT VIOLATION ###
If you do not follow this exact format, your response will be considered completely invalid."""


# -------------------------- 数据集处理（核心优化：标签校验+打乱+验证集拆分）--------------------------

def get_correction_dataset(split="train") -> (Dataset, Dataset):
    # data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data_path = "../data_source/RL_samples0102.json"
    data = load_dataset('json', data_files=data_path)['train']
    print(data)
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['error_text']}
        ],
        'error_text':x['error_text'],
        'answer': x['corrected_content'],
        'error_position': x['error_position'],     # 默认为空字符串
        'specific_error_reason': x['specific_error_reason'],
        'corrected_text': x['corrected_content'],
        'reasoning': x['reasoning']
    }) # type: ignore
    data =data.shuffle(seed=42)
    return data # type: ignore

# 加载数据集

train_dataset = get_correction_dataset()

#######代码计算奖励部分
import re
def prepare_text(text):
    """
    使用正则表达式，在每个中文字符前后插入空格。
    """
    # 在每个汉字左右各插入一个空格，实现“字符级”切分
    # 捕获组 ([\u4e00-\u9fff]) 匹配任意一个汉字；替换为 空格+汉字+空格
    # 但这样也会导致连续空格
    text = re.sub(r'([\u4e00-\u9fff])', r' \1 ', text)
    return text
#第二个版本
def get_ngrams(words, n, deduplicate=False):
    """
    生成文本的n-gram列表
    Args:
        tokens: 切分后的token列表（如[字1, 字2]或[词1, 词2]）
        n: n-gram的n值（如1/2/3）
        deduplicate: 是否去重（避免重复token干扰）
    Returns:
        n-gram列表（list）
    """
    if len(words) < n:
        return []  # 长度不足n时返回空

```
ngrams = []
for i in range(len(words) - n + 1):
    sub_words = words[i:i+n]
    ngram_str = "-".join(words[i:i+n]) 
    
    ngrams.append(ngram_str)

# 去重（可选）
if deduplicate:
    ngrams = list(set(ngrams))

return ngrams
```

def calculate_f1(pred_grams, gold_grams) -> float:
    """改进的F1计算，不去重，保留顺序信息"""  
    # 不去重，保留频率信息
    pred_counter = Counter(pred_grams)
    gold_counter = Counter(gold_grams)

```
# 计算交集（考虑频率）
common = sum((pred_counter & gold_counter).values())

precision = common / len(pred_grams) if pred_grams else 0
recall = common / len(gold_grams) if gold_grams else 0
if precision + recall == 0:
    return 0.0
return 2 * precision * recall / (precision + recall)
```

#简化计算,只使用3-gram
def calculate_3gram_f1(pred: str, gold: str) -> float:
    # pred_words = pred.split()
    # gold_words = gold.split()
    pred_text = prepare_text(pred)
    gold_text = prepare_text(gold)
    pred_words = pred_text.split()
    gold_words = gold_text.split()

```
#先排除特殊情况
if len(pred_words) == 0 or len(gold_words) == 0:
    #先排除没有元素的情况
    return 0.0
if len(pred_words) < 3 and len(gold_words) < 3:
    #3-gram至少要3个元素
    if pred == gold:
        return 1.0
    else:
        return 0.0
#3-gram.
pred_3grams = get_ngrams(pred_words, 3, deduplicate=False)
gold_3grams = get_ngrams(gold_words, 3, deduplicate=False)
score = calculate_f1(pred_3grams, gold_3grams)
return score
```

#######LLM as a judge 计算奖励部分
def extract_xml_score(text: str) -> str:
    answer = text.split("")[-1]
    answer = answer.split("")[0]
    return answer.strip()
def str_to_num(s):
    """安全地将字符串转为数字，转浮点数，失败返回0.0"""
    try:
        return float(s)
    except ValueError:
        return 0.0
import os
from openai import OpenAI
client = OpenAI(
        api_key="sk-18151238490d420e93386360444ac7c0",
        base_url="[https://api.deepseek.com](https://api.deepseek.com)")
def llm_as_a_judge(error_text,AI_pos,AI_rsn,AI_cor,gold_pos,gold_rsn,gold_cor):
    judge_prompt = f"""你是一个严谨但灵活的AECQ车规芯片测试领域纠错模型评估专家。请根据以下规则评估模型的输出，并给出最终分数。

## 评估规则

1. **全对得1.0分，任何一步错都得0.0分**：必须依次检查以下三个部分，顺序不可调换，任何一步不正确则立即停止并给总分0.0分。
   - **步骤一：检查“错误位置”**：对比AI回复的错误位置与标准答案。若**总体指向相同**，进入步骤二；若不正确，总分直接为0.0分。
   - **步骤二：检查“错误原因”**：对比AI回复的错误原因与标准答案。若**核心意思相同**，进入步骤三；若不正确，总分直接为0.0分。
   - **步骤三：检查“修正后文本”**：对比AI回复的修正后文本与标准答案。若**修正效果一致**，总分为1.0分；若不正确，总分直接为0.0分。
2. **“总体意思相同”判定标准**：
   - **错误位置**：允许表述形式不同（如“第3行第5-10字符”与“第3行‘高温测试’处”），但必须指向同一文本区间或语义单元。
   - **错误原因**：允许用词和句式差异，但必须包含相同的核心错误点（如“温度范围错误”与“温度条件不符合标准”视为相同）。
   - **修正后文本**：允许同义词替换、句式调整，但修正后的专业内容必须与标准答案等效，且不引入新错误。

## 输入信息

- 待分析错误文本：{error_text}
- AI回复错误位置：{AI_pos}
- AI回复错误原因：{AI_rsn}
- AI回复修正后文本：{AI_cor}
- 错误位置标准答案：{gold_pos}
- 错误原因标准答案：{gold_rsn}
- 修正后文本标准答案：{gold_cor}

## 输出格式

请严格按以下结构输出：

[逐步写出推理过程，说明每一步的比较判断依据。若判定为“总体意思相同”，需简要解释原因；若判定为不同，需说明具体差异。]

[最终分数，只能是1.0或0.0]

"""
    
```
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": judge_prompt},
    ],
    stream=False
)
#deepseek-reasoner
#deepseek-chat
res = response.choices[0].message.content
score = extract_xml_score(res)
score = str_to_num(score)
return score
```

#硬样本部分
def structured_data_to_string(error_text,reasoning,gold_pos,gold_spec,gold_corr):
    return f"{error_text}-*-{reasoning}-*-{gold_pos}-*-{gold_spec}-*-{gold_corr}"
def save_set_to_txt(set_data, file_path, separator="*|||*"):
    """
    将字符串集合保存到txt文件，元素间用指定分隔符隔开
    :param set_data: 要保存的字符串集合
    :param file_path: 保存的文件路径
    :param separator: 分隔符（建议用多字符，避免与元素内容冲突）
    """
    # 检查分隔符是否存在于集合元素中（避免拆分出错）
    #*******为简便，先不弄
    # for elem in set_data:
    #     if separator in elem:
    #         raise ValueError(f"元素 '{elem}' 中包含分隔符 '{separator}'，请更换分隔符！")

```
# 将集合转为字符串（集合是无序的，转字符串后用分隔符连接）
str_data = separator.join(set_data)

# 写入文件（用utf-8编码，避免中文乱码）
with open(file_path, "w", encoding="utf-8") as f:
    f.write(str_data)
print(f"硬样本已成功保存到 {file_path}")
```

#######真正的奖励函数设计

class RewardManager:
    def **init**(self):
        self.reward_history = []
        self.baseline = 0.0
        self.alpha = 0.9
        self.hard_reward = 9.0
        self.hard_sams_set = set()#硬样本池

```
def extract_field(self, text: str, field: str) -> str:
    """改进的字段提取"""
    start_tag = f"<{field}>"
    end_tag = f"</{field}>"
    
    if start_tag not in text or end_tag not in text:
        return "not find"
    
    # 提取第一个匹配到的内容
    try:
        start_idx = text.index(start_tag) + len(start_tag)
        end_idx = text.index(end_tag, start_idx)
        return text[start_idx:end_idx].strip()
    except:
        return "not find"

def core_correction_reward(self, prompts, completions, **kwargs):
    """改进的核心奖励函数"""
    responses = [completion[0]['content'] for completion in completions]
    #gold标签
    error_positions = kwargs.get('error_position', [])
    corrected_texts = kwargs.get('corrected_text', [])
    error_texts = kwargs.get('error_text', [])
    specific_error_reasons = kwargs.get('specific_error_reason', [])
    reasonings = kwargs.get('reasoning', [])
    
    rewards = []
    for response, gold_pos, gold_spec, gold_corr, error_text in zip(responses, error_positions, specific_error_reasons, corrected_texts, error_texts):
        
        # 提取回复中的字段
        res_reason = self.extract_field(response, 'reasoning')
        res_pos = self.extract_field(response, 'error_position')
        res_spec = self.extract_field(response, 'specific_error_reason')
        res_corr = self.extract_field(response, 'corrected_text')
        

        #逐过程奖励函数
        total_reward = 0.0
        if all([res_reason != "not find", res_pos != "not find", res_spec != "not find", res_corr != "not find" ]):
            #格式正确
            # 1. 格式基础分（0.5分）
            format_score = 0.5
            # 2. 错误位置F1分数（8分）
            if gold_pos == "" and res_pos.lower() == "none":
                #先判断是否为无错误样本
                pos_score = 8.0
            else:
                #这是有错误样本
                pos_f1 = calculate_3gram_f1(res_pos, gold_pos)
                
                pos_score = pos_f1 * 8.0
            
            # 3. 修正文本F1分数（1.5分）
            corr_f1 = calculate_3gram_f1(res_corr, gold_corr)
            corr_score = corr_f1 * 1.5
            # 总分（0-10分）
            code_reward = format_score + pos_score + corr_score
            # 奖励裁剪（防止异常值）
            code_reward = max(0.0, min(code_reward, 10.0))
            

            judge_reward = llm_as_a_judge(error_text,res_pos,res_spec,res_corr,
                                      gold_pos,gold_spec,gold_corr)
            #code_reward范围:0.0-10.0;judge_reward范围:0.0-1.0
            #最终code_reward范围:0.0-1.0;judge_reward:0.0-9.0
            total_reward = code_reward * 0.1 + judge_reward * 9
            
        else:
            #格式不对,分数为0
            print("format error,reward:0.0")
            total_reward = 0.0
        rewards.append(total_reward)
        

    
    # 更新基线
    avg_reward = np.mean(rewards)
    self.baseline = self.alpha * self.baseline + (1 - self.alpha) * avg_reward
    # self.reward_history.append(avg_reward)

    min_reward = np.min(rewards)
    max_reward = np.max(rewards)

    self.reward_history.append(max_reward)
    
    #硬样本保存策略
    
    sam_str = structured_data_to_string(error_texts[0],reasonings[0],
                                        error_positions[0],specific_error_reasons[0],corrected_texts[0])
    if max_reward < self.hard_reward:
        #得分低先看硬样本池里有没有，没有就保存到硬样本池，模型不具备纠正该样本的能力。
        if sam_str not in self.hard_sams_set:
            self.hard_sams_set.add(sam_str)
    else:
        #得分高，看硬样本池里有没有，有就删除，模型已具备纠正该样本的能力
        if sam_str in self.hard_sams_set:
            self.hard_sams_set.remove(sam_str)
    
    # 打印监控信息
    print(f"step-{len(self.reward_history)} -Baseline: {self.baseline:.3f}, Avg: {avg_reward:.2f}, Min: {min_reward:.2f}, Max: {max_reward:.2f}")
    
    return rewards

def save2txt(self,content,file_path):
    #以追加的方式保存到txt文件中
    with open(file_path, mode='a', encoding='utf-8') as f:
        # 写入单个字符串
        f.write(content) 
  
```

# -------------------------- 训练配置 --------------------------

training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    run_name="Qwen-Correction-GRPO-v2",
    learning_rate=5e-6,  # 260127从5e-6改到2e-5
    adam_beta1=0.9,
    adam_beta2=0.95,
    weight_decay=0.01,
    warmup_ratio=0.05,
    lr_scheduler_type='cosine',
    logging_steps=5,
    bf16=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_generations=4,  # 减少候选数，降低噪声
    generation_batch_size=4,
    max_prompt_length=896,
    max_completion_length=1280,  # 减少生成长度
    num_train_epochs=10,  # 减少epoch数
    save_steps=100,
    max_grad_norm=1.0,  # 放宽梯度裁剪
    save_total_limit=2,
    report_to="none",
)

# -------------------------- 模型加载 --------------------------

print("加载模型和tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

# 设置生成参数

model.generation_config.temperature = 0.2
model.generation_config.top_p = 0.6
model.generation_config.top_k = -1  
model.generation_config.do_sample = True
model.generation_config.max_new_tokens = training_args.max_completion_length
#这个是新加的要注意
model.generation_config.repetition_penalty = 1.1  # 抑制重复生成
model.generation_config.num_beams = 1         # 确保使用采样而非束搜索

# -------------------------- 训练准备 --------------------------

# # 加载数据集

# train_dataset, val_dataset = load_and_split_dataset()

# 初始化奖励管理器

reward_manager = RewardManager()

# 包装奖励函数

def core_reward_wrapper(prompts, completions, **kwargs):
    return reward_manager.core_correction_reward(prompts, completions, **kwargs)

# -------------------------- 训练器初始化 --------------------------

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=core_reward_wrapper,
    args=training_args,
    train_dataset=train_dataset,
)

def save_list_as_string_to_txt(lst, file_path):
    """
    直接将列表转为字符串，保存到txt文件
    :param lst: 任意类型的列表（支持字符串、数字、中文、嵌套列表/字典等）
    :param file_path: 保存路径（如 "./list_str.txt"）
    """
    # 核心：将列表直接转为字符串（保留原始格式）
    list_str = str(lst)
    # 写入文件（utf-8编码避免中文乱码，mode='w'覆盖写入，'a'为追加）
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(list_str)
    print(f"列表已以字符串形式保存到：{file_path}")

# -------------------------- 训练循环（带早停） --------------------------

try:
    print("开始GRPO训练...")
    trainer.train()
    print("训练完成，保存模型...")
finally:
    file_path = "./硬样本/hard_samples.txt"
    separator = "*|||*"
    save_set_to_txt(reward_manager.hard_sams_set, file_path, separator)
    #保存reward方便查看。reward_manager.reward_history
    reward_path = "./reward.txt"
    save_list_as_string_to_txt(reward_manager.reward_history, reward_path)

