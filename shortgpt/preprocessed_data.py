from datasets import load_dataset
from utils import *

def get_preprocessed_samsum(tokenizer, split_type = "train", num_samples = None):
    if num_samples is None:
        dataset = load_dataset("samsum", split=split_type)
    else:
        dataset = load_dataset("samsum", split = f"{split_type}[:{num_samples}]")

    prompt = (
        f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(dialog=sample["dialogue"]),
            "summary": sample["summary"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"], add_special_tokens=False)
        sample = {
            "input_ids": prompt + summary,
            "attention_mask" : [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset


def get_preprocessed_c4(tokenizer, split_type = "train", num_samples = None, trim = False, trim_length = 128):

    dataset = load_dataset('allenai/c4', data_files="en/c4-train.00001-of-01024.json.gz", split = split_type)
    dataset = dataset.shuffle(seed=42).select([i for i in range(num_samples)])
    
    def tokenize_add_label(sample):
        text_tokenized = tokenizer(sample["text"], add_special_tokens=False, padding='max_length', max_length = trim_length)
        text_tokenized = text_tokenized.input_ids
        sample = {
            "input_ids": text_tokenized[:trim_length],
            # "attention_mask" : [1]*len(text_tokenized),
            "labels": text_tokenized[:trim_length].copy(),
        }
        return sample

    # dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    return dataset

def get_preprocessed_c4_zh(tokenizer, split_type = "train", num_samples = None, trim = False, trim_length = 128):

    dataset = load_dataset('allenai/c4', data_files="multilingual/c4-zh-validation*.json.gz",split=split_type)
    dataset = dataset.filter(lambda example: len(example["text"].strip()) > 100)
    dataset = dataset.shuffle(seed=42).select([i for i in range(num_samples)])
    
    def tokenize_add_label(sample):
        text_tokenized = tokenizer(sample["text"], add_special_tokens=False, padding='max_length', max_length = trim_length)
        text_tokenized = text_tokenized.input_ids
        sample = {
            "input_ids": text_tokenized[:trim_length],
            # "attention_mask" : [1]*len(text_tokenized),
            "labels": text_tokenized[:trim_length].copy(),
        }
        return sample

    # dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    return dataset

configs = ['accountant', 'advanced_mathematics', 'art_studies', 'basic_medicine', 'business_administration', 'chinese_language_and_literature', 'civil_servant', 'clinical_medicine', 'college_chemistry', 'college_economics', 'college_physics', 'college_programming', 'computer_architecture', 'computer_network', 'discrete_mathematics', 'education_science', 'electrical_engineer', 'environmental_impact_assessment_engineer', 'fire_engineer', 'high_school_biology', 'high_school_chemistry', 'high_school_chinese', 'high_school_geography', 'high_school_history', 'high_school_mathematics', 'high_school_physics', 'high_school_politics', 'ideological_and_moral_cultivation', 'law', 'legal_professional', 'logic', 'mao_zedong_thought', 'marxism', 'metrology_engineer', 'middle_school_biology', 'middle_school_chemistry', 'middle_school_geography', 'middle_school_history', 'middle_school_mathematics', 'middle_school_physics', 'middle_school_politics', 'modern_chinese_history', 'operating_system', 'physician', 'plant_protection', 'probability_and_statistics', 'professional_tour_guide', 'sports_science', 'tax_accountant', 'teacher_qualification', 'urban_and_rural_planner', 'veterinary_medicine']
SUBJECTS = {
    "computer_network": "计算机网络",
    "operating_system": "操作系统",
    "computer_architecture": "计算机组成",
    "college_programming": "大学编程",
    "college_physics": "大学物理",
    "college_chemistry": "大学化学",
    "advanced_mathematics": "高等数学",
    "probability_and_statistics": "概率统计",
    "discrete_mathematics": "离散数学",
    "electrical_engineer": "注册电气工程师",
    "metrology_engineer": "注册计量师",
    "high_school_mathematics": "高中数学",
    "high_school_physics": "高中物理",
    "high_school_chemistry": "高中化学",
    "high_school_biology": "高中生物",
    "middle_school_mathematics": "初中数学",
    "middle_school_biology": "初中生物",
    "middle_school_physics": "初中物理",
    "middle_school_chemistry": "初中化学",
    "veterinary_medicine": "兽医学",
    "college_economics": "大学经济学",
    "business_administration": "工商管理",
    "marxism": "马克思主义基本原理",
    "mao_zedong_thought": "毛泽东思想和中国特色社会主义理论体系概论",
    "education_science": "教育学",
    "teacher_qualification": "教师资格",
    "high_school_politics": "高中政治",
    "high_school_geography": "高中地理",
    "middle_school_politics": "初中政治",
    "middle_school_geography": "初中地理",
    "modern_chinese_history": "近代史纲要",
    "ideological_and_moral_cultivation": "思想道德修养与法律基础",
    "logic": "逻辑学",
    "law": "法学",
    "chinese_language_and_literature": "中国语言文学",
    "art_studies": "艺术学",
    "professional_tour_guide": "导游资格",
    "legal_professional": "法律职业资格",
    "high_school_chinese": "高中语文",
    "high_school_history": "高中历史",
    "middle_school_history": "初中历史",
    "civil_servant": "公务员",
    "sports_science": "体育学",
    "plant_protection": "植物保护",
    "basic_medicine": "基础医学",
    "clinical_medicine": "临床医学",
    "urban_and_rural_planner": "注册城乡规划师",
    "accountant": "注册会计师",
    "fire_engineer": "注册消防工程师",
    "environmental_impact_assessment_engineer": "环境影响评价工程师",
    "tax_accountant": "税务师",
    "physician": "医师资格",
}
from tqdm import tqdm

def get_preprocessed_ceval(tokenizer, split_type = "train", num_samples = None, trim = False, trim_length = 128):

    input_ids, labels = [],[]
    for subject_eng, subject_zh in tqdm(SUBJECTS.items()):
        description = (
                    f"以下是中国关于{subject_zh}的单项选择题，请选出其中的正确答案。\n\n"
                )
        dataset=load_dataset(r"ceval/ceval-exam",subject_eng,split="val")
        trim_length = 128
        
        def apply_prompt_template(sample):
            x = {
                "prompt": description + "{}\nA. {}\nB. {}\nC. {}\nD. {}\n答案：".format(sample["question"].strip(), sample['A'], sample['B'], sample['C'], sample['D']),
                "result": "{}".format(sample['answer'])
            }
            return x
        
        dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
        
        def tokenize_add_label(sample):
            prompt = tokenizer.encode(sample["prompt"], add_special_tokens=False,padding='max_length', max_length = trim_length)
            result = tokenizer.encode(sample["result"], add_special_tokens=False)
            sample = {
                "input_ids": prompt[:trim_length] + result,
                # "attention_mask" : [1] * (len(prompt) + len(result)),
                "labels": [-100] * len(prompt[:trim_length]) + result,
                }
            # if len(prompt) < trim_length:
            # global input_ids
            # global labels
            input_ids.append(sample["input_ids"])
            labels.append(sample["labels"])
                

            return sample
        dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    from datasets import Dataset
    # breakpoint()
    final_dataset =  Dataset.from_dict({"input_ids": input_ids,"labels": labels})
    return final_dataset

def get_preprocessed_wikitext2(tokenizer, split_type = "train", num_samples = None, trim = False, trim_length = 128):
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split = split_type)
    
    dataset = dataset.filter(lambda example: len(example["text"].strip()) > 100)
    dataset = dataset.shuffle(seed=42).select([i for i in range(num_samples)])
    if num_samples is not None:
        dataset = dataset.select([i for i in range(num_samples)])
    def tokenize_add_label(sample):
        text_tokenized = tokenizer(sample["text"], add_special_tokens=False, padding='max_length', max_length = trim_length)
        text_tokenized = text_tokenized.input_ids
        sample = {
            "input_ids": text_tokenized[:trim_length],
            # "attention_mask" : [1]*len(text_tokenized),
            "labels": text_tokenized[:trim_length].copy(),
        }
        return sample

    # dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    return dataset

