import json
import sys
from pathlib import Path

# Reuse the single definition of the override convention rather than restating the naming rule here
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code" / "train"))
from ablation_config import override  # noqa: E402

# ========== 配置路径 ==========
# Defaults are the pipeline's absolute paths. An ablation variant overrides them so it converts its
# own dump into its own working directory instead of the shared one.
input_txt_path = Path(override(
    "txt2json.INPUT_TXT", r"C:\Users\shy\AI-report\data\train\Hard_samples\hard_samples.txt"))
output_json_path = Path(override(
    "txt2json.OUTPUT_JSON", r"C:\Users\shy\AI-report\data\train\Hard_samples\origin_hard_samples.json"))

# ========== 定义函数 ==========
def string_to_structured_data(s):
    """
    将拼接字符串解析回原始字段
    格式: error_text-*-reasoning-*-gold_pos-*-gold_spec-*-gold_corr
    """
    parts = s.split("-*-")
    if len(parts) != 5:
        raise ValueError(f"字段数量不匹配（期望5个，实际{len(parts)}个）: {s}")
    error_text, reasoning, gold_pos, gold_spec, gold_corr = parts
    return {
        "error_text": error_text,
        "reasoning": reasoning,
        "error_position": gold_pos,
        "specific_error_reason": gold_spec,
        "corrected_content": gold_corr,
    }

def load_txt_to_list(file_path, separator="*|||*\n"):
    """
    读取txt并按分隔符切分为字符串列表
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    # 去掉末尾可能残留的分隔符
    if content.endswith(separator):
        content = content[: -len(separator)]
    if not content.strip():
        return []
    return content.split(separator)

# ========== 主流程 ==========
if __name__ == "__main__":
    # 读取txt
    str_list = load_txt_to_list(input_txt_path, separator="*|||*\n")

    # 逐条还原
    restored_data = []
    for idx, s in enumerate(str_list):
        try:
            restored_data.append(string_to_structured_data(s))
        except ValueError as e:
            print(f"[跳过] 第{idx}条解析失败: {e}")

    # 写入json
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(restored_data, f, ensure_ascii=False, indent=2)

    print(f"成功还原样本数：{len(restored_data)}，已保存至 {output_json_path}")