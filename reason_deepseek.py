import json
import requests
from typing import Dict, List, Any
from tqdm import tqdm  # 添加进度条库

class NewsAnalyzer:
    def __init__(self, api_key: str):
        self.api_endpoint = "https://api.deepseek.com/v1/chat/completions"
        self.api_key = api_key
        self.prompt_template = """As a logical reasoning expert, please analyze the truthfulness of the following news content:
News: {news_content}

Analysis Criteria (Logic and Common Sense):
- Analyze from a common sense perspective, such as the political systems of various countries, living habits of animals, and other common knowledge
- Whether the cause-effect relationships follow logical patterns
- Whether this is physically possible according to scientific laws
- Whether any numbers or statistics are realistically feasible

Output Format:
Analysis: [2-3 sentence logical assessment]
Prediction: true/fake
Important: Keep the entire output under 190 tokens."""

    def analyze_news(self, news_content: str) -> Dict[str, str]:
        """调用DeepSeek API分析单条新闻"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": self.prompt_template.format(news_content=news_content)}
            ],
            "temperature": 0.1
        }
        
        try:
            response = requests.post(self.api_endpoint, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            analysis_text = result['choices'][0]['message']['content']
            
            # 解析输出
            analysis, prediction = self._parse_output(analysis_text)
            
            return {
                "analysis": analysis,
                "prediction": prediction
            }
            
        except Exception as e:
            print(f"API调用失败: {str(e)}")  # 添加错误提示
            return {
                "analysis": f"API调用失败: {str(e)}",
                "prediction": "error"
            }

    def _parse_output(self, output_text: str) -> tuple:
        """解析API返回的文本"""
        lines = output_text.strip().split('\n')
        analysis = ""
        prediction = ""
        
        for line in lines:
            if line.startswith("Analysis:"):
                analysis = line.replace("Analysis:", "").strip()
            elif line.startswith("Prediction:"):
                prediction_text = line.replace("Prediction:", "").strip().lower()
                prediction = "true" if "true" in prediction_text else "fake"
        
        return analysis, prediction

    def process_json_file(self, input_file: str, output_file: str, content_column: str = "content"):
        """批量处理JSON文件"""
        print(f"正在读取文件: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_items = 0
        processed_items = 0
        
        if isinstance(data, list):
            total_items = len(data)
            print(f"发现 {total_items} 条待处理数据")
            
            # 使用tqdm添加进度条
            for item in tqdm(data, desc="处理进度", unit="条"):
                if content_column in item:
                    result = self.analyze_news(item[content_column])
                    item["analysis"] = result
                    processed_items += 1
                    
                    # 可选：每处理10条保存一次进度
                    if processed_items % 10 == 0:
                        self._save_progress(data, output_file)
                        
        elif isinstance(data, dict):
            print("处理单条数据")
            if content_column in data:
                result = self.analyze_news(data[content_column])
                data["analysis"] = result
                processed_items = 1
        
        # 最终保存
        print(f"正在保存结果到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"处理完成！共处理 {processed_items}/{total_items if total_items > 0 else 1} 条数据")
        return data
    
    def _save_progress(self, data: Any, output_file: str):
        """保存处理进度"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

# 使用示例
if __name__ == "__main__":
    # 初始化分析器
    analyzer = NewsAnalyzer(api_key="YOUR_DEEPSE")
    
    # 批量处理JSON文件
    input_path = ""
    output_path = ""
    
    print("开始新闻分析任务...")
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print("-" * 50)
    
    analyzer.process_json_file(input_path, output_path)
    
    print("\n任务完成！")