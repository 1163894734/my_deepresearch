import argparse
import os
import threading
import time
from smolagents import InferenceClientModel
from smolagents import OpenAIModel

from dotenv import load_dotenv
from scripts.text_inspector_tool import TextInspectorTool
from scripts.custom_tools import LongWriterTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer

from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    DuckDuckGoSearchTool,
    # InferenceClientModel,
    LiteLLMModel,
    ToolCallingAgent,
)
from scripts.skill_loader import load_skills_from_directory
from scripts.long_writer_agent_v3 import LongWriterAgent


load_dotenv(override=True)

append_answer_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question", type=str, help="for example: 'How many studio albums did Mercedes Sosa release before 2007?'"
    )
    parser.add_argument("--model-id", type=str, default="o1")
    return parser.parse_args()


custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)


def create_agent(model_id="o1"):
    model_params = {
        "model_id": model_id,
        "custom_role_conversions": custom_role_conversions,
        "max_completion_tokens": 8192,
    }
    if model_id == "o1":
        model_params["reasoning_effort"] = "high"
    # model = LiteLLMModel(**model_params)

    model = OpenAIModel(
        model_id="Qwen3-235B-A22B-Instruct-2507",
        api_base="https://llmapi.paratera.com/v1",
        api_key=os.environ["DYM_API_KEY"]
    )
    skills_dir = "./skills" 
    
    print(f"Loading skills from {skills_dir}...")
    custom_tools = load_skills_from_directory(skills_dir, model=model)
    
    print(f"Loaded {len(custom_tools)} custom tools: {[t.name for t in custom_tools]}")
    # long_writer_tool = LongWriterTool(model)
    # model = OpenAIModel(
    #     model_id="deepseek-chat",  # 根据DeepSeek V3的实际模型ID调整
    #     api_base="https://api.deepseek.com/v1",  # 替换为DeepSeek V3的API基础地址
    #     api_key=os.environ["DEEPSEEK_API_KEY"],  # 需设置环境变量存储API密钥
    # )

    # model = OpenAIModel(
    #     model_id="gpt-5",  # 根据DeepSeek V3的实际模型ID调整
    #     api_base="https://api.gptsapi.net/v1",  # 替换为DeepSeek V3的API基础地址
    #     api_key=os.environ["GPTSAPI_KEY"],  # 需设置环境变量存储API密钥
    #     # max_tokens=2048,  # 强制必填：设置生成的最大 tokens 数（根据 GPTs 限制调整）
    #     # temperature=0.7,  # 可选：保持原有配置
    # )
    text_limit = 100000
    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    WEB_TOOLS = [
        # GoogleSearchTool(provider="serper"),
        DuckDuckGoSearchTool(),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]
    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
    )
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

    # 为 LongWriterAgent 添加 web_search 工具
    web_search_tool = DuckDuckGoSearchTool()
    web_search_tool.name = "web_search"  # 确保名称匹配
    
    # 创建 LongWriterAgent V3（多阶段长文本生成代理）
    long_writer_agent = LongWriterAgent(
        model=model,
        tools=custom_tools + [web_search_tool],  # 传入所有 skills + web_search
        managed_agents=[text_webbrowser_agent],  # 细粒度检索时可调用 search_agent
        max_steps=30,
        verbosity_level=2,
        name="long_writer_agent",
        description="""**专门负责长文本学术写作的智能体（论文、综述、调研报告）**

**何时必须委托给我**：
- 用户要求写"综述"、"调研报告"、"学术论文"、"技术报告"
- 要求字数超过 5000 字或 1 万字
- 需要"参考文献"、"引用文献"、"列出文献"
- 需要多章节结构化内容（引言、正文、结论等）
- 要求"完整"、"详尽"、"系统性"的文档

**我的核心能力**：
- 自动生成学术大纲 + 迭代优化
- 分段撰写 + 证据引用管理
- 反思循环确保质量
- 自动整理参考文献列表

**委托方式**：
直接调用 `long_writer_agent(task="用户的完整任务描述")`，不要自己调用 outline_generation、section_write 等工具。
""",
        provide_run_summary=True,
    )

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, TextInspectorTool(model, text_limit)] + custom_tools,
        max_steps=20,
        verbosity_level=2,
        additional_authorized_imports=["*"],
        planning_interval=2,
        # 将 long_writer_agent 和 search_agent 注册为 managed agents
        managed_agents=[long_writer_agent, text_webbrowser_agent],

    )

    return manager_agent


def main():
    args = parse_args()

    agent = create_agent(model_id="ollama/qwen2.5:1.5b")

    test ="""总结这段文本：老舍笔下的鼓书艺人，喉头翻滚着千言万语，却终被沉重现实压成一声叹息；艾青化身为鸟，甘愿以嘶哑的喉咙刺破长夜，只为向土地倾注赤子对家国的深沉热爱；而穆旦诗中的“我”则用“带血的手”，将个体的痛楚绘成民族新生的图腾。在时代的激流中，个体的觉醒与担当，始终与国家命运血肉相连。唯有每一个“我”都肩负起“我们”的责任，方能在沉默中孕育呐喊，于伤痕处见证重生，最终让微小的浪花汇聚成推动民族前行的磅礴浪潮。 （开头精彩！将三则材料化为一组排比，画面感极强，点出“我”与“我们”的关系，突出个体与国家的血肉相连，将“小我”之情化为民族感情。）

身处压迫的时代，我们想发声。老舍笔下的鼓书艺人方宝庆在面对受苦受难的孩子们时，“心里直翻腾，开不了口”，这正是个人心灵在历史巨大创伤面前最真实的状态。这“失语”不是怯懦或退缩，也不是无奈或麻痹，而是个体在无声中咀嚼伤痛、确认自身的隐秘过程。正如谭嗣同在变法失败后拒绝逃亡，以沉默的鲜血唤醒国民的反抗；又如京剧大师梅兰芳在抗战时期蓄须明志，坚守艺术家的气节。“乱世之音怨以怒”，这份喑哑，是文化血脉被强行截断时的无声挣扎。（使用“不是……，也不是……，而是……”这样的句式，在对比中让读者一层层深入鼓书艺人的内心，挖掘乱世中普通人的反抗，再以谭嗣同、梅兰芳两位名人的论例来表明，有些反抗纵然无声，也绝非顺从。）

身处抗争的时代，我们要发声。鲁迅曾激昂写下：“中国者，中国人之中国。可容外族之研究，不容外族之探险；可容外族之赞叹，不容外族之觊觎者也。”抗争的本质是在困境中凿出生路，在沉默中爆发惊雷。“正义是杀不完的，因为真理永远存在！”我们听到闻一多那掷地有声的宣讲，它划破了暗夜的阴霾。我们看到，萧红在《生死场》中描绘的东北农民、路翎在《财主底儿女们》中刻画的青年知识分子，都以各自的方式加入这场嘶哑的合唱，在荒芜的大地上，挑起时代的重任。发自生命深处的呐喊，唤醒了中华民族沉睡已久的民族精神。 （鲁迅的名言强烈地表达出了个人对国家和民族的深情，闻一多、萧红、路翎等文化名人以“各自的方式”表达自己的爱国情感，体现了中华民族的韧性与不屈精神。）

身处崛起的时代，我们能发声。一声出而万音和，高声歌唱不是对苦难的蔑视，而是对团结的自信和对永不停息的奋斗的向往。当我们牵起穆旦笔下那“带血的手”，就有可能跨越一路的颠沛流离，望见民族复兴的未来。试看，昔日“跳水皇后”郭晶晶担任巴黎奥运会跳水比赛裁判长，这一惊艳亮相的背后，是她深知要获得公平，国际裁判席上必须有中国声音；导演饺子坚持以中国传统文化为电影题材，终以《哪吒》系列电影书写了中国动画传奇，让世界电影市场看到中国文化的独特魅力。当战争的烽火成为历史，新时代的角逐已在科技革新、经济博弈与文化交融中全面展开。国家的独立自主与繁荣昌盛，正是我们每个中国人奋斗的意义所在。身处百年未有之大变局，吾辈青年更该发出自己的声音，使其汇聚成流，在新时代发出更为强劲的民族之声！ （从历史走向现实，作者以郭晶晶、饺子勇担责任为例，表明更多的战场已转到科技、经济和文化领域，赋予爱国主义更广阔的现实意义。）

那喑哑长歌已汇入民族血脉奔涌的江河，它从不曾真正停歇，而是在每一代人心中激荡回响。让我们以青春之手接过这“带血的拥抱”，让沉默中积蓄的伟力、喑哑中迸发的勇气，为中华文明生生不息的壮丽画卷续写不朽华章。赤子之喉发出的歌声回荡在中国的每一寸山河，那将是民族魂在永恒时空里最嘹亮的回响！（“喑哑长歌”“带血的拥抱”回扣材料，再次点题，上升到中华文明的层面，再次点明“民族魂”的主旨，从“喑哑长歌”到“嘹亮的回响”，表明“一个民族已经起来”，完美收束全文。）

思路清晰，结构严谨。本文以“声音”为灵魂意象，构建起一部荡气回肠的民族精神史诗。文章以“嘶哑”开头，以“想发声”“要发声”“能发声”分别对应“压迫”“抗争”“崛起”三个时代，将个体声带震颤升华为民族精神脉动，从“嘶哑的喉咙”到结尾“嘹亮的回响”，极具象征意义，尤其是“喑哑长歌已汇入民族血脉奔涌的江河”之喻，使抽象的家国情怀具象为可听可感的声浪洪流。
    """
    answer = agent.run(args.question)

    print(f"Got this answer: {answer}")


if __name__ == "__main__":
    main()
