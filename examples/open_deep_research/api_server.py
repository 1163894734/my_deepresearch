import os
import uuid
import time
import logging
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict

# 1. 任务状态存储中心（内存字典）
# 生产环境中如果有多台服务器，建议将其替换为 Redis 或数据库
tasks_db: Dict[str, dict] = {}

app = FastAPI(title="Deep Research API Server")

# 2. 定义接口接收的 JSON 参数格式
class TaskRequest(BaseModel):
    task_type: str  # 可选: "deepresearch" 或 "test"
    target_topic: Optional[str] = "Automated Scientific Discovery with Large Language Models"
    search_tasks: Optional[str] = None  # 用于 run_test.py 的指令段

# 3. 自定义日志处理器：将 Agent 运行时的日志精准推送到对应任务的缓存中
class TaskLogHandler(logging.Handler):
    def __init__(self, task_id: str):
        super().__init__()
        self.task_id = task_id
        # 设置日志格式
        self.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))

    def emit(self, record):
        if self.task_id in tasks_db:
            msg = self.format(record)
            tasks_db[self.task_id]["logs"].append(msg)

# 4. 后台任务执行包装器
def execute_agent_task(task_id: str, request_data: TaskRequest):
    # 初始化运行目录
    time_str = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(os.getcwd(), "outputs", f"task_{request_data.task_type}_{time_str}")
    os.makedirs(run_dir, exist_ok=True)
    
    tasks_db[task_id]["run_dir"] = run_dir

    # 配置专属 Logger
    task_logger = logging.getLogger(f"agent_logger_{task_id}")
    task_logger.setLevel(logging.INFO)
    # 清除旧的 handler 避免重复打印
    if task_logger.hasHandlers():
        task_logger.handlers.clear()
    
    # 将日志发送到任务 DB
    memory_handler = TaskLogHandler(task_id)
    task_logger.addHandler(memory_handler)
    # 同时将日志保存到文件，方便事后排查
    file_handler = logging.FileHandler(os.path.join(run_dir, "run.log"), encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    task_logger.addHandler(file_handler)

    try:
        task_logger.info(f"🚀 任务 {task_id} 开始执行，类型: {request_data.task_type}")
        
        # 动态路由到对应的业务逻辑
        if request_data.task_type == "deepresearch":
            # 导入你改造后的业务函数
            from run_deepresearch import run_deepresearch_core
            result = run_deepresearch_core(
                target_topic=request_data.target_topic, 
                run_dir=run_dir, 
                logger=task_logger
            )
            tasks_db[task_id]["result"] = result
            
        elif request_data.task_type == "test":
            from examples.open_deep_research.run_frontier import run_test_core
            result = run_test_core(
                search_tasks=request_data.search_tasks, 
                run_dir=run_dir, 
                logger=task_logger
            )
            tasks_db[task_id]["result"] = result
            
        else:
            raise ValueError("未知的 task_type")

        tasks_db[task_id]["status"] = "completed"
        task_logger.info("✅ 任务执行完毕！")

    except Exception as e:
        task_logger.error(f"❌ 任务执行失败: {str(e)}", exc_info=True)
        tasks_db[task_id]["status"] = "failed"
        tasks_db[task_id]["error"] = str(e)


# ================== API 路由 ==================

@app.post("/api/research/start")
async def start_research(request: TaskRequest, background_tasks: BackgroundTasks):
    """
    提交任务接口：接收 JSON 参数，生成 Task ID，将任务推入后台，立即返回。
    """
    task_id = str(uuid.uuid4())
    
    # 初始化任务状态
    tasks_db[task_id] = {
        "status": "running",
        "task_type": request.task_type,
        "logs": [],
        "result": None,
        "error": None,
        "created_at": time.time()
    }
    
    # 丢入 FastAPI 后台线程池执行
    background_tasks.add_task(execute_agent_task, task_id, request)
    
    return {"message": "任务已受理", "task_id": task_id}


@app.get("/api/research/status/{task_id}")
async def get_task_status(task_id: str):
    """
    查询状态接口：前端通过 Task ID 轮询此接口，获取最新日志和最终结果。
    """
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="任务 ID 不存在")
    
    return tasks_db[task_id]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)