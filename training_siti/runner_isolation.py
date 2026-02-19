import agentlightning as agl
from siti_agent import LitSitiAgent
import pyarrow.parquet as pq
from typing import cast, List, Dict, Any
import asyncio
agl.setup_logging("DEBUG")




async def runner():
    """Funzione per testare il runner di Agent lighting in isolamento"""
    tracer = agl.AgentOpsTracer()
    runner = agl.LitAgentRunner(tracer)
    store = agl.LightningStoreClient("http://127.0.0.1:4747")

    table = pq.read_table("./training_siti/data/train.parquet")
    df = cast(List[Dict[str, Any]], table.to_pylist()[:10])
    print(f"running rollout with input task: {df[0]}")
    with runner.run_context(agent=LitSitiAgent(), store=store):
        rollout = await runner.step(
            df[5],
            resources={
            "main_llm": agl.LLM(
                endpoint="http://127.0.0.1:8000",
                model="unsloth/granite-4.0-micro-unsloth-bnb-4bit",
                sampling_parameters={"temperature": 0.7}    
            )
        },
        )

        # query the store
        spans = await store.query_spans(rollout.rollout_id)
        """ adapter = TraceToDictAdapter()
        result = adapter.adapt(spans=spans)
        print(result) """
        adapter = agl.TracerTraceToTriplet(agent_match="chat", match_w_itself=True)

        # convert span in trajectory
        #adapter.visualize(spans)
        trajectory = adapter.adapt(spans)

        print(f"number of trajectory: {len(trajectory)}")
        for i,t in enumerate(trajectory):
            print(f"dispaly triplet number {i}")
            print(f"[{i}] input: {t.prompt}")
            print(f"[{i}]response: {t.response}")
            print(f"[{i}]reward: {t.reward}")
            print('-'*50)

        
if __name__ == '__main__':
    asyncio.run(runner())