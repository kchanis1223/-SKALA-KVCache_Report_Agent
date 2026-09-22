from skala_agent.schemas import PERSPECTIVES


def run(state):
    # TODO(이준형): 설계서 4-7의 다섯 질문으로 상충 및 trade-off 추출.
    return {"synthesis": [item for key in PERSPECTIVES for item in state["analyses"][key]]}
