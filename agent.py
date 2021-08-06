from proprocess import generate_dataset
from transformers import TFAutoModelForSequenceClassification

def init_agent(name : str, path : str, num_labels : int, **kwargs) -> dict:
    agent = {}
    agent['model'] = TFAutoModelForSequenceClassification(name, num_labels=num_labels)
    agent['model'].compile(**kwargs['agent_config'])
    agent['dataset'] = generate_dataset(path, kwargs['dataset_config'])
    agent['y'] = agent['dataset'].pop('labels')
    agent['x'] = agent.pop('dataset')
    return agent

def run(mode : str, agent : dict, **kwargs) -> dict:
    model = agent.pop('model')
    kwargs = {**agent, **kwargs}
    model.fit(**kwargs) if mode=="train" else model.evaluate(**kwargs)
    agent['model'] = model
    return agent
