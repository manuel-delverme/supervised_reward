import config
import random
import tqdm

env = config.environment()

for _ in tqdm.tqdm(range(100000)):
    if random.random() < 0.001:
        env.reset()

    env.step(random.choice(range(5)))
