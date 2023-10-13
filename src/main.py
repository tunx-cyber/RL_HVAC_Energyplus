import actor_critic
import Config
import env as Env
import replay_memory
# INPUT
input_param=[
    ["price", "out_temperature", "num_occupants"]
]

episodes = 100
cfg = Config.config()
n_state = 3
n_actions = 10
agents = [actor_critic.actor_critic(n_state, n_actions,cfg) for _ in range(cfg.N)]
damper_agent = actor_critic.actor_critic(n_state, n_actions,cfg)

# actions
air_supply_rate_set = [i*0.1*450 for i in range(11)]
damper_position_set = [i*0.1 for i in range(11)]

# buffer
buffer = replay_memory.replay_memory(48000)

# env
env = Env.env(cfg=cfg)
def train():
    for episode in range(episodes):
        # Reset environments, and get initial observation state oi,1 for each agent i
        state = env.reset()
        done = False
        total_reward = 0

        for t in range(cfg.L):
            '''
            TODO:
            select actions a(i,t) for each agent i
            a: action
            '''
            actions = []
            for i in range(cfg.N + 1):
                actions.append(agents[i].take_action(state))
            
            '''
            TODO:
            send actions a(i,t) to all parallel environments and get o(i,t+1) and r(i,t+1)
            a: action
            o: observation
            r: reward
            '''


            '''
            store transitions (o(t), a(t), o(t+1), r(t+1)) in D
            '''

            if True:# if G(memeory)>B(size) and mod(t,T(update))

                '''
                TODO:
                sapmle mini-batch B with B(size) transitions from D
                '''

                '''
                TODO:
                calculate Q
                calculate a
                calculate Q
                update critic network
                calculate a
                calculate Q
                update policues
                update the weights of target network
                '''






def execute(actor_weight):
    '''
    return action
    '''
    
    # 1 All agents receive initial local observation
    for t in range(cfg.L):
        # Each agent i selects its action ai,t in parallel according to the learned policy at the beginning of slot t;
        # Each agent i takes action ai,t in parallel, which affects the operation the HVAC system;
        # Each agent i receives new observation
        pass

