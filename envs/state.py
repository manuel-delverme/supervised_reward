class State(object):
    def __init__(self, state_hash, state_info):
        self.state_hash = state_hash
        self.state_info = state_info

    def __int__(self):
        return self.state_hash

    def __repr__(self):
        return "State({}): {}".format(self.state_hash, repr(self.state_info))

