import RuleBased

class FixBased(RuleBased.RuleBased):
    def __init__(self) -> None:
        super().__init__()
    
    def get_action(self):
        action_val = [22 for i in range(10)]      
        return action_val  
    
fix_test = FixBased()
fix_test.run()
fix_test.ep.energyplus_exec_thread.join()
fix_test.env
print("fix point result: ", fix_test.get_result())