from src.OrienteeringScenarios.SolutionPermutation import *
from src.OrienteeringScenarios.TSP_OP_solution import *
from src.OrienteeringScenarios.ScenarioGeneration import *
from src.OrienteeringScenarios.AverageScenarios import *
from tqdm.auto import tqdm

class SampleGeneration():

    def __init__(self, graph,permutation,deadline_mode="mst_based", deadline_coeff=0.6):
        self.graph = graph 
        self.tsp = TSP_OP(self.graph)
        if deadline_mode == "mst_based" or deadline_mode is None:
            self.tsp_opt = self.tsp.TSPOptimal() 
            self.deadline_coeff = deadline_coeff
            self.deadline = self.tsp_opt['route_time']*deadline_coeff
            
        else:
            raise SampleGeneration_Exception__DeadlineMode(deadline_mode)
        if isinstance(permutation, SolutionPermutation):
            self.permutation = permutation
        else:
            raise SampleGeneration_Exception__SolutionPermutationInstance(permutation)


        self.solution_memory = dict()
        self.solution_df = pd.DataFrame(columns=["key",'solution','time_values','time_mean','time_variance','prize_values','prize_mean','prize_variance','deadline','deadline_coeff','count_availability','count_visited'])


    def getIstanceGenrate(self,mode="dict"):
        if mode=="dict":
            return self.solution_memory
        elif mode=="pandas":
            return self.solution_df
        else:
            return None

    def makeSample(self,solutions_n, scenarios_n, save_onfile=False,save_filename="scenario.pkl",save_scenarios="None"):
        for k in tqdm(range(solutions_n)):
        #for k in range(solutions_n): 

            _permutation = self.permutation.next()
            s = ScenarioGeneration(_permutation,self.graph)    

            scs = s.scenarios_generation(scenarios_n)
            solutions = dict()
            setScenarios = AverageScenarios(self.graph)
            for i in range(scenarios_n):
                solutions[i] = self.tsp.OPSolver(scs[i],self.deadline,back2ware=False)
                solutions[i].idSolutionId = i
                if save_scenarios=="all":
                    setScenarios.addScenario(solutions[i],save_scenario=True)
                elif save_scenarios=="last" and k==solutions_n-1:
                    setScenarios.addScenario(solutions[i],save_scenario=True)
                elif save_scenarios=="first" and k==0:
                    setScenarios.addScenario(solutions[i],save_scenario=True)
                else:
                    setScenarios.addScenario(solutions[i],save_scenario=False)
            self.solution_memory[k] = {"permutation_temp":_permutation,"scenarios":setScenarios}
            timeStats = setScenarios.getStats("t")
            prizeStats = setScenarios.getStats("p")
            solution_summary = {'key': k,'solution':np.array(_permutation),
                                'time_values':timeStats['values'],'time_mean':timeStats['mean'],'time_variance':timeStats['variance'],
                                'prize_values':prizeStats['values'],'prize_mean':prizeStats['mean'],'prize_variance':prizeStats['variance'],
                                'deadline':self.deadline,'deadline_coeff':self.deadline_coeff,'count_availability':setScenarios.count_availability,
                                'count_visited':setScenarios.count_visited}
            self.solution_df = pd.concat([self.solution_df, pd.DataFrame.from_records([solution_summary])])
            if save_onfile:
                self.solution_df.to_pickle(save_filename)

class SampleGeneration_Exception__DeadlineMode(Exception):
      """Exception raised for error no training modality recognized"""

      def __init__(self,value):
          self.value = None

      def __str__(self):
          return f"{self.value} is not recognized."

class SampleGeneration_Exception__SolutionPermutationInstance(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,instance):
        self.instance = instance
          
    def __str__(self):
        return f"SampleGeneration module require an instance 'SolutionPermutation' but receive an '{str(type(self.instance_type))}' object."