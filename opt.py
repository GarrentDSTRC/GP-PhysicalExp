# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
import torch
import gpytorch
from pymoo.core.population import Population
global model,likelihood
from GPy import UPB,LOWB

def optIGD(mymodel,mylikelihood,num_task=-2,testmode="test_WFG",train_x=[]):
    global model, likelihood
    model=mymodel
    likelihood=mylikelihood
    model.eval()
    likelihood.eval()
    # # Create an instance of the problem with the observed predictions
    if testmode == "CFD":
        problem = MyProblem(num_task,testmode,constr=1,n_var=train_x.shape[1])
    else:
        problem = MyProblem(num_task, testmode, constr=0,n_var=train_x.shape[1])
    # # Define the reference directions for the Pareto front
    from pymoo.util.ref_dirs import get_reference_directions
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)
    pop = Population.new("X", np.concatenate((train_x[:70, :].numpy(), train_x[-230:, :].numpy())))
    # Create an instance of the NSGA-II algorithm
    algorithm = NSGA2(pop_size=400, eliminate_duplicates=True, sampling=pop)
    # Minimize the problem using the algorithm
    res = minimize(problem,
                   algorithm,
                   ("n_gen", 40),#70
                   seed=70,
                   verbose=True)


    ######################IGD spread
    # 导入pymoo模块
    from pymoo.indicators.igd import IGD
    from pymoo.problems import get_problem

    # 创建一个DTLZ1问题的实例，指定变量数和目标数
    if testmode=="test_WFG" :
        problem = get_problem("wfg1", n_var=7, n_obj=2)
        plf = -problem.pareto_front()
    else:
        problem = get_problem("dtlz1", n_var=7, n_obj=2)
        #problem = get_problem("zdt1", n_var=7)
        plf = problem.pareto_front()
        plf = -np.power(plf, 1 / 3)

    plt.clf()
    plt.scatter(-1*res.F[:, 0], -1*res.F[:, 1], c="blue", marker="o", s=20) # Make the markers smaller
    #plt.scatter(plf[:, 0], plf[:, 1], c="red", marker="o", s=10)  # Make the markers smaller
    plt.xlabel("ct ", fontsize=22, fontfamily='Times New Roman') # Increase the font size and use Times New Roman font
    plt.ylabel("cl ", fontsize=22, rotation=0, fontfamily='Times New Roman') # Rotate the y-axis label to be vertical
    plt.rc('font', family='Times New Roman') # Set the font family for all text elements
    #plt.xlim(-1, 4)  # 固定横轴范围为-1到4
    #plt.ylim(-1, 3)  # 固定纵轴范围为-1到3
    # Add a legend with larger font size
    #plt.legend(["Designs"], loc="upper right", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    # Save the plot as a high-resolution PDF file
    plt.subplots_adjust(bottom=0.2)
    plt.savefig("pareto.pdf", dpi=300)
    # Show the plot on screen
    #plt.show()
    # Import the pandas library for data manipulation
    import pandas as pd
    # Get the Pareto front solutions from the result object
    pf = res.opt
    # Get the decision variables and objective values of the Pareto front solutions
    X = pf.get("X")
    F = -pf.get("F")
    # Create a pandas dataframe with the decision variables and objective values
    df = pd.DataFrame(np.hstack([X, F]))
    # Save the dataframe to a csv file
    df.to_csv("pareto_front.csv", index=False)

    # 创建一个解集，这里假设是由某个算法得到的
    A = -1*res.F

    # 创建IGD和spread指标的实例，传入真实帕累托前沿作为参考集
    igd = IGD(plf)

    # 计算解集A的IGD和spread值
    igd_value = igd(A)
    # 打印结果
    print("IGD:", igd_value)

    final_population_X = res.pop.get("X")
    final_population_X_tensor = torch.tensor(final_population_X, dtype=torch.float32)

    # ... The rest of the code remains the same...

    return igd_value, final_population_X


# Define a custom problem class that takes the observed predictions as objectives
class MyProblem(Problem):

    def __init__(self,num_task=-2,testmode=[],constr=[],n_var=1):
        super().__init__(n_var=n_var,
                         n_obj=abs(num_task),
                         n_constr=constr,
                         xu=np.array([1]*n_var),
                         xl=np.array([0]*n_var))
        self.num_task=num_task
        self.testmode=testmode


    def _evaluate(self, x, out, *args, **kwargs):
        # Use the observed predictions as objectives
        test_x = torch.tensor(x).to(torch.float32)
        test_i_task2 = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=1)
        with torch.no_grad(), gpytorch.settings.fast_pred_var(),gpytorch.settings.cholesky_jitter(1e-0):
            if self.num_task==-2:
                observed_pred_yH = likelihood(*model((test_x, test_i_task2), (test_x, test_i_task2)))
            else:
                observed_pred_yH = likelihood(*model(test_x, test_x))
            observed_pred_yHC = -1*np.array(observed_pred_yH[0].mean.tolist())  # ct high
            observed_pred_yHE = -1*np.array(observed_pred_yH[1].mean.tolist())  # eta high

        N=np.array([observed_pred_yHC,observed_pred_yHE]).T
        out["F"] =N
        if self.testmode=="CFD":
            N1=-N[:, 1]-0.97
            out["G"] = [N[:, 0]]         #.reshape(-1, 3, 1)