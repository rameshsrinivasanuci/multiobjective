using JuMP
using NPZ
import HiGHS
import MultiObjectiveAlgorithms as MOA

data = npzread(joinpath(@__DIR__, "problems", "problem_0.npz"))["arr_0"]
capacity = 55 #60
n_selected = 5 #6
N = size(data, 1)

obj1 = data[:, 1]
obj2 = data[:, 2]
cost = data[:, 3]
# obj3 = data[:, 3]
# cost = data[:, 4]

model = Model(() -> MOA.Optimizer(HiGHS.Optimizer))
set_attribute(model, MOA.Algorithm(), MOA.KirlikSayin())
set_silent(model)

@variable(model, x[1:N], Bin)
@constraint(model, sum(cost[i] * x[i] for i in 1:N) <= capacity)
@constraint(model, sum(x) == n_selected)
@expression(model, obj1_expr, sum(obj1[i] * x[i] for i in 1:N))
@expression(model, obj2_expr, sum(obj2[i] * x[i] for i in 1:N))
# @expression(model, obj3_expr, sum(obj3[i] * x[i] for i in 1:N))
@objective(model, Max, [obj1_expr, obj2_expr]) #, obj3_expr])


optimize!(model)
assert_is_solved_and_feasible(model)
solution_summary(model)

K = result_count(model)
pareto_front = reduce(vcat, (objective_value(model; result = i)' for i in 1:K))  # K×3
selections = [findall(v -> v > 0.5, value.(x; result = i)) for i in 1:K]         # chosen item indices
for i in 1:K
    println(i, ": objectives = ", pareto_front[i, :], ", items = ", selections[i])
end
npzwrite(joinpath(@__DIR__, "problems", "pf_1.npz"), Dict("pf" => pareto_front))