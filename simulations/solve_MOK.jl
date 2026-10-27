using JuMP
using NPZ
import HiGHS
import MultiObjectiveAlgorithms as MOA

n_obj = 3
n_items = 50
n_selected = 12
rho = 0.3
data = npzread(joinpath(@__DIR__, "problems", "items_$(n_obj)obj_$(n_items)items_$(rho)rho_allneg.npz"))
items = data["items"]
capacity = data["capacity"]
N = size(items, 1)

obj1 = items[:, 1]
obj2 = items[:, 2]
obj3 = items[:, 3]
cost = items[:, n_obj + 1]

model = Model(() -> MOA.Optimizer(HiGHS.Optimizer))
set_attribute(model, MOA.Algorithm(), MOA.TambyVanderpooten())
# set_silent(model)

@variable(model, x[1:N], Bin)
@constraint(model, sum(cost[i] * x[i] for i in 1:N) <= capacity)
@constraint(model, sum(x) <= n_selected)
@expression(model, obj1_expr, sum(obj1[i] * x[i] for i in 1:N))
@expression(model, obj2_expr, sum(obj2[i] * x[i] for i in 1:N))
@expression(model, obj3_expr, sum(obj3[i] * x[i] for i in 1:N))
@objective(model, Max, [obj1_expr, obj2_expr, obj3_expr])


optimize!(model)
assert_is_solved_and_feasible(model)
solution_summary(model)

K = result_count(model)
pareto_front = reduce(vcat, (objective_value(model; result = i)' for i in 1:K))  # K×3
selections = [findall(v -> v > 0.5, value.(x; result = i)) for i in 1:K]         # chosen item indices
for i in 1:K
    println(i, ": objectives = ", pareto_front[i, :], ", items = ", selections[i])
end
npzwrite(joinpath(@__DIR__, "results", "upper_bound", "TV_$(n_obj)obj_$(n_items)items_$(rho)rho_allneg.npz"), Dict("pf" => pareto_front))