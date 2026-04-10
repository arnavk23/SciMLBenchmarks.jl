const ISSUE_935_SKIPLIST = Set([
    "avgasb", "bloweya", "chardis1", "cleuven4", "cmpc3", "cmpc10", "cvxqp2",
    "cvxqp3", "dittert", "expfita", "haifal", "hanging", "hier13", "himmelp2",
    "hs67", "hs85", "hs101", "liswet9", "lukvle8", "lukvli7", "lukvli13",
    "mpc2", "mss1", "ninenew", "patternne", "reading2", "reading6", "s268",
    "sosqp2", "stcqp1", "synthes1",
])

filter_problematic(problems; skiplist = ISSUE_935_SKIPLIST) =
    filter(p -> !(lowercase(p) in skiplist), problems)

function get_stats(sol, optimizer_name)
    solve_time = try
        hasfield(typeof(sol), :stats) && hasfield(typeof(sol.stats), :time) ? getfield(sol.stats, :time) : NaN
    catch
        NaN
    end
    return (length(sol.u), solve_time, optimizer_name, Symbol(sol.retcode))
end

function run_benchmarks(problems, optimizers; chunk_size = 1, max_nvar = 10000, maxiters = 1000,
    maxtime = 30.0)
    problem = String[]
    n_vars = Int64[]
    secs = Float64[]
    solver = String[]
    retcode = Symbol[]
    optz = length(optimizers)
    n = length(problems)
    @info "Processing $(n) problems with $(optz) optimizers in chunks of $(chunk_size)"
    broadcast(c -> sizehint!(c, optz * n), [problem, n_vars, secs, solver, retcode])
    for chunk_start in 1:chunk_size:n
        chunk_end = min(chunk_start + chunk_size - 1, n)
        chunk_problems = problems[chunk_start:chunk_end]
        @info "Processing chunk $(div(chunk_start - 1, chunk_size) + 1)/$(div(n - 1, chunk_size) + 1): problems $(chunk_start)-$(chunk_end)"
        for (idx, prob_name) in enumerate(chunk_problems)
            current_problem = chunk_start + idx - 1
            @info "Problem $(current_problem)/$(n): $(prob_name)"
            nlp_prob = nothing
            try
                nlp_prob = CUTEstModel(prob_name)
                if nlp_prob.meta.nvar > max_nvar
                    @info "  Skipping $(prob_name) (too large: $(nlp_prob.meta.nvar) variables)"
                    finalize(nlp_prob)
                    continue
                end
                prob = OptimizationNLPModels.OptimizationProblem(nlp_prob, Optimization.AutoFiniteDiff())
                for (optimizer_name, optimizer) in optimizers
                    try
                        sol = solve(prob, optimizer; maxiters = maxiters, maxtime = maxtime)
                        @info "✓ Solved $(prob_name) with $(optimizer_name) - Status: $(sol.retcode)"
                        vars, time, alg, code = get_stats(sol, optimizer_name)
                        push!(problem, prob_name)
                        push!(n_vars, vars)
                        push!(secs, time)
                        push!(solver, alg)
                        push!(retcode, code)
                    catch
                        push!(problem, prob_name)
                        push!(n_vars, nlp_prob !== nothing ? nlp_prob.meta.nvar : -1)
                        push!(secs, NaN)
                        push!(solver, optimizer_name)
                        push!(retcode, :FAILED)
                    end
                end
            catch
                for (optimizer_name, optimizer) in optimizers
                    push!(problem, prob_name)
                    push!(n_vars, -1)
                    push!(secs, NaN)
                    push!(solver, optimizer_name)
                    push!(retcode, :LOAD_FAILED)
                end
            finally
                if nlp_prob !== nothing
                    try
                        finalize(nlp_prob)
                    catch
                    end
                end
            end
        end
        GC.gc()
        @info "Completed chunk, memory usage cleaned up"
    end
    return DataFrame(problem = problem, n_vars = n_vars, secs = secs, solver = solver,
        retcode = retcode)
end