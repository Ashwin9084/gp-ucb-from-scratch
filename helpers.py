import numpy as np
import matplotlib.pyplot as plt

def report(solver, name):
    print(f"{solver.counts}, The Slot Machine is : {np.argmax(solver.counts)}")
    print(f"{np.round(solver.bandit.f_values, 3)}, The Real Best Slot Machine is : {np.argmax(solver.bandit.f_values)}")
    print(f"{name} Method Done")


def plot_results(solvers, solver_names=None, filename="results.png"):
    """
    Plot four views of how bandit solvers performed.

    1st: cumulative regret over time (lower is better).
    2nd: each solver's estimated value vs the true one.
    3rd: the fraction of all pulls that went to each arm.
    4th: how far the solver's recommended point is from the best point, over time.

    Args:
        solvers: list of solvers that have already been run (same bandit).
        solver_names: list of labels, one per solver.
        filename: where to save the figure.
    """
    if solver_names is None:
        solver_names = [type(solver).__name__ for solver in solvers]   # e.g. "UCB1", "GPUCB"
    assert len(solver_names) == len(solvers), "need exactly one name per solver"

    bandit = solvers[0].bandit
    n_arms = bandit.n
    true_probs = np.array(bandit.f_values)

    # Order the arms from worst to best, so the middle plot reads left to right.
    arm_order = np.argsort(true_probs)
    arm_positions = np.arange(n_arms)

    fig, (ax_regret, ax_estimates, ax_counts, ax_gap) = plt.subplots(
        1, 4, figsize=(20, 4.5), constrained_layout=True
    )

    # 1) Cumulative regret over time.
    for solver, name in zip(solvers, solver_names):
        ax_regret.plot(solver.regrets, label=name)
    ax_regret.set_xlabel("Time step")
    ax_regret.set_ylabel("Cumulative regret")
    ax_regret.set_title("Regret over time")

    # 2) Estimated vs true probability, arms sorted by true probability.
    ax_estimates.plot(arm_positions, true_probs[arm_order],
                      color="black", linestyle="--", label="True probability")
    for solver, name in zip(solvers, solver_names):
        estimates = np.array(solver.estimates)
        ax_estimates.plot(arm_positions, estimates[arm_order],
                          marker="x", linestyle="none", markeredgewidth=2, label=name)
    ax_estimates.set_xlabel("Arms, sorted by true probability")
    ax_estimates.set_ylabel("Estimated probability")
    ax_estimates.set_title("What each solver believes")

    # 3) Fraction of all pulls that went to each arm (in the original arm order).
    for solver, name in zip(solvers, solver_names):
        fraction_of_pulls = np.array(solver.counts) / len(solver.actions)
        ax_counts.plot(arm_positions, fraction_of_pulls,
                       drawstyle="steps-mid", linewidth=2, label=name)
    ax_counts.set_xlabel("Arm index")
    ax_counts.set_ylabel("Fraction of pulls")
    ax_counts.set_title("Where the pulls went")

    # 4) Gap between the best value and the value at the solver's recommended point.
    #    0 means the solver would name the best point if asked right now.
    for solver, name in zip(solvers, solver_names):
        recommended_values = bandit.f_values[solver.recommendations]
        gap = bandit.f_star - recommended_values
        ax_gap.plot(gap, label=name)
    ax_gap.set_yscale("symlog", linthresh=1e-2)   # the gap can be exactly 0, so not a plain log axis
    ax_gap.set_xlabel("Time step")
    ax_gap.set_ylabel("Gap to the best value")
    ax_gap.set_title("How good is the recommended point")

    for ax in (ax_regret, ax_estimates, ax_counts, ax_gap):
        ax.grid(True, linestyle="--", alpha=0.3)

    # One shared legend below the four plots (the solvers plus the true-probability line).
    handles, labels = ax_estimates.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))

    fig.savefig(filename, dpi=150)
    plt.show()
