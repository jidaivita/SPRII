"""Resolve multistart optimizer outcomes without rewarding numerical cost ties."""

def select_solution(solutions):
    if not solutions:raise ValueError('no optimizer solutions')
    best=min(solutions,key=lambda fit:fit.cost)
    converged=[fit for fit in solutions if fit.success]
    if not converged:return best
    best_converged=min(converged,key=lambda fit:fit.cost)
    # Both tolerances are in the pixel-normalized least-squares objective and
    # fixed before formal training. A materially better failed run remains a
    # failed run; an insignificant numerical improvement does not erase a
    # converged solution at the same minimum.
    tolerance=max(1e-8,1e-7*abs(best.cost))
    return best_converged if best_converged.cost<=best.cost+tolerance else best
