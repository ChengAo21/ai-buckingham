"""Exact Buckingham null-space construction."""
from __future__ import annotations
from .shared import *


@tool
def calc_basis_tool(
    D_in: list[list[float]],
    var_names: list[str],
) -> dict:
    """Compute a full-precision null-space basis for Buckingham Pi groups.

    Numerical coefficients returned in ``basis_matrices`` are not rounded.
    Rounding is used only in display-oriented formula strings.
    """
    try:
        if not D_in or not D_in[0]:
            return {
                "status": "error",
                "message": "D_in must be a non-empty 2D matrix.",
            }
        if any(len(row) != len(D_in[0]) for row in D_in):
            return {
                "status": "error",
                "message": "All rows of D_in must have the same length.",
            }
        if len(D_in[0]) != len(var_names):
            return {
                "status": "error",
                "message": (
                    "The number of columns in D_in must match the "
                    "number of variables in var_names."
                ),
            }

        # nsimplify prevents decimal inputs such as 1.0 from unnecessarily
        # degrading the exact null-space calculation.
        M = sympy.Matrix([
            [sympy.nsimplify(value) for value in row]
            for row in D_in
        ])
        null_space_vectors = M.nullspace()
        if not null_space_vectors:
            return {
                "status": "error",
                "message": (
                    "The dimensional matrix has a trivial null space; "
                    "no independent Pi group exists."
                ),
            }

        pi_formulas = {}
        basis_matrices = []
        display_basis_matrices = []

        def format_display_exponent(value):
            simplified = sympy.simplify(value)
            if simplified == 1:
                return ""
            if simplified.is_Integer:
                return str(int(simplified))
            if simplified.is_Rational:
                return (
                    f"{int(simplified.p)}/{int(simplified.q)}"
                )
            return f"{float(simplified.evalf(12)):.8g}"

        for i, vec in enumerate(null_space_vectors):
            full_precision_vector = [
                float(value.evalf(17))
                for value in vec
            ]
            basis_matrices.append(full_precision_vector)
            display_basis_matrices.append([
                round(value, 6)
                for value in full_precision_vector
            ])

            numerator = []
            denominator = []
            for var, exact_exp in zip(var_names, vec):
                exact_exp = sympy.simplify(exact_exp)
                if exact_exp == 0:
                    continue

                positive_exp = abs(exact_exp)
                exponent_text = format_display_exponent(
                    positive_exp
                )
                item = (
                    var
                    if exponent_text == ""
                    else f"{var}^{{{exponent_text}}}"
                )
                if exact_exp > 0:
                    numerator.append(item)
                else:
                    denominator.append(item)

            numerator_text = (
                " * ".join(numerator)
                if numerator
                else "1"
            )
            denominator_text = " * ".join(denominator)
            formula = (
                f"({numerator_text}) / ({denominator_text})"
                if denominator_text
                else numerator_text
            )
            pi_formulas[f"Pi_{i + 1}"] = formula

        B = np.asarray(basis_matrices, dtype=float)
        D_numeric = np.asarray(D_in, dtype=float)
        null_space_residual = float(
            np.linalg.norm(D_numeric @ B.T, ord=np.inf)
        )
        if null_space_residual > 1e-9:
            return {
                "status": "error",
                "message": (
                    "Computed Pi basis failed the dimensional-null-space "
                    f"check: residual={null_space_residual:.3e}."
                ),
            }

        return {
            "status": "success",
            "pi_formulas": pi_formulas,
            "basis_matrices": basis_matrices,
            "basis_matrices_display": display_basis_matrices,
            "null_space_residual_inf": null_space_residual,
            "basis_precision_note": (
                "basis_matrices stores full floating-point precision; "
                "basis_matrices_display is rounded for display only."
            ),
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": (
                "Error computing null space: "
                f"{str(exc)}"
            ),
        }
