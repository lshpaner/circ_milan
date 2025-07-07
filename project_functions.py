import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

import plotly.graph_objects as go


class HealthMetrics:
    """
    A class to calculate health-related metrics such as Body Mass Index (BMI)
    and Mean Arterial Pressure (MAP) for individuals in a DataFrame.
    """

    def __init__(self):
        pass  # Not storing state in the instance.

    @staticmethod
    def split_bp_column(
        df,
        combined_bp_col,
        systolic_col_name="SBP",
        diastolic_col_name="DBP",
    ):
        """
        Split a combined blood pressure column in "Systolic/Diastolic" format
        into separate systolic (SBP) and diastolic (DBP) columns.

        Parameters:
            df (DataFrame): The pandas DataFrame containing the combined column.
            combined_bp_col (str): Column name of the combined BP values.
            systolic_col_name (str): Column name for the new systolic BP values.
            diastolic_col_name (str): Column name for the new diastolic BP values.

        Returns:
            None: The DataFrame is updated in place with the new columns.
        """
        # Split the combined BP column into two new columns
        df[[systolic_col_name, diastolic_col_name]] = df[combined_bp_col].str.split(
            "/", expand=True
        )

        # Convert the new columns to numeric types
        df[systolic_col_name] = pd.to_numeric(
            df[systolic_col_name],
            errors="coerce",
        )
        df[diastolic_col_name] = pd.to_numeric(
            df[diastolic_col_name],
            errors="coerce",
        )

    @staticmethod
    def calculate_bmi_category(
        df,
        bmi_col="BMI",
        bmi_category_col="BMI_Category",
    ):
        """
        Categorize BMI based on the WHO BMI classifications and add the
        categories as a new column.

        Parameters:
            df (DataFrame): DataFrame containing the BMI data.
            bmi_col (str): Column name containing BMI values. Default is "BMI".
            bmi_category_col (str): Column name for BMI categories.
            Default is "BMI_Category".
        """

        def classify_bmi(bmi):
            if bmi < 18.5:
                return "Underweight"
            elif 18.5 <= bmi < 24.9:
                return "Normal_Weight"
            elif 25 <= bmi < 29.9:
                return "Overweight"
            else:
                return "Obese"

        # Apply the classification function to the BMI column
        df[bmi_category_col] = df[bmi_col].apply(classify_bmi)

    @staticmethod
    def calculate_bmi(
        df,
        weight_col,
        height_col,
        weight_unit="kg",
        height_unit="m",
    ):
        """
        Calculate Body Mass Index (BMI) and update the DataFrame with a new
        BMI column.

        Parameters:
            df (DataFrame): DataFrame containing the data.
            weight_col (str): Column name for weight.
            height_col (str): Column name for height.
            weight_unit (str): Unit of weight, default is "kg".
            height_unit (str): Unit of height, default is "m".
        """
        # Ensure weight is in kilograms
        if weight_unit == "lbs":
            df[weight_col] = df[weight_col] * 0.45359237
        # Ensure height is in meters
        if height_unit == "in":
            df[height_col] = df[height_col] * 0.0254
        elif height_unit == "cm":
            df[height_col] = df[height_col] / 100

        # Calculate BMI and update the DataFrame
        df["BMI"] = round(df[weight_col] / (df[height_col] ** 2), 2)

    @staticmethod
    def calculate_map(
        df,
        map_col_name="MAP",
        systolic_col=None,
        diastolic_col=None,
        combined_bp_col=None,
    ):
        """
        Calculate Mean Arterial Pressure (MAP) and update the DataFrame with a
        new MAP column.

        This method can operate based on separate systolic and diastolic columns,
        or a single combined column in the "Systolic/Diastolic" format. At
        least one of the column parameter sets must be provided.

        Parameters:
            df (DataFrame): The pandas DataFrame to calculate MAP for.
            map_col_name (str): Column name where MAP values will be stored.
            systolic_col (str, optional): Column name for systolic blood pressure.
            diastolic_col (str, optional): Column name for diastolic blood pressure.
            combined_bp_col (str, optional): Column name for combined
            "Systolic/Diastolic" blood pressure readings.
        """
        if systolic_col and diastolic_col:
            # Calculate MAP using separate systolic and diastolic columns
            systolic = df[systolic_col]
            diastolic = df[diastolic_col]
        elif combined_bp_col:
            # Calculate MAP using a combined BP column
            split_bp = df[combined_bp_col].str.split("/", expand=True).astype(float)
            systolic, diastolic = split_bp[0], split_bp[1]
        else:
            raise ValueError(
                "Must provide either systolic_col and diastolic_col, or combined_bp_col"
            )

        # MAP calculation formula: diastolic + (systolic - diastolic) / 3
        df[map_col_name] = round(diastolic + (systolic - diastolic) / 3, 2)


################################################################################


def plot_svm_decision_boundary_2d(
    X,
    y,
    feature_pair=("feature_1", "feature_2"),
    figsize=(8, 6),
    grid_density=100,
    C=100,
    gamma="auto",
    kernel="rbf",
    title=None,
    image_path_svg=None,
    image_path_png=None,
    margin=False,
):
    # Step 1: Extract just the 2 features
    X_pair = X[list(feature_pair)].copy()

    # Step 2: Encode categorical features if needed
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                [col for col in feature_pair if pd.api.types.is_numeric_dtype(X[col])],
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "label_enc",
                            FunctionTransformer(
                                lambda x: LabelEncoder().fit_transform(x.squeeze()),
                                validate=False,
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                [
                    col
                    for col in feature_pair
                    if not pd.api.types.is_numeric_dtype(X[col])
                ],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Step 3: Fit transformer and transform X
    X_transformed = preprocessor.fit_transform(X_pair)

    # Step 4: Encode y if needed
    if isinstance(y, pd.DataFrame):
        y = y.squeeze()  # convert (n,1) -> (n,)
    y_encoded = LabelEncoder().fit_transform(y)

    # Step 5: Train a fresh SVC
    svc = SVC(C=C, gamma=gamma, kernel=kernel, probability=True)
    svc.fit(X_transformed, y_encoded)

    # Step 6: Plotting
    plt.figure(figsize=figsize)
    x_min, x_max = X_transformed[:, 0].min() - 1, X_transformed[:, 0].max() + 1
    y_min, y_max = X_transformed[:, 1].min() - 1, X_transformed[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_density), np.linspace(y_min, y_max, grid_density)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = svc.decision_function(grid).reshape(xx.shape)

    plt.contourf(
        xx,
        yy,
        Z,
        levels=np.linspace(Z.min(), Z.max(), 10),
        cmap=plt.cm.coolwarm,
        alpha=0.6,
    )
    plt.contour(xx, yy, Z, colors="k", levels=[0], linestyles=["-"])

    if margin:
        # plot the ±1 margins
        plt.contour(
            xx,
            yy,
            Z,
            levels=[-1, 1],
            colors=["#0000FF", "#950714"],
            linestyles=["--", "--"],
            linewidths=2,
        )

    plt.scatter(
        X_transformed[:, 0],
        X_transformed[:, 1],
        c=y_encoded,
        cmap=plt.cm.coolwarm,
        edgecolors="k",
    )
    plt.scatter(
        svc.support_vectors_[:, 0],
        svc.support_vectors_[:, 1],
        s=100,
        facecolors="none",
        edgecolors="k",
        linewidth=1.5,
        label="Support Vectors",
    )

    plt.xlabel(feature_pair[0])
    plt.ylabel(feature_pair[1])
    plt.title(title or f"SVM Decision Boundary: {feature_pair[0]} vs {feature_pair[1]}")
    plt.legend(loc="best")
    plt.tight_layout()
    # Save plot if requested
    if image_path_png:
        plt.savefig(image_path_png, format="png", bbox_inches="tight")
    if image_path_svg:
        plt.savefig(image_path_svg, format="svg", bbox_inches="tight")
    plt.show()


################################################################################


def plot_svm_decision_surface_3d(
    X,
    y,
    feature_pair=("feature_1", "feature_2"),
    elev=30,
    azim=135,
    grid_density=50,
    C=100,
    gamma="auto",
    kernel="rbf",
    title=None,
    image_path_png=None,
    image_path_svg=None,
    figsize=(10, 8),
    equal_axes=True,
    zlim=None,
    show_support_vectors=True,
):
    # Step 1: Subset features
    X_pair = X[list(feature_pair)].copy()

    # Step 2: Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                [col for col in feature_pair if pd.api.types.is_numeric_dtype(X[col])],
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "label_enc",
                            FunctionTransformer(
                                lambda x: LabelEncoder().fit_transform(x.squeeze()),
                                validate=False,
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                [
                    col
                    for col in feature_pair
                    if not pd.api.types.is_numeric_dtype(X[col])
                ],
            ),
        ],
        verbose_feature_names_out=False,
    )

    X_transformed = preprocessor.fit_transform(X_pair)

    if isinstance(y, pd.DataFrame):
        y = y.squeeze()
    y_encoded = LabelEncoder().fit_transform(y)

    # Step 3: Train SVM
    svc = SVC(C=C, gamma=gamma, kernel=kernel)
    svc.fit(X_transformed, y_encoded)

    # Step 4: Meshgrid
    x_min, x_max = X_transformed[:, 0].min() - 1, X_transformed[:, 0].max() + 1
    y_min, y_max = X_transformed[:, 1].min() - 1, X_transformed[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_density),
        np.linspace(y_min, y_max, grid_density),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = svc.decision_function(grid).reshape(xx.shape)

    # Step 5: Plotting
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(xx, yy, Z, cmap="coolwarm", alpha=0.8, antialiased=True)

    # Data points
    ax.scatter(
        X_transformed[:, 0],
        X_transformed[:, 1],
        svc.decision_function(X_transformed),
        c=y_encoded,
        cmap="coolwarm",
        edgecolor="k",
        s=50,
        alpha=0.9,
    )

    # Optional: support vectors
    if show_support_vectors:
        ax.scatter(
            svc.support_vectors_[:, 0],
            svc.support_vectors_[:, 1],
            svc.decision_function(svc.support_vectors_),
            s=100,
            facecolors="none",
            edgecolors="k",
            linewidths=1.5,
            label="Support Vectors",
        )

    # Updated: Draw decision boundary at z=0
    ax.contour(
        xx,
        yy,
        Z,
        levels=[0],
        colors="k",
        linewidths=2,
        linestyles="dashed",
        offset=0,  # Now aligned with actual decision surface
    )

    # Labels and aesthetics
    ax.set_xlabel(feature_pair[0])
    ax.set_ylabel(feature_pair[1])
    ax.set_zlabel("decision fn")
    ax.set_title(title or "SVM Decision Surface")
    ax.view_init(elev=elev, azim=azim)

    if equal_axes:
        ax.set_box_aspect([1, 1, 1])

    if zlim:
        ax.set_zlim(zlim)

    fig.colorbar(surf, shrink=0.6, aspect=10, label="decision fn")

    if image_path_png:
        plt.savefig(image_path_png, format="png", bbox_inches="tight")
    if image_path_svg:
        plt.savefig(image_path_svg, format="svg", bbox_inches="tight")

    plt.show()


#################################################################################


def plot_svm_decision_surface_3d_plotly(
    X,
    y,
    feature_pair=("feature_1", "feature_2"),
    grid_density=50,
    C=100,
    gamma="auto",
    kernel="rbf",
    title="SVM Decision Surface",
    show_support_vectors=True,
    html_path=None,
):
    # PREPROCESS and TRAIN
    X_pair = X[list(feature_pair)]
    preproc = ColumnTransformer(
        [
            (
                "num",
                StandardScaler(),
                [c for c in feature_pair if pd.api.types.is_numeric_dtype(X[c])],
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "lbl",
                            FunctionTransformer(
                                lambda x: LabelEncoder().fit_transform(x.squeeze()),
                                validate=False,
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                [c for c in feature_pair if not pd.api.types.is_numeric_dtype(X[c])],
            ),
        ],
        verbose_feature_names_out=False,
    )

    Xt = preproc.fit_transform(X_pair)
    y_arr = LabelEncoder().fit_transform(
        y.squeeze() if isinstance(y, pd.DataFrame) else y
    )

    svc = SVC(C=C, gamma=gamma, kernel=kernel)
    svc.fit(Xt, y_arr)

    # MESH and DECISION FUNCTION
    x0_min, x0_max = Xt[:, 0].min() - 1, Xt[:, 0].max() + 1
    x1_min, x1_max = Xt[:, 1].min() - 1, Xt[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x0_min, x0_max, grid_density),
        np.linspace(x1_min, x1_max, grid_density),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = svc.decision_function(grid).reshape(xx.shape)

    # SURFACE TRACE
    surface = go.Surface(
        x=xx,
        y=yy,
        z=zz,
        colorscale="RdBu",
        reversescale=True,
        opacity=0.8,
        showscale=True,
        colorbar=dict(
            title="Decision f(x)",
            tickfont=dict(size=14),
            titlefont=dict(size=16),
            len=0.8,  # make it 60% of the plot height
            lenmode="fraction",  # interpret len as fraction of the plotting area
            y=0.5,  # center it vertically
            yanchor="middle",  # anchor the y position in the middle
        ),
        hovertemplate="Decision f(x): %{z:.2f}<extra></extra>",
        showlegend=False,
    )

    # DATA POINTS
    hover_fmt = (
        feature_pair[0]
        + ": %{x:.2f}<br>"
        + feature_pair[1]
        + ": %{y:.2f}<br>"
        + "Decision f(x): %{z:.2f}<extra></extra>"
    )
    pts3d = go.Scatter3d(
        x=Xt[:, 0],
        y=Xt[:, 1],
        z=svc.decision_function(Xt),
        mode="markers",
        marker=dict(
            size=5,
            color=y_arr,
            colorscale="RdBu",
            reversescale=True,
            line=dict(color="black", width=0.5),
            opacity=0.80,
        ),
        hovertemplate=hover_fmt,
        showlegend=False,
    )

    # SUPPORT VECTORS
    traces = [surface, pts3d]
    if show_support_vectors:
        sv = svc.support_vectors_
        sv_idx = svc.support_
        sv_dec = svc.decision_function(sv)
        sv_labels = y_arr[sv_idx]

        for lbl, color, name in [
            (0, "blue", "No Complications"),
            (1, "red", "Complications"),
        ]:
            mask = sv_labels == lbl
            sv_pts = sv[mask]
            sv_z = sv_dec[mask]
            traces.append(
                go.Scatter3d(
                    x=sv_pts[:, 0],
                    y=sv_pts[:, 1],
                    z=sv_z,
                    mode="markers",
                    marker=dict(
                        size=8,
                        symbol="circle-open",
                        color=color,
                        line=dict(color="black", width=2),
                        opacity=1,
                    ),
                    name=f"Support Vectors ({name})",
                    hovertemplate=f"SV: {name}<extra></extra>",
                )
            )

    # LAYOUT with the scene lifted
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=24)),
        autosize=True,
        height=950,
        template="plotly_white",
        legend=dict(
            orientation="h",
            x=0.5,
            y=0.95,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=18),
            itemsizing="constant",
        ),
        margin=dict(l=0, r=0, t=90, b=180),
        scene=dict(
            domain=dict(x=[0, 1], y=[0.1, 1]),
            xaxis=dict(
                title=dict(text=feature_pair[0], font=dict(size=18)),
                tickfont=dict(size=14),
                autorange="reversed",
                dtick=2,
            ),
            yaxis=dict(
                title=dict(text=feature_pair[1], font=dict(size=18)),
                tickfont=dict(size=14),
                autorange="reversed",
                dtick=1,
            ),
            zaxis=dict(
                title=dict(text="Decision f(x)", font=dict(size=18)),
                tickfont=dict(size=14),
            ),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=-1.7, z=1.2)),
        ),
    )

    config = {
        "responsive": True,
        "scrollZoom": False,  # disable zoom on mouse wheel
    }

    # when showing in a notebook or browser
    fig.show(config=config)

    if html_path:
        fig.write_html(
            html_path,
            include_plotlyjs="cdn",
            full_html=True,
            default_height=950,  # match layout.height
            default_width="100%",
            config=config,
        )
        print(f"Saved interactive plot to {html_path}")
