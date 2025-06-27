import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


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
