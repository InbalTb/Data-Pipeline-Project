import pandas as pd
from tqdm import tqdm
import math
import statistics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print_debug = True


# ---------------- Ingestion Methods ---------------------------#

def get_data_from_file():
    path_string = rf"C:/Users/רותם/loanapprovaldatasetbefore.csv"
    simulation_data = pd.read_csv(path_string, encoding="ISO-8859-1")
    return simulation_data


# ---------------- Transformation Methods ----------------------#

def transfer_data(data):
    ready_list = []
    for data_row in data.iterrows():
        ready_data_tuple = handle_data_row(data_row[1], data)
        if ready_data_tuple is not None:  # 5.1 Skip rows with missing values
            ready_list.append(ready_data_tuple)
    return ready_list


def handle_data_row(data_row, data):
    columns = list(data_row.index)  # Get the list of column names

    loan_id = create_int(data_item=data_row[columns[0]])
    no_of_dependents = validate_and_create_int_in_range(data_item=data_row[columns[1]], data_id=loan_id, min_val=0,
                                                        max_val=5, col_name=str(columns[1]), data=data)
    education = validate_education(data_row[columns[2]])
    self_employed = validate_self_employed(data_row[columns[3]])
    income_annum = validate_and_create_int_in_range(data_item=data_row[columns[4]], data_id=loan_id, min_val=200000,
                                                    max_val=9900000, col_name=str(columns[4]), data=data)
    loan_amount = validate_and_create_int_in_range(data_item=data_row[columns[5]], data_id=loan_id, min_val=300000,
                                                   max_val=39500000, col_name=str(columns[5]), data=data)
    loan_term = validate_and_create_int_in_range(data_item=data_row[columns[6]], data_id=loan_id, min_val=2, max_val=20,
                                                 col_name=str(columns[6]), data=data)
    cibil_score = validate_and_create_int_in_range(data_item=data_row[columns[7]], data_id=loan_id, min_val=300,
                                                   max_val=900, col_name=str(columns[7]), data=data)
    residential_assets_value = validate_and_create_int_in_range(data_item=data_row[columns[8]],
                                                                data_id=loan_id, min_val=-100000, max_val=29100000,
                                                                col_name=str(columns[8]), data=data)
    commercial_assets_value = validate_and_create_int_in_range(data_item=data_row[columns[9]],
                                                               data_id=loan_id, min_val=0, max_val=19400000,
                                                               col_name=str(columns[9]), data=data)
    luxury_assets_value = validate_and_create_int_in_range(data_item=data_row[columns[10]],
                                                           data_id=loan_id, min_val=300000, max_val=39200000,
                                                           col_name=str(columns[10]), data=data)
    bank_asset_value = validate_and_create_int_in_range(data_item=data_row[columns[11]],
                                                        data_id=loan_id, min_val=0, max_val=14700000,
                                                        col_name=str(columns[11]), data=data)
    loan_status = validate_loan_status(data_row[columns[12]])
    gender = validate_gender(data_row[columns[13]])
    city = create_category(data_row[columns[14]])

    # 5.1 - Binary variable and categorical variable - for these variables, if there is a missing value, delete the entire row.
    if any(value is None for value in (education, self_employed, loan_status, gender, city)):
        return None

    #in case type error or value error to numeric values- delete the row
    if any(value is None for value in (loan_id, no_of_dependents, income_annum,
        loan_amount, loan_term, cibil_score, residential_assets_value,
        commercial_assets_value, luxury_assets_value, bank_asset_value)):
        return None

    # -----------------6. Call the function to invert loan_amount------------------------#
    loan_amount_ln = invert_loan_amount(loan_amount)
    loan_amount_ln = validate_and_create_int_in_range(data_item=loan_amount_ln, data_id=loan_id,
                                                      min_val=math.log(30000), max_val=math.log(39500000))

    ready_data_tuple = (
        loan_id, no_of_dependents, education, self_employed, income_annum,
        loan_amount, loan_term, cibil_score, residential_assets_value,
        commercial_assets_value, luxury_assets_value, bank_asset_value,
        loan_status, gender, city, loan_amount_ln
    )

    return ready_data_tuple


def create_int(data_item):
    #in case there is no loan_id - it will also remove the line and show msg.
    try:
        data_item = int(data_item)
        return data_item
    except (ValueError, TypeError):
        if print_debug:
            print(f"Failed to convert data to int: {data_item}")
        return None


# -------------------------------4.2.2--------------------------------#
def validate_education(education):
    if not pd.isna(education):
        if education.strip() == 'Graduate':
            return 1
        elif education.strip() == 'Not Graduate':
            return 0
        else:
            if print_debug:
                print(f"Invalid education value received: {education}")
            return None
    else:
        return None

def validate_self_employed(self_employed):
    if not pd.isna(self_employed):
        if self_employed.strip() == 'Yes':
            return 1
        elif self_employed.strip() == 'No':
            return 0
        else:
            if print_debug:
                print(f"Invalid loan_status value received: {self_employed}")
            return None
    else:
        return None


def validate_loan_status(loan_status):
    if not pd.isna(loan_status):
        if loan_status.strip() == 'Approved':
            return 1
        elif loan_status.strip() == 'Rejected':
            return 0
        else:
            if print_debug:
                print(f"Invalid loan_status value received: {loan_status}")
            return None
    else:
        return None


def validate_gender(gender):
    if not pd.isna(gender):
        if gender.strip() == 'M':
            return 1
        elif gender.strip() == 'F':
            return 0
        else:
            if print_debug:
                print(f"Invalid gender value received: {gender}")
            return None
    else:
        return None


# ---------------------- 4.2.3,4.2.2  and 5.2. Continuous variable- will be completed with the average ------------------#
def validate_and_create_int_in_range(data_item, data_id=None, min_val=None, max_val=None, col_name=None, data=None):
    if not pd.isna(data_item):
        try:
            value = int(data_item)
            if min_val is not None and value < min_val:
                if print_debug:
                    print(f"Value below the minimum threshold for row {data_id}: {value}")
                return None
            if max_val is not None and value > max_val:
                if print_debug:
                    print(f"Value above the maximum threshold for row {data_id}: {value}")
                return None
            return value
        except (ValueError, TypeError):
            if print_debug:
                print(f"Failed to convert data to int: {data_item}")
            return None
    else:
        # Calculate the mean value of the column
        data_item = data[col_name].mean()
        return data_item


def create_category(data_item):
    category_dict = {
        'Miami': 1,
        'Albany': 2,
        'New York City': 3,
        'Los Angeles': 4,
        'Chicago': 5,
        'Houston': 6,
    }

    if data_item in category_dict:
        numeric_value = category_dict[data_item]
        return numeric_value

    else:
        if pd.isna(data_item):
            return None
        else:
            print(f"Invalid category value received: {data_item}")
            return None


# -------------------------6. invert_loan_amount--------------------------------#
def invert_loan_amount(loan_amount):
    if loan_amount > 0:
        return math.log(loan_amount)
    return None


# -----------------------the_percentage_of_rows_deleted-------------------------#
def the_percentage_of_rows_deleted(row_before, row_after):
    percentage = ((row_before - row_after) / row_before) * 100
    return percentage


# ------------------------------linear_regression-------------------------------#
def preprocess_data(df):
    # Feature selection: Choose relevant columns for X,
    # variables (features) that you believe are most relevant or influential in predicting the target variable.
    x = df[['income_annum', 'luxury_assets_value', 'commercial_assets_value', 'bank_asset_value']]
    # Target variable
    y = df['loan_amount']
    return x, y


def train_linear_regression(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Linear Regression model
    model = LinearRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2


# ----------------------------- Graph Types ----------------------------------- #

def create_graph_area(x, y):
    plt.stackplot(x, y, labels=["Area-Stack"], colors="grey", alpha=0.5)


def create_line_graph(x, y):
    plt.plot(x, y, label="Line Graph - avg of all cities", color='blue')


def create_bar_graph(x, y):
    plt.bar(x, y, label='Bar graph - avg of all cities', color='red', width=0.3)


def convert_to_dataframe(ready_list, column_names):
    df = pd.DataFrame(ready_list, columns=column_names)
    return df


# ----------------------------- Graph Axis Labels ----------------------------- #
def create_title_labels():
    # Loan Term is Loan Duration
    plt.xlabel("Loan Duration [Years]")
    plt.ylabel("Loan Amount [10^7]")


def create_title():
    plt.title("Average Requested Loan Amount vs. Loan Duration (Years) for All Cities")


# ------------------------- Graph Axis Limits --------------------------------- #
def create_axis_limits(x, y):
    x_min_value = min(x)
    x_max_value = max(x)
    y_min_value = min(y)
    y_max_value = max(y)
    plt.xlim([x_min_value, x_max_value])
    plt.ylim([y_min_value, y_max_value])


# ------------------------- Graph Vertical and Horizontal Lines --------------- #
def create_vertical_horizontal_line(x, y):
    average_x = statistics.mean(x)
    average_y = statistics.mean(y)

    plt.vlines(x=average_x, ymin=min(y), ymax=max(y), colors='black', ls=':', linewidth=2, label="V-Line", color="navy")
    plt.hlines(y=average_y, xmin=min(x), xmax=max(x), colors='black', ls='--', linewidth=2, label="H-Line",
               color="black")


# ------------------------- Graph Legend --------------------------------------- #
def create_legend():
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), ncol=7, shadow=True, fontsize=10)
    plt.subplots_adjust(bottom=0.2)


# ------------------------- Supporting Legend ---------------------------------- #
def shift_x(x, width):
    returned_list = []

    for number in x:
        number = number + width

        returned_list.append(number)

    return returned_list


def show_graph():
    plt.show()


# -----------------------  main  -----------------------#
def main():
    data = get_data_from_file()
    row_in_data_before = len(data)
    ready_data_list = transfer_data(data)
    row_in_data_after = len(ready_data_list)

    for data_tuple in tqdm(ready_data_list, desc="Processing Data", ncols=100):
        print(data_tuple)

    column_names = [
        'loan_id', 'no_of_dependents', 'education', 'self_employed',
        'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
        'residential_assets_value', 'commercial_assets_value',
        'luxury_assets_value', 'bank_asset_value', 'loan_status',
        'gender', 'city', 'loan_amount_ln'
    ]
    # Call the function with the correct column names
    df = convert_to_dataframe(ready_data_list, column_names)

    print("\nThe percentage of deleted rows is:", the_percentage_of_rows_deleted(row_in_data_before, row_in_data_after),
          "%\n")

    # linear_regression
    X, y = preprocess_data(df)
    model, X_test, y_test = train_linear_regression(X, y)
    mse, r2 = evaluate_model(model, X_test, y_test)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R2): {r2}")

    # graphs
    x_col = 'loan_term'
    y_col = "loan_amount"
    df = df.sort_values(by=[x_col])
    temp_df = df.groupby(x_col)['loan_amount'].mean().reset_index()

    # create graph
    create_graph_area(
        x=temp_df[x_col],
        y=temp_df[y_col]
    )

    create_line_graph(
        x=temp_df[x_col],
        y=temp_df[y_col],
    )

    create_bar_graph(
        x=temp_df[x_col],
        y=temp_df[y_col],
    )

    create_title_labels()

    create_axis_limits(x=df[x_col],
                       y=df[y_col])

    create_vertical_horizontal_line(x=df[x_col],
                                    y=df[y_col])

    create_legend()
    create_title()

    # show graph
    show_graph()


if __name__ == "__main__":
    main()
